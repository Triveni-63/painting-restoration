"""Synthetic damage generator for painting restoration.

    damaged, mask, info = apply_damage(clean, cfg, rng)

clean   : float32 array (H, W, 3) in [0, 1]
damaged : float32 array (H, W, 3) in [0, 1]
mask    : float32 array (H, W)    1 = damaged / to be restored, 0 = intact
          (binary {0,1} if soft_mask=False, feathered in [0,1] otherwise)
info    : dict(types=[...], coverage=float)

Guarantee: damaged == clean wherever mask == 0 (so the model can trust the mask).
"""
from dataclasses import dataclass, field
import numpy as np
import cv2
from src.data.damage import DamageConfig

TYPES = ("patch", "crack", "discolor", "stain", "scratch")


@dataclass
class DamageConfig:
    min_area: float = 0.05            # target damaged fraction of the image
    max_area: float = 0.50
    probs: dict = field(default_factory=lambda: dict(
        patch=0.30, crack=0.25, discolor=0.20, stain=0.15, scratch=0.10))
    soft_mask_prob: float = 0.5       # chance of a feathered mask instead of binary
    max_components: int = 30


# ----------------------------------------------------------------- helpers
def _fbm(h, w, rng, base=4, octaves=4):
    """Cheap fractal value noise in [0, 1]."""
    out, amp, tot = np.zeros((h, w), np.float32), 1.0, 0.0
    for o in range(octaves):
        n = rng.random((base * 2 ** o + 1, base * 2 ** o + 1)).astype(np.float32)
        out += amp * cv2.resize(n, (w, h), interpolation=cv2.INTER_CUBIC)
        tot += amp
        amp *= 0.5
    out /= tot
    return np.clip((out - out.min()) / (out.max() - out.min() + 1e-8), 0, 1)


def _blob(h, w, area_frac, rng, feather=0.0):
    """Irregular blob alpha in [0,1] whose visible area (alpha>=0.05) is ~area_frac."""
    st, k = rng.bit_generator.state, 1.0
    for _ in range(4):                      # feathering spreads alpha: rescale to hit the budget
        rng.bit_generator.state = st
        m = _blob_once(h, w, area_frac * k, rng, feather)
        cov = float((m >= 0.05).mean())
        if cov <= area_frac * 1.1 or cov == 0:
            break
        k *= area_frac / cov
    return m


def _blob_once(h, w, area_frac, rng, feather=0.0):
    r = np.sqrt(max(area_frac, 1e-4) * h * w / np.pi)
    cx, cy = rng.uniform(0, w), rng.uniform(0, h)
    th = np.linspace(0, 2 * np.pi, 90, endpoint=False)
    rad = np.ones_like(th)
    for k in range(2, 7):                                   # random Fourier wobble
        rad += rng.uniform(0, 0.35 / k) * np.cos(k * th + rng.uniform(0, 2 * np.pi))
    rad *= r
    pts = np.stack([cx + rad * np.cos(th), cy + rad * np.sin(th)], 1).astype(np.int32)
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [pts], 255, lineType=cv2.LINE_AA)
    m = m.astype(np.float32) / 255.0
    sigma = max(0.8, feather * r)
    return np.clip(cv2.GaussianBlur(m, (0, 0), sigma), 0, 1)


def _canvas_texture(h, w, rng):
    base = np.array([0.86, 0.82, 0.72], np.float32) + rng.uniform(-0.05, 0.05, 3).astype(np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    weave = 0.012 * (np.sin(xx * 1.9) + np.sin(yy * 1.9))
    grain = rng.normal(0, 0.02, (h, w)).astype(np.float32)
    tex = base[None, None, :] + (weave + grain)[..., None]
    return np.clip(tex, 0, 1)


def _lines_alpha(h, w, rng, kind, budget, scale):
    """Draw cracks (branching random walks) or scratches (near-straight) until the
    dilated mask reaches ~budget of the image. Returns (alpha, mask) float32."""
    a = np.zeros((h, w), np.uint8)
    n_max = 60 if kind == "crack" else 25
    for _ in range(n_max):
        x, y = rng.uniform(0, w), rng.uniform(0, h)
        ang = rng.uniform(0, 2 * np.pi)
        if kind == "crack":
            stack = [(x, y, ang, int(rng.integers(40, 140)))]
            thick = max(1, int(round(scale * rng.choice([1, 1, 2]))))
        else:
            stack = [(x, y, ang, int(rng.integers(20, 90)))]
            thick = max(1, int(round(scale)))
        while stack:
            x, y, ang, steps = stack.pop()
            for _s in range(steps):
                ang += rng.normal(0, 0.35 if kind == "crack" else 0.03)
                nx, ny = x + 3 * scale * np.cos(ang), y + 3 * scale * np.sin(ang)
                cv2.line(a, (int(x), int(y)), (int(nx), int(ny)), 255, thick, cv2.LINE_AA)
                x, y = nx, ny
                if kind == "crack" and steps > 12 and rng.random() < 0.05:
                    stack.append((x, y, ang + rng.choice([-1, 1]) * rng.uniform(0.4, 1.2), steps // 2))
                if not (0 <= x < w and 0 <= y < h):
                    break
        k = max(3, int(2 * scale) | 1)
        if (cv2.dilate(a, np.ones((k, k), np.uint8)) > 0).mean() >= budget:
            break
    alpha = a.astype(np.float32) / 255.0
    k = max(3, int(2 * scale) | 1)
    mask = (cv2.dilate(a, np.ones((k, k), np.uint8)) > 0).astype(np.float32)
    return alpha, mask


# ----------------------------------------------------------------- damage types
# each returns (damaged_layer (H,W,3), alpha (H,W), mask (H,W)) given the CURRENT image
def _patch(img, rng, budget):
    h, w, _ = img.shape
    alpha = _blob(h, w, budget, rng, feather=0.02)
    fill = rng.choice(["white", "grey", "canvas", "canvas", "dark"])
    if fill == "white":
        layer = np.full_like(img, 0.97)
    elif fill == "grey":
        layer = np.full_like(img, rng.uniform(0.45, 0.7))
    elif fill == "dark":
        layer = np.full_like(img, rng.uniform(0.05, 0.2))
    else:
        layer = _canvas_texture(h, w, rng)
    return layer, alpha, alpha


def _crack(img, rng, budget):
    h, w, _ = img.shape
    scale = h / 256.0
    alpha, mask = _lines_alpha(h, w, rng, "crack", budget, scale)
    col = np.array([0.08, 0.06, 0.05]) if rng.random() < 0.7 else np.array([0.92, 0.9, 0.82])
    col = col + rng.uniform(-0.04, 0.04, 3)
    layer = np.broadcast_to(col.astype(np.float32), img.shape).copy()
    alpha = cv2.GaussianBlur(alpha, (0, 0), 0.5 * scale) * rng.uniform(0.7, 1.0)
    return layer, alpha, mask


def _scratch(img, rng, budget):
    h, w, _ = img.shape
    scale = h / 256.0
    alpha, mask = _lines_alpha(h, w, rng, "scratch", budget, scale)
    v = rng.choice([0.93, 0.12])
    layer = np.full_like(img, v)
    alpha = cv2.GaussianBlur(alpha, (0, 0), 0.4 * scale) * rng.uniform(0.6, 1.0)
    return layer, alpha, mask


def _discolor(img, rng, budget):
    h, w, _ = img.shape
    blob = _blob(h, w, budget, rng, feather=0.35)
    alpha = blob * (0.55 + 0.45 * _fbm(h, w, rng)) * rng.uniform(0.6, 1.0)
    mode = rng.choice(["yellow", "fade", "desat", "shift"])
    if mode == "yellow":
        layer = img * np.array([1.0, 0.92, 0.62], np.float32) + np.array([0.08, 0.05, 0.0], np.float32)
    elif mode == "fade":
        layer = img * 0.45 + 0.55 * 0.92
    elif mode == "desat":
        g = img.mean(2, keepdims=True)
        layer = 0.2 * img + 0.8 * g
    else:
        hsv = cv2.cvtColor(np.clip(img, 0, 1).astype(np.float32), cv2.COLOR_RGB2HSV)
        hsv[..., 0] = (hsv[..., 0] + rng.uniform(25, 90)) % 360
        layer = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return np.clip(layer, 0, 1).astype(np.float32), alpha, alpha


def _stain(img, rng, budget):
    h, w, _ = img.shape
    blob = _blob(h, w, budget, rng, feather=0.25)
    alpha = blob * (0.6 + 0.4 * _fbm(h, w, rng, base=6)) * rng.uniform(0.5, 0.9)
    tint = np.array([0.35, 0.22, 0.10], np.float32) * rng.uniform(0.7, 1.3)
    layer = np.clip(img * tint * 1.8, 0, 1).astype(np.float32)   # multiply-like brown stain
    return layer, alpha, alpha


_FN = dict(patch=_patch, crack=_crack, discolor=_discolor, stain=_stain, scratch=_scratch)


# ----------------------------------------------------------------- public API
def apply_damage(clean, cfg=None, rng=None, types=None, target_area=None):
    """Damage `clean` (float32 HxWx3 in [0,1]). `types` restricts to a subset
    (useful for tests); `target_area` overrides the sampled severity."""
    cfg = cfg or DamageConfig(min_area=0.10, max_area=0.40, soft_mask_prob=0.5,
                   probs=dict(patch=.35, crack=.25, discolor=.2, stain=.1, scratch=.1))
    rng = rng or np.random.default_rng()
    clean = np.asarray(clean, np.float32)
    h, w, _ = clean.shape
    soft = rng.random() < cfg.soft_mask_prob
    names = list(types or TYPES)
    p = np.array([cfg.probs[t] for t in names], np.float64); p /= p.sum()
    target = target_area if target_area is not None else rng.uniform(cfg.min_area, cfg.max_area)

    img = clean.copy()
    mask = np.zeros((h, w), np.float32)
    used = []
    for _ in range(cfg.max_components):
        cov = float((mask > 0).mean())
        if cov >= target:
            break
        t = str(rng.choice(names, p=p))
        budget = (target - cov) * rng.uniform(0.4, 1.0)
        layer, alpha, m = _FN[t](img, rng, budget)
        alpha = np.where(alpha < 0.05, 0.0, alpha).astype(np.float32)   # drop faint tails
        m = np.where(m < 0.05, 0.0, m).astype(np.float32)               # same tail cut for the mask
        m = np.maximum(m, (alpha > 0).astype(np.float32))
        if (np.maximum(mask, m) > 0).mean() > cfg.max_area * 1.15:      # overshoot -> skip
            continue
        img = img * (1 - alpha[..., None]) + layer * alpha[..., None]
        mask = np.maximum(mask, m)
        used.append(t)

    img = np.clip(img, 0, 1).astype(np.float32)
    if soft:       # feathered declared mask (superset of the visual damage)
        mask = np.clip(cv2.GaussianBlur((mask > 0).astype(np.float32), (0, 0), 1.0 * h / 256.0) * 1.3, 0, 1)
        mask = np.where(mask < 0.05, 0.0, mask).astype(np.float32)
        mask = np.maximum(mask, (np.abs(img - clean).max(2) > 0).astype(np.float32))
    else:
        mask = (mask > 0).astype(np.float32)
    img = np.where(mask[..., None] > 0, img, clean)                     # hard guarantee
    return img, mask, dict(types=used, coverage=float((mask > 0).mean()), soft=bool(soft))
