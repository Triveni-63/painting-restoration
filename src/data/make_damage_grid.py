"""Unit test by eye: save a grid of 20 (clean | damaged | mask) samples -> outputs/damage_grid.png"""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.data.dataset import PaintingDamageDataset

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/processed/256")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default="outputs/damage_grid.png")
    a = ap.parse_args()
    ds = PaintingDamageDataset(a.root, a.split, augment=False)
    n = min(a.n, len(ds)); per_row = 2; rows = (n + per_row - 1) // per_row
    fig, ax = plt.subplots(rows, 3 * per_row, figsize=(3 * per_row * 2.2, rows * 2.4), squeeze=False)
    for ax_ in ax.flat: ax_.axis("off")
    step = max(1, len(ds) // n)
    for k in range(n):
        s = ds[(k * step) % len(ds)]
        r, c = divmod(k, per_row); c *= 3
        ax[r][c].imshow((s["clean"].permute(1, 2, 0) + 1) / 2);   ax[r][c].set_title(s["artist"], fontsize=8)
        ax[r][c+1].imshow((s["damaged"].permute(1, 2, 0) + 1) / 2); ax[r][c+1].set_title(f"damaged {s['coverage']:.0%}", fontsize=8)
        ax[r][c+2].imshow(s["mask"][0], cmap="gray", vmin=0, vmax=1); ax[r][c+2].set_title("mask", fontsize=8)
    plt.tight_layout(); Path(a.out).parent.mkdir(exist_ok=True)
    plt.savefig(a.out, dpi=130); print("saved", a.out)

if __name__ == "__main__":
    main()
