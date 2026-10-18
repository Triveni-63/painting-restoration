"""Run:  python -m unittest src.data.test_damage -v"""
import unittest
import numpy as np
from src.data.damage import apply_damage, DamageConfig, TYPES

def fake_painting(h=256, w=256, seed=0):
    r = np.random.default_rng(seed)
    import cv2
    x = cv2.resize(r.random((6, 6, 3)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    return np.clip(x, 0, 1).astype(np.float32)

class TestDamage(unittest.TestCase):
    def test_shapes_and_ranges(self):
        d, m, info = apply_damage(fake_painting(), rng=np.random.default_rng(0))
        self.assertEqual(d.shape, (256, 256, 3)); self.assertEqual(m.shape, (256, 256))
        self.assertTrue(0 <= d.min() and d.max() <= 1); self.assertTrue(0 <= m.min() and m.max() <= 1)

    def test_outside_mask_unchanged(self):
        for s in range(40):
            c = fake_painting(seed=s)
            d, m, _ = apply_damage(c, rng=np.random.default_rng(s))
            self.assertTrue(np.array_equal(d[m == 0], c[m == 0]), f"seed {s}")

    def test_each_type_works(self):
        for t in TYPES:
            c = fake_painting()
            d, m, info = apply_damage(c, rng=np.random.default_rng(1), types=[t], target_area=0.1)
            self.assertGreater(m.sum(), 0, t); self.assertGreater(np.abs(d - c).max(), 0.01, t)

    def test_severity_range(self):
        cov = np.array([apply_damage(fake_painting(seed=s), rng=np.random.default_rng(s))[2]["coverage"] for s in range(100)])
        print(f"\ncoverage mean={cov.mean():.2f} min={cov.min():.2f} max={cov.max():.2f}")
        self.assertGreater(((cov >= 0.03) & (cov <= 0.6)).mean(), 0.85)

    def test_reproducible(self):
        c = fake_painting()
        a = apply_damage(c, rng=np.random.default_rng(7))[0]; b = apply_damage(c, rng=np.random.default_rng(7))[0]
        self.assertTrue(np.array_equal(a, b))

    def test_varies(self):
        c = fake_painting()
        self.assertFalse(np.array_equal(apply_damage(c, rng=np.random.default_rng(1))[0], apply_damage(c, rng=np.random.default_rng(2))[0]))

    def test_binary_and_soft_masks(self):
        c = fake_painting()
        b = apply_damage(c, DamageConfig(soft_mask_prob=0.0), np.random.default_rng(3))[1]
        s = apply_damage(c, DamageConfig(soft_mask_prob=1.0), np.random.default_rng(3))[1]
        self.assertTrue(set(np.unique(b)) <= {0.0, 1.0}); self.assertGreater(len(np.unique(s)), 2)

if __name__ == "__main__":
    unittest.main()
