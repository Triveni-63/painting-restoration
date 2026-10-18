"""PyTorch Dataset: loads clean crops from splits.csv and damages them ON THE FLY.
train -> new random damage every time; val/test -> deterministic damage (seed = index)
so metrics are reproducible."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from .damage import DamageConfig, apply_damage


class PaintingDamageDataset(Dataset):
    def __init__(self, root="data/processed/256", split="train", cfg=None,
                 artists=None, augment=None, seed=1234):
        self.root = Path(root)
        df = pd.read_csv(self.root / "splits.csv")
        self.artists = artists or sorted(df["artist"].unique())
        df = df[(df.split == split) & df.artist.isin(self.artists)].reset_index(drop=True)
        assert len(df), f"No rows for split={split} in {self.root / 'splits.csv'}"
        self.df, self.split, self.seed = df, split, seed
        self.cfg = cfg or DamageConfig()
        self.augment = (split == "train") if augment is None else augment
        self.a2i = {a: i for i, a in enumerate(self.artists)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        clean = np.asarray(Image.open(self.root / row.file).convert("RGB"), np.float32) / 255.0
        # train: fresh entropy each call; val/test: reproducible per index
        rng = np.random.default_rng() if self.split == "train" else np.random.default_rng(self.seed + i)
        if self.augment and rng.random() < 0.5:
            clean = clean[:, ::-1]                       # horizontal flip only (keeps style)
        clean = np.ascontiguousarray(clean)
        damaged, mask, info = apply_damage(clean, self.cfg, rng)
        to_t = lambda x: torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1)
        return dict(
            clean=to_t(clean) * 2 - 1,                   # [-1, 1]
            damaged=to_t(damaged) * 2 - 1,               # [-1, 1]
            mask=torch.from_numpy(mask)[None],           # [0, 1], 1 = damaged
            artist_id=torch.tensor(self.a2i[row.artist]),
            artist=row.artist, path=row.file, coverage=info["coverage"],
        )
