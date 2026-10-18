"""Step 2: 80/10/10 split BY SOURCE IMAGE, stratified by artist -> splits.csv"""
import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/processed/512")
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    d = Path(a.dir)
    m = pd.read_csv(d / "manifest.csv")
    src = m[["source", "artist"]].drop_duplicates()
    trainval, test = train_test_split(src, test_size=a.test, stratify=src["artist"], random_state=a.seed)
    train, val = train_test_split(trainval, test_size=a.val / (1 - a.test),
                                  stratify=trainval["artist"], random_state=a.seed)
    lut = {**{s: "train" for s in train.source}, **{s: "val" for s in val.source}, **{s: "test" for s in test.source}}
    m["split"] = m["source"].map(lut)

    # leakage check: no source image may appear in more than one split
    assert (m.groupby("source")["split"].nunique() == 1).all(), "LEAKAGE: a source is in 2 splits"
    m.to_csv(d / "splits.csv", index=False)
    print(pd.crosstab(m["artist"], m["split"], margins=True))
    print(f"\nLeakage check passed. Wrote {d / 'splits.csv'}")

if __name__ == "__main__":
    main()
