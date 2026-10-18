"""Step 3: class balance table + bar chart + sample grid per artist -> outputs/"""
import argparse, random
from pathlib import Path
import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/processed/512")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    d, out = Path(a.dir), Path(a.out); out.mkdir(exist_ok=True)
    m = pd.read_csv(d / "splits.csv")

    tab = pd.crosstab(m["artist"], m["split"])[["train", "val", "test"]]
    tab["total"] = tab.sum(axis=1)
    tab["share_%"] = (100 * tab["total"] / tab["total"].sum()).round(1)
    print(tab.to_string()); tab.to_csv(out / "class_balance.csv")

    ax = tab[["train", "val", "test"]].plot(kind="bar", stacked=True, figsize=(6, 4))
    ax.set_title("Crops per artist and split"); ax.set_ylabel("count"); plt.tight_layout()
    plt.savefig(out / "class_balance.png", dpi=150); plt.close()

    rnd = random.Random(a.seed)
    artists = sorted(m["artist"].unique())
    fig, axes = plt.subplots(len(artists), a.n, figsize=(2.2 * a.n, 2.4 * len(artists)), squeeze=False)
    for r, art in enumerate(artists):
        files = m[m.artist == art]["file"].tolist()
        pick = rnd.sample(files, min(a.n, len(files)))
        for c in range(a.n):
            ax = axes[r][c]; ax.axis("off")
            if c < len(pick):
                ax.imshow(Image.open(d / pick[c]))
            if c == 0:
                ax.set_title(art, loc="left", fontsize=10)
    plt.tight_layout(); plt.savefig(out / "samples_grid.png", dpi=150); plt.close()
    print(f"Saved {out/'class_balance.png'}, {out/'samples_grid.png'}, {out/'class_balance.csv'}")

if __name__ == "__main__":
    main()
