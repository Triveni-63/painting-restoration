
"""Clean raw painting images and save a resumable manifest.

Features:
- Corrupt/small image filtering
- Fast perceptual-hash near-duplicate detection using a BK-tree
- Center cropping or tiled cropping
- SQLite checkpointing for interruption-safe resume
- Manifest regenerated from saved records
"""

import argparse
import sqlite3
from pathlib import Path

import pandas as pd
import imagehash
from PIL import Image, ImageFile, UnidentifiedImageError
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = False

EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".webp", ".tif", ".tiff"
}


def load_rgb(path):
    """Load a verified image and return an independent RGB image."""
    with Image.open(path) as im:
        im.verify()

    with Image.open(path) as im:
        im.load()
        return im.convert("RGB")


def to_tiles(im, size, mode, max_tiles):
    """Resize and return square crops."""
    w, h = im.size
    scale = size / min(w, h)

    im = im.resize(
        (max(size, round(w * scale)),
         max(size, round(h * scale))),
        Image.Resampling.LANCZOS
    )

    w, h = im.size

    if mode == "center":
        left = (w - size) // 2
        top = (h - size) // 2
        return [im.crop((left, top, left + size, top + size))]

    tiles = []
    n = max(1, min(max_tiles, max(w, h) // size))

    for i in range(n):
        offset = (
            0 if n == 1
            else round(i * (max(w, h) - size) / (n - 1))
        )

        if w >= h:
            box = (offset, 0, offset + size, size)
        else:
            box = (0, offset, size, offset + size)

        tiles.append(im.crop(box))

    return tiles


def phash_int(im):
    """Convert the 64-bit perceptual hash to an integer."""
    return int(str(imagehash.phash(im)), 16)


class BKTree:
    """BK-tree for searching perceptual hashes by Hamming distance."""

    def __init__(self):
        self.root = None

    @staticmethod
    def distance(a, b):
        return (a ^ b).bit_count()

    def add(self, value):
        if self.root is None:
            self.root = (value, {})
            return

        node = self.root

        while True:
            current, children = node
            distance = self.distance(value, current)

            if distance == 0:
                return

            if distance not in children:
                children[distance] = (value, {})
                return

            node = children[distance]

    def contains_near(self, value, threshold):
        if self.root is None:
            return False

        stack = [self.root]

        while stack:
            current, children = stack.pop()
            distance = self.distance(value, current)

            if distance <= threshold:
                return True

            low = distance - threshold
            high = distance + threshold

            for edge, child in children.items():
                if low <= edge <= high:
                    stack.append(child)

        return False


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--raw", default="data/raw")
    parser.add_argument("--out", default="data/processed")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument(
        "--mode", choices=["center", "tile"], default="center"
    )
    parser.add_argument("--max-tiles", type=int, default=3)
    parser.add_argument("--min-side", type=int, default=256)
    parser.add_argument("--phash-threshold", type=int, default=4)

    a = parser.parse_args()

    if a.size <= 0 or a.min_side < 0:
        parser.error("--size must be positive and --min-side nonnegative")

    if a.max_tiles < 1:
        parser.error("--max-tiles must be at least 1")

    if not 0 <= a.phash_threshold <= 64:
        parser.error("--phash-threshold must be between 0 and 64")

    raw = Path(a.raw)
    out = Path(a.out) / str(a.size)
    out.mkdir(parents=True, exist_ok=True)

    if not raw.is_dir():
        parser.error(f"Raw dataset folder not found: {raw}")

    artists = sorted(p.name for p in raw.iterdir() if p.is_dir())

    if not artists:
        parser.error(f"No artist folders found in {raw}")

    # Different settings use separate checkpoints.
    state_file = out / (
        f".clean_state_{a.mode}_min{a.min_side}"
        f"_ph{a.phash_threshold}_tiles{a.max_tiles}.sqlite3"
    )

    db = sqlite3.connect(state_file)

    db.execute("""
        CREATE TABLE IF NOT EXISTS processed (
            source TEXT PRIMARY KEY,
            artist TEXT NOT NULL,
            status TEXT NOT NULL,
            phash TEXT,
            orig_w INTEGER,
            orig_h INTEGER
        )
    """)

    db.execute("""
        CREATE TABLE IF NOT EXISTS outputs (
            file TEXT PRIMARY KEY,
            artist TEXT NOT NULL,
            source TEXT NOT NULL,
            orig_w INTEGER NOT NULL,
            orig_h INTEGER NOT NULL
        )
    """)

    db.commit()

    # Restore previously kept hashes for duplicate detection.
    tree = BKTree()

    for (saved_hash,) in db.execute(
        "SELECT DISTINCT phash FROM processed "
        "WHERE status = 'kept' AND phash IS NOT NULL"
    ):
        tree.add(int(saved_hash, 16))

    stats = {}

    try:
        for artist in artists:
            files = sorted(
                f for f in (raw / artist).rglob("*")
                if f.is_file() and f.suffix.lower() in EXTS
            )

            st = {
                "found": len(files),
                "resumed": 0,
                "corrupt": 0,
                "small": 0,
                "duplicate": 0,
                "kept": 0,
            }

            (out / artist).mkdir(parents=True, exist_ok=True)

            for f in tqdm(files, desc=artist):
                source = f.relative_to(raw).as_posix()

                previous = db.execute(
                    "SELECT status FROM processed WHERE source = ?",
                    (source,)
                ).fetchone()

                if previous:
                    status = previous[0]

                    # Reprocess a kept image if any expected output
                    # has been deleted.
                    if status == "kept":
                        saved_files = db.execute(
                            "SELECT file FROM outputs WHERE source = ?",
                            (source,)
                        ).fetchall()

                        if saved_files and all(
                            (out / item[0]).is_file()
                            for item in saved_files
                        ):
                            st["resumed"] += 1
                            continue

                        db.execute(
                            "DELETE FROM outputs WHERE source = ?",
                            (source,)
                        )
                        db.execute(
                            "DELETE FROM processed WHERE source = ?",
                            (source,)
                        )
                        db.commit()
                    else:
                        st["resumed"] += 1
                        continue

                try:
                    im = load_rgb(f)
                except (
                    UnidentifiedImageError,
                    OSError,
                    SyntaxError,
                    ValueError,
                ):
                    db.execute(
                        "INSERT OR REPLACE INTO processed "
                        "(source, artist, status) VALUES (?, ?, ?)",
                        (source, artist, "corrupt")
                    )
                    db.commit()
                    st["corrupt"] += 1
                    continue

                w, h = im.size

                if min(w, h) < a.min_side:
                    db.execute(
                        "INSERT OR REPLACE INTO processed "
                        "(source, artist, status, orig_w, orig_h) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (source, artist, "small", w, h)
                    )
                    db.commit()
                    st["small"] += 1
                    continue

                hash_value = phash_int(im)

                if tree.contains_near(hash_value, a.phash_threshold):
                    db.execute(
                        "INSERT OR REPLACE INTO processed "
                        "(source, artist, status, phash, orig_w, orig_h) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            source, artist, "duplicate",
                            f"{hash_value:016x}", w, h
                        )
                    )
                    db.commit()
                    st["duplicate"] += 1
                    continue

                # Save output crops before committing the checkpoint.
                tiles = to_tiles(
                    im, a.size, a.mode, a.max_tiles
                )

                output_rows = []

                for i, tile in enumerate(tiles):
                    name = f"{f.stem}_t{i}.jpg"
                    relative_output = (Path(artist) / name).as_posix()
                    destination = out / relative_output

                    tile.save(destination, format="JPEG", quality=95)

                    output_rows.append((
                        relative_output, artist, source, w, h
                    ))

                with db:
                    db.execute(
                        "INSERT OR REPLACE INTO processed "
                        "(source, artist, status, phash, orig_w, orig_h) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            source, artist, "kept",
                            f"{hash_value:016x}", w, h
                        )
                    )

                    db.executemany(
                        "INSERT OR REPLACE INTO outputs "
                        "(file, artist, source, orig_w, orig_h) "
                        "VALUES (?, ?, ?, ?, ?)",
                        output_rows
                    )

                tree.add(hash_value)
                st["kept"] += 1

            stats[artist] = st

    except KeyboardInterrupt:
        print("\nInterrupted. Saved progress is available for resuming.")

    finally:
        # Rebuild the manifest from committed database records.
        rows = db.execute(
            "SELECT file, artist, source, orig_w, orig_h "
            "FROM outputs ORDER BY artist, source, file"
        ).fetchall()

        columns = ["file", "artist", "source", "orig_w", "orig_h"]

        pd.DataFrame(rows, columns=columns).to_csv(
            out / "manifest.csv", index=False
        )

        db.close()

        if stats:
            print("\nCleaning report (images processed this run):")
            print(pd.DataFrame(stats).T.to_string())

        print(f"\nManifest: {out / 'manifest.csv'}")
        print(f"Checkpoint: {state_file}")
        print(f"Saved output crops: {len(rows)}")
        print("Run the same command again to resume.")


if __name__ == "__main__":
    main()
