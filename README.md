# Controllable Style-Aware Restoration of Damaged Paintings
Phase 1: data setup. See the steps in chat; commands:
```
python src/data/clean_dataset.py --size 256
python src/data/split_dataset.py --dir data/processed/256
python src/data/sanity_check.py  --dir data/processed/256
```
