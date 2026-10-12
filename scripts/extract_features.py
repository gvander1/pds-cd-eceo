"""Step 0: encode train and val once and cache patch-level features.

    python scripts/extract_features.py --data levir
    python scripts/extract_features.py --data hiucd --splits val --overwrite
    python scripts/extract_features.py --data levir --encoder dinov2_vitb14     (step 3)

Writes features/<dataset>/<encoder>/<split>.npz (see src/pds_cd/features.py for the content).
Any key of the yaml configs can be overridden like in train.py.
"""
import argparse

from pds_cd.config import load_config
from pds_cd.features import cache_path, extract_split, save_features
from pds_cd.utils import get_device


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--splits", default="train,val", help="comma-separated, e.g. train,val")
    p.add_argument("--overwrite", action="store_true")
    args, rest = p.parse_known_args()
    cfg = load_config(rest)
    device = get_device()

    for split in args.splits.split(","):
        path = cache_path(cfg["dataset"], cfg["encoder"], split)
        if path.exists() and not args.overwrite:
            print(f"[skip] {path} exists (use --overwrite to recompute)")
            continue
        print(f"[{split}] encoding with {cfg['encoder']} on {device} ...")
        arrays = extract_split(cfg, split, device)
        save_features(arrays, path)
        m = arrays["label"]
        print(f"[{split}] {path}: {len(m)} patches from {int(arrays['sample'].max()) + 1} crops, "
              f"{m.mean():.2%} changed, size {path.stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()