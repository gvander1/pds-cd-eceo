"""Step 0: extract frozen-encoder features at PATCH level and cache them on disk.

One file per dataset x encoder x split:   features/<dataset>/<encoder>/<split>.npz

Each row of the cache is one patch of the encoder grid (16 x 16 patches per 256 crop for
DINOv2 ViT-S/14 with resize_to=224). Arrays, N = n_samples * h * w rows, in the order
(sample, row, col):
    x, y        (N, C) float16   features of T1 and T2
    label       (N,)   uint8     1 if more than `label_thr` of the patch's pixels changed
    frac        (N,)   float16   fraction of changed pixels in the patch
    sample, row, col (N,) int32  where the patch comes from (crop index, grid position)
    lc1, lc2    (N,)   uint8     Hi-UCD only: majority land-cover class of the patch at T1 / T2
    im1, im2    (n_samples, H, W, 3) uint8   crops (only if save_images=True, default for val)
    change      (n_samples, H, W)    uint8   pixel change map (same condition)
    meta        JSON string: config used, grid size, number of samples

Crops are drawn exactly like evaluate.py does (no augmentation, torch RNG fixed with eval_seed,
same batch_size / num_workers), so the cached train/val patches are the pixels evaluate.py scores.
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import ROOT
from .datasets import build_dataset
from .model import DinoV2Encoder
from .utils import fixed_rng

FEATURES_DIR = ROOT / "features"


def cache_path(dataset, encoder, split, root=FEATURES_DIR):
    return Path(root) / dataset / encoder / f"{split}.npz"


def build_encoder(cfg, device):
    """Frozen encoder from the config. Step 3 (other encoders) only needs a new branch here."""
    if cfg["encoder"].startswith("dinov2"):
        return DinoV2Encoder(model_name=cfg["encoder"], freeze=True, resize_to=cfg["resize_to"]).to(device).eval()
    raise ValueError(f"Unknown encoder {cfg['encoder']!r}")


def _patch_majority(classes, grid_hw):
    """(B, H, W) int class map -> (B, h, w) most frequent class inside each patch."""
    n_cls = int(classes.max()) + 1
    onehot = F.one_hot(classes.long(), n_cls).permute(0, 3, 1, 2).float()   # (B, K, H, W)
    return F.adaptive_avg_pool2d(onehot, grid_hw).argmax(1)                  # (B, h, w)


@torch.no_grad()
def extract_split(cfg, split, device, save_images=None, label_thr=0.5):
    """Encode every crop of one split and return the patch-level arrays (see module docstring)."""
    save_images = (split != "train") if save_images is None else save_images
    ds = build_dataset(cfg, split, augment=False)
    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])
    enc = build_encoder(cfg, device)

    out = {k: [] for k in ("x", "y", "label", "frac", "lc1", "lc2", "im1", "im2", "change")}
    n_samples, grid_hw = 0, None
    with fixed_rng(cfg["eval_seed"]):  # same random Hi-UCD crops as evaluate.py
        for batch in loader:
            im1, im2, change = batch[0], batch[1], batch[2].float()
            x = enc(im1.to(device))                      # (B, C, h, w)
            y = enc(im2.to(device))
            grid_hw = tuple(x.shape[-2:])
            B, C = x.shape[:2]

            # (B, C, h, w) -> (B*h*w, C), rows ordered (sample, row, col)
            out["x"].append(x.permute(0, 2, 3, 1).reshape(-1, C).half().cpu().numpy())
            out["y"].append(y.permute(0, 2, 3, 1).reshape(-1, C).half().cpu().numpy())

            # the encoder grid covers the whole crop, so average-pooling the pixel map to h x w
            # gives the fraction of changed pixels inside each patch
            frac = F.adaptive_avg_pool2d(change.unsqueeze(1), grid_hw).reshape(-1)
            out["frac"].append(frac.half().numpy())
            out["label"].append((frac > label_thr).to(torch.uint8).numpy())

            if len(batch) >= 5:  # Hi-UCD: (im1, im2, change, lc1, lc2)
                out["lc1"].append(_patch_majority(batch[3], grid_hw).reshape(-1).to(torch.uint8).numpy())
                out["lc2"].append(_patch_majority(batch[4], grid_hw).reshape(-1).to(torch.uint8).numpy())

            if save_images:
                to_u8 = lambda t: (t.permute(0, 2, 3, 1) * 255).round().clamp(0, 255).to(torch.uint8).numpy()
                out["im1"].append(to_u8(im1))
                out["im2"].append(to_u8(im2))
                out["change"].append(change.to(torch.uint8).numpy())
            n_samples += B

    h, w = grid_hw
    arrays = {k: np.concatenate(v) for k, v in out.items() if v}
    arrays["sample"] = np.repeat(np.arange(n_samples, dtype=np.int32), h * w)
    arrays["row"] = np.tile(np.repeat(np.arange(h, dtype=np.int32), w), n_samples)
    arrays["col"] = np.tile(np.arange(w, dtype=np.int32), n_samples * h)
    meta = {"split": split, "n_samples": n_samples, "grid": [h, w], "label_thr": label_thr,
            "has_images": bool(save_images), "has_landcover": "lc1" in arrays,
            "cfg": {k: v for k, v in cfg.items() if isinstance(v, (str, int, float, bool))}}
    arrays["meta"] = np.array(json.dumps(meta))
    return arrays


def save_features(arrays, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)  # uncompressed: much faster to load, features don't compress well anyway


def load_features(dataset, encoder, split, root=FEATURES_DIR, with_images=True):
    """Returns a dict of numpy arrays + 'meta' (dict). with_images=False skips the big image arrays."""
    path = cache_path(dataset, encoder, split, root)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run  python scripts/extract_features.py --data {dataset}")
    with np.load(path) as f:
        keys = [k for k in f.files if with_images or k not in ("im1", "im2", "change")]
        d = {k: f[k] for k in keys}
    d["meta"] = json.loads(str(d["meta"]))
    return d


def to_grid(values, meta):
    """(N,) or (N, C) patch values -> (n_samples, h, w) or (n_samples, h, w, C), e.g. to draw score maps."""
    h, w = meta["grid"]
    return values.reshape(meta["n_samples"], h, w, *values.shape[1:])