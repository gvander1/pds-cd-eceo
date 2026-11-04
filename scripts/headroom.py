"""How much of the T1 -> T2 feature difference can a predictor x -> x_hat remove, and does it help detection?

    python scripts/headroom.py --data hiucd --encoder dinov3_vits16

Uses the cached patch features (python scripts/extract_features.py --data ... --encoder ...).
Every predictor is fitted on TRAIN patches with no changed pixel (like the transformer), then scored on VAL:
    raw           x_hat = x                                   (copy T1)
    mean shift    x_hat = x + mean(y - x)                     (one global offset)
    affine        x_hat = a * x + b, per channel              (global colour/contrast-like change)
    linear        x_hat = W x + b, full C x C (ridge)         (best patch-wise linear predictor)
    crop shift    d = (y - x) - mean over the crop of (y - x) (no fitting, no labels: removes a per-crop offset)
Prints, for each: MSE on unchanged val patches (what the transformer minimises), and patch-level AUROC / AP
of the L2 and cosine scores (patch changed if > label_thr of its pixels changed, as in analyze_features.py).
If even the full linear predictor barely lowers the unchanged MSE, there is little predictable signal in T1 alone.
"""
import argparse

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from pds_cd.config import load_config
from pds_cd.features import load_features


def fit_predictors(x, y, ridge=1e-3):
    d = y - x
    shift = d.mean(0)
    xm, ym = x.mean(0), y.mean(0)
    xc, yc = x - xm, y - ym
    a = (xc * yc).mean(0) / (xc.var(0) + 1e-8)                     # per-channel least squares slope
    b = ym - a * xm
    cov = xc.T @ xc / len(x)
    W = np.linalg.solve(cov + ridge * np.trace(cov) / len(cov) * np.eye(len(cov)), xc.T @ yc / len(x))  # (C, C)
    return {"mean shift": lambda x_: x_ + shift,
            "affine": lambda x_: a * x_ + b,
            "linear": lambda x_: (x_ - xm) @ W + ym}


def scores(x_hat, y):
    l2 = ((y - x_hat) ** 2).mean(1)
    cos = 1 - (x_hat * y).sum(1) / (np.linalg.norm(x_hat, axis=1) * np.linalg.norm(y, axis=1) + 1e-8)
    return l2, cos


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ridge", type=float, default=1e-3)
    args, rest = p.parse_known_args()
    cfg = load_config(rest)
    tr = load_features(cfg["dataset"], cfg["encoder"], "train", with_images=False)
    va = load_features(cfg["dataset"], cfg["encoder"], "val", with_images=False)
    f32 = lambda a: a.astype(np.float64)

    unch_tr = tr["frac"] == 0
    x_tr, y_tr = f32(tr["x"][unch_tr]), f32(tr["y"][unch_tr])
    x_va, y_va = f32(va["x"]), f32(va["y"])
    lab, unch_va = va["label"].astype(bool), va["frac"] == 0
    print(f"{cfg['dataset']} | {cfg['encoder']} | C = {x_va.shape[1]}")
    print(f"train: {unch_tr.sum()} unchanged patches used for fitting | val: {len(lab)} patches, "
          f"{lab.mean():.2%} changed (label), {unch_va.mean():.2%} with no changed pixel")
    print(f"feature scale (val x): mean |x| {np.abs(x_va).mean():.3f}, std {x_va.std():.3f}, "
          f"max |x| {np.abs(x_va).max():.1f}, mean patch norm {np.linalg.norm(x_va, axis=1).mean():.2f}")

    x_hats = {"raw": x_va, **{k: f(x_va) for k, f in fit_predictors(x_tr, y_tr, args.ridge).items()}}
    # crop shift: per-crop offset removed from x, estimated on all the crop's patches (no labels)
    d = y_va - x_va
    crop_mean = np.zeros_like(d)
    for s in np.unique(va["sample"]):
        m = va["sample"] == s
        crop_mean[m] = d[m].mean(0)
    x_hats["crop shift"] = x_va + crop_mean

    rows = []
    for name, xh in x_hats.items():
        l2, cos = scores(xh, y_va)
        rows.append((name, l2[unch_va].mean(), roc_auc_score(lab, l2), average_precision_score(lab, l2),
                     roc_auc_score(lab, cos), average_precision_score(lab, cos)))
    raw_mse = rows[0][1]
    print(f"\n{'predictor':12s} {'MSE unch':>10s} {'vs raw':>8s} {'AUROC L2':>9s} {'AP L2':>7s} {'AUROC cos':>10s} {'AP cos':>7s}")
    for name, mse, a1, p1, a2, p2 in rows:
        print(f"{name:12s} {mse:10.5f} {100 * (mse / raw_mse - 1):+7.1f}% {a1:9.3f} {p1:7.3f} {a2:10.3f} {p2:7.3f}")

    if "lc1" in va:  # Hi-UCD: land-cover class 0 = unlabelled
        unl = (va["lc1"] == 0) | (va["lc2"] == 0)
        print(f"\nval patches whose majority land-cover class is 0 (unlabelled) at T1 or T2: {unl.mean():.2%}"
              f" | among them labelled changed: {lab[unl].mean():.2%} (vs {lab[~unl].mean():.2%} elsewhere)")
        if unl.any() and (~unl).any():
            l2 = scores(x_va, y_va)[0]
            print(f"raw L2 AUROC / AP without those patches: {roc_auc_score(lab[~unl], l2[~unl]):.3f} / "
                  f"{average_precision_score(lab[~unl], l2[~unl]):.3f}   | mean raw L2 on them (unchanged): {l2[unl & ~lab].mean():.4f} "
                  f"vs {l2[~unl & ~lab].mean():.4f} on other unchanged patches")


if __name__ == "__main__":
    main()