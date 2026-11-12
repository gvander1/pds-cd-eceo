"""Bug hunt: checks every stage of the pipeline on the real val data, one by one.

    python scripts/sanity_checks.py --data hiucd --encoder dinov3_vits16
    python scripts/sanity_checks.py --data hiucd --encoder dinov3_vits16 --ckpt checkpoints/<run>.pt

Each check prints PASS / FAIL / INFO with the numbers behind it.
  1. labels         : values of the change map, fraction changed, fraction of fully unchanged crops
  2. image pairing  : T1/T2/label of a crop come from the same place (pixel difference is larger on changes)
  3. grid orientation: the 16x16 feature grid is aligned with the label grid (no transpose / flip / shift)
  4. metric pipeline: an oracle predictor gets near-perfect AP through error_map + separability
  5. zero start     : a fresh transformer gives exactly x_hat = x
  6. training mask  : unchanged_crops() keeps exactly the crops with no changed pixel
  7. trained model  : (with --ckpt) how far x_hat moves from x, and in which direction
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from pds_cd.config import load_config
from pds_cd.datasets import build_dataset
from pds_cd.metrics import error_map, separability
from pds_cd.model import FeatureTransformer, build_models
from pds_cd.utils import fixed_rng, get_device

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import unchanged_crops  # noqa: E402  (the function the training really uses)


def report(ok, name, msg):
    tag = {True: "PASS", False: "FAIL", None: "INFO"}[ok]
    print(f"[{tag}] {name}: {msg}")


def auroc_ap(score, lab):
    return roc_auc_score(lab, score), average_precision_score(lab, score)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=None)
    args, rest = p.parse_known_args()
    cfg = load_config(rest)
    device = get_device()
    enc, tr = build_models(cfg, device)
    if args.ckpt:
        ck = torch.load(args.ckpt, map_location=device)
        tr.load_state_dict(ck["transformer"])
    enc.eval(); tr.eval()

    ds = build_dataset(cfg, "val")
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=4)   # same as train.py / evaluate.py
    X, Y, XH, IM1, IM2, LAB = [], [], [], [], [], []
    with fixed_rng(cfg["eval_seed"]):                                     # same val crops as training
        for b in loader:
            im1, im2, cm = b[0].to(device), b[1].to(device), b[2]
            x, y = enc(im1), enc(im2)
            X.append(x.cpu()); Y.append(y.cpu()); XH.append(tr(x).cpu())
            IM1.append(im1.cpu()); IM2.append(im2.cpu()); LAB.append(cm)
    x, y, xh = torch.cat(X), torch.cat(Y), torch.cat(XH)
    im1, im2, lab = torch.cat(IM1), torch.cat(IM2), torch.cat(LAB)
    n, C, h, w = x.shape
    H, W = lab.shape[-2:]
    print(f"val: {n} crops {H}x{W}, features {C} x {h}x{w}, device {device}\n")

    # 1. labels -----------------------------------------------------------------------------------
    vals = torch.unique(lab).tolist()
    report(set(vals) <= {0, 1}, "labels", f"values {vals}, changed pixels {lab.float().mean():.2%}, "
           f"crops with no changed pixel {(lab.flatten(1).amax(1) == 0).float().mean():.2%}")

    # 2. image pairing: if T1/T2/label came from different places, the pixel difference would not
    #    be larger on changed pixels (AUROC ~ 0.5)
    d_pix = (im1 - im2).abs().mean(1)                                       # (n, H, W)
    a, _ = auroc_ap(d_pix.flatten().numpy(), lab.flatten().numpy())
    report(a > 0.6, "image pairing", f"AUROC of raw RGB difference |T1 - T2| vs labels = {a:.3f} (0.5 = no link)")

    # 3. grid orientation: raw feature error on the 16x16 grid vs patch labels under every transform.
    #    The identity must be clearly the best, otherwise a reshape / permute mixes up rows and columns.
    err = ((y - x) ** 2).mean(1)                                            # (n, h, w)
    frac = F.adaptive_avg_pool2d(lab.float().unsqueeze(1), (h, w)).squeeze(1)
    plab = (frac > 0.5).numpy().ravel()
    transforms = {
        "identity": lambda t: t, "transpose": lambda t: t.transpose(-1, -2),
        "flip rows": lambda t: t.flip(-2), "flip cols": lambda t: t.flip(-1), "rot180": lambda t: t.flip(-1, -2),
        "shift +1 row": lambda t: t.roll(1, -2), "shift +1 col": lambda t: t.roll(1, -1),
    }
    res = {k: auroc_ap(f(err).numpy().ravel(), plab)[0] for k, f in transforms.items()}
    best = max(res, key=res.get)
    report(best == "identity" and res["identity"] - max(v for k, v in res.items() if k != "identity") > 0.02,
           "grid orientation", "patch AUROC " + ", ".join(f"{k} {v:.3f}" for k, v in res.items()))

    # 4. metric pipeline: oracle x_hat = y on patches without change, x elsewhere -> error only on
    #    changed patches. Through the same error_map + separability as training, AP must be high.
    unch = (frac == 0).unsqueeze(1)                                         # (n, 1, h, w)
    oracle = torch.where(unch, y, x)
    e_or = error_map(oracle, y, (H, W)).flatten().numpy()
    s = separability(e_or, lab.flatten().numpy().astype(np.uint8))
    report(s["ap"] > 0.8, "metric pipeline", f"oracle AUROC {s['auroc']:.3f}, AP {s['ap']:.3f} "
           f"(below 1 only because of 16x16 patches upsampled to pixels)")
    s_raw = separability(error_map(x, y, (H, W)).flatten().numpy(), lab.flatten().numpy().astype(np.uint8))
    report(None, "raw baseline", f"pixel AUROC {s_raw['auroc']:.4f}, AP {s_raw['ap']:.4f} (W&B val/*_raw should match)")

    # 5. zero start: a freshly built transformer must output exactly x
    fresh = FeatureTransformer(embed_dim=C, n_heads=cfg["n_heads"], n_encoder_layers=cfg["n_encoder_layers"],
                               n_decoder_layers=cfg["n_decoder_layers"], dropout=cfg["dropout"],
                               n_tokens=h * w).eval()
    xs = x[:4]
    report(torch.equal(fresh(xs), xs), "zero start", f"max |x_hat - x| = {(fresh(xs) - xs).abs().max():.2e}")

    # 6. training mask: unchanged_crops (used in train.py) vs the definition "no changed pixel"
    keep = unchanged_crops(lab)
    ref = lab.flatten(1).sum(1) == 0
    report(torch.equal(keep, ref), "training mask", f"{int(keep.sum())}/{n} crops kept, definition agrees: {torch.equal(keep, ref)}")

    # 7. trained model
    if args.ckpt:
        corr = xh - x
        target = y - x
        m = unch.expand_as(x)                                               # unchanged patches only
        cos = F.cosine_similarity(corr.permute(0, 2, 3, 1).reshape(-1, C), target.permute(0, 2, 3, 1).reshape(-1, C), dim=1)
        cos_u = cos[unch.flatten()].mean().item()
        cos_c = cos[(frac > 0.5).flatten()].mean().item()
        rel = (corr.norm(dim=1).mean() / target.norm(dim=1).mean()).item()
        mse_raw, mse_mod = (target[m] ** 2).mean().item(), ((y - xh)[m] ** 2).mean().item()
        report(None, "trained model", f"|correction| / |y - x| = {rel:.2f}; cosine(correction, y - x): "
               f"unchanged patches {cos_u:.3f}, changed patches {cos_c:.3f}; unchanged MSE {mse_mod:.4f} vs raw {mse_raw:.4f}")
        s_mod = separability(error_map(xh, y, (H, W)).flatten().numpy(), lab.flatten().numpy().astype(np.uint8))
        report(None, "trained model", f"pixel AUROC {s_mod['auroc']:.4f}, AP {s_mod['ap']:.4f} (W&B val/*_model should match)")


if __name__ == "__main__":
    main()

