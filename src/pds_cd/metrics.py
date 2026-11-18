"""Error maps, threshold selection and scores.

Two change signals are always computed side by side:
  "model" : err = MSE over channels between transformer(encoder(T1)) and encoder(T2)
  "raw"   : err = MSE over channels between encoder(T1) and encoder(T2)   (no transformer)
Per-patch errors (16x16 grid) are upsampled bilinearly to the label resolution.
"""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from torchmetrics.functional.classification import binary_auroc, binary_average_precision
from .model import split_tiles
METHODS = ("model", "raw")


def error_map(pred, target, size):
    """(B, C, h, w) features -> (B, H, W) per-pixel error."""
    err = ((pred - target) ** 2).mean(dim=1, keepdim=True)
    return F.interpolate(err, size=size, mode="bilinear", align_corners=False).squeeze(1)


@torch.no_grad()
def collect_errors(loader, enc, tr, device, max_pixels=None, seed=0, on_device=False):
    """Run the whole loader once. Returns
        {"model": float32 array, "raw": float32 array, "labels": uint8 array,
         "feat_loss_unchanged": float, "feat_loss_unchanged_raw": float}
    model / raw / labels: per-pixel errors and labels on ALL pixels (needed for AUROC, AP, F1).
    feat_loss_unchanged(_raw): mean feature MSE of x_hat (resp. x, copy T1) vs y, only over the patches
    with no changed pixel -> how well the model predicts "normal" T2.
    max_pixels: random subsample of pixels (same positions for both methods) to keep it fast.
    """
    enc.eval()
    tr.eval()
    errs = {m: [] for m in METHODS}
    labels = []
    unch_sum, unch_n = {m: 0.0 for m in METHODS}, 0.0   # sum of per-patch MSE / number of unchanged patches
    for batch in loader:
        im1, im2, cm = batch[0].to(device), batch[1].to(device), batch[2]
        x, y = enc(im1), enc(im2)
        cm = split_tiles(cm, getattr(enc, "tiles", 1))   # full-image encoders return one sample per tile
        x_hat = tr(x)
        frac = F.adaptive_avg_pool2d(cm.to(device).float().unsqueeze(1), y.shape[-2:])   # (B, 1, h, w) changed fraction
        keep = (frac == 0).float()                                                       # unchanged patches
        for m, pred in (("model", x_hat), ("raw", x)):
            unch_sum[m] += (((pred - y) ** 2).mean(dim=1, keepdim=True) * keep).sum().item()
            e = error_map(pred, y, cm.shape[-2:]).flatten()
            errs[m].append(e if on_device else e.cpu())
        unch_n += keep.sum().item()
        labels.append((cm.to(device) if on_device else cm).flatten().to(torch.uint8))

    if on_device:
        out = {m: torch.cat(v) for m, v in errs.items()}
        out["labels"] = torch.cat(labels)
    else:
        out = {m: torch.cat(v).numpy() for m, v in errs.items()}
        out["labels"] = torch.cat(labels).numpy()
    if not on_device and max_pixels is not None and out["labels"].size > max_pixels:
        idx = np.random.default_rng(seed).choice(out["labels"].size, max_pixels, replace=False)
        out = {k: v[idx] for k, v in out.items()}
    out["feat_loss_unchanged"] = unch_sum["model"] / max(unch_n, 1)
    out["feat_loss_unchanged_raw"] = unch_sum["raw"] / max(unch_n, 1)
    return out

def separability(scores, labels):
    """Threshold-free metrics."""
    return {"auroc": roc_auc_score(labels, scores), "ap": average_precision_score(labels, scores)}

def separability_torch(scores, labels):
    """Same as separability (identical values, checked against sklearn) but on torch tensors, on the GPU."""
    labels = labels.long()
    return {"auroc": binary_auroc(scores, labels).item(), "ap": binary_average_precision(scores, labels).item()}

def best_f1_threshold(scores, labels):
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    best = int(f1.argmax())
    return float(thresholds[best]), float(f1[best])


def scores_at_threshold(scores, labels, thr):
    pred = scores > thr
    gt = labels.astype(bool)
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"f1": f1, "precision": precision, "recall": recall,
            "pred_pos_frac": float(pred.mean()), "true_pos_frac": float(gt.mean())}


def _thin(x, y, n=2000):
    """Keep only the curve points where x or y moves by >= 1/n (a curve on millions of pixels
    has millions of points: too heavy to plot, and invisible at figure resolution)."""
    key = np.floor(x * n).astype(np.int64) * (n + 1) + np.floor(y * n).astype(np.int64)
    keep = np.r_[True, key[1:] != key[:-1]]
    keep[-1] = True
    return x[keep], y[keep]


def pr_roc_curves(scores, labels):
    """PR and ROC curves for plotting (thinned). AUPRC = AP (step-wise area under the PR curve,
    sklearn average_precision_score), AUROC on all pixels.
    Also returns the best-F1 point ON THIS SPLIT (oracle threshold, for reference only)."""
    precision, recall, thr = precision_recall_curve(labels, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    best = int(f1.argmax())
    fpr, tpr, _ = roc_curve(labels, scores)
    out = {"auprc": average_precision_score(labels, scores), "auroc": roc_auc_score(labels, scores),
           "prevalence": float(np.mean(labels)),
           "oracle": {"threshold": float(thr[best]), "f1": float(f1[best]),
                      "precision": float(precision[best]), "recall": float(recall[best])}}
    out["recall"], out["precision"] = _thin(recall, precision)
    out["fpr"], out["tpr"] = _thin(fpr, tpr)
    return out