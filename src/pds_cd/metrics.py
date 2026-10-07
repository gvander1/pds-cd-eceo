"""Error maps, threshold selection and scores.

Two change signals are always computed side by side:
  "model" : err = MSE over channels between transformer(encoder(T1)) and encoder(T2)
  "raw"   : err = MSE over channels between encoder(T1) and encoder(T2)   (no transformer)
Per-patch errors (16x16 grid) are upsampled bilinearly to the label resolution.
"""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

METHODS = ("model", "raw")


def error_map(pred, target, size):
    """(B, C, h, w) features -> (B, H, W) per-pixel error."""
    err = ((pred - target) ** 2).mean(dim=1, keepdim=True)
    return F.interpolate(err, size=size, mode="bilinear", align_corners=False).squeeze(1)


@torch.no_grad()
def collect_errors(loader, enc, tr, device, max_pixels=None, seed=0):
    """Run the whole loader once. Returns
        {"model": float32 array, "raw": float32 array, "labels": uint8 array, "feat_loss": float}
    feat_loss = mean MSE(transformer(x), y) in feature space (same quantity as the train loss).
    max_pixels: random subsample of pixels (same positions for both methods) to keep it fast.
    """
    enc.eval()
    tr.eval()
    errs = {m: [] for m in METHODS}
    labels, feat_loss, n_batches = [], 0.0, 0
    for batch in loader:
        im1, im2, cm = batch[0].to(device), batch[1].to(device), batch[2]
        x, y = enc(im1), enc(im2)
        x_hat = tr(x)
        feat_loss += F.mse_loss(x_hat, y).item()
        n_batches += 1
        errs["model"].append(error_map(x_hat, y, cm.shape[-2:]).flatten().cpu())
        errs["raw"].append(error_map(x, y, cm.shape[-2:]).flatten().cpu())
        labels.append(cm.flatten().to(torch.uint8))

    out = {m: torch.cat(v).numpy() for m, v in errs.items()}
    out["labels"] = torch.cat(labels).numpy()
    if max_pixels is not None and out["labels"].size > max_pixels:
        idx = np.random.default_rng(seed).choice(out["labels"].size, max_pixels, replace=False)
        out = {k: v[idx] for k, v in out.items()}
    out["feat_loss"] = feat_loss / max(n_batches, 1)
    return out


def separability(scores, labels):
    """Threshold-free metrics."""
    return {"auroc": roc_auc_score(labels, scores), "ap": average_precision_score(labels, scores)}


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
