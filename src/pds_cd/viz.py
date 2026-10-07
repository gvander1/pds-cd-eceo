"""Figures. Every function RETURNS the matplotlib figure: show it in a notebook with
plt.show(), or log it to W&B with wandb.Image(fig)."""
import random

import matplotlib.pyplot as plt
import numpy as np
import torch

from .metrics import error_map


@torch.no_grad()
def predict_samples(enc, tr, dataset, device, indices):
    """Returns im1, im2 (n,3,H,W), labels (n,H,W), err_model, err_raw (n,H,W) on CPU."""
    enc.eval()
    tr.eval()
    samples = [dataset[i] for i in indices]
    im1 = torch.stack([s[0] for s in samples]).to(device)
    im2 = torch.stack([s[1] for s in samples]).to(device)
    labels = torch.stack([s[2] for s in samples])
    x, y = enc(im1), enc(im2)
    err_model = error_map(tr(x), y, labels.shape[-2:]).cpu()
    err_raw = error_map(x, y, labels.shape[-2:]).cpu()
    return im1.cpu(), im2.cpu(), labels, err_model, err_raw


def pick_indices(dataset, n, seed=0):
    return random.Random(seed).sample(range(len(dataset)), min(n, len(dataset)))


def visualize_predictions(im1, im2, labels, err, threshold=None, vmin=None, vmax=None, indices=None, title=None):
    """Grid: T1 | T2 | true change | error | binary prediction (if threshold)."""
    n = len(im1)
    vmin = float(np.percentile(err, 1)) if vmin is None else vmin
    vmax = float(np.percentile(err, 99)) if vmax is None else vmax  # percentiles: outliers don't crush the scale
    ncols = 5 if threshold is not None else 4
    fig, axs = plt.subplots(n, ncols, figsize=(3.2 * ncols, 3.0 * n), constrained_layout=True, squeeze=False)
    titles = ["T1", "T2", "true change", "error"] + ([f"pred (thr={threshold:.2f})"] if threshold is not None else [])
    err_im = None
    for r in range(n):
        axs[r, 0].imshow(im1[r].permute(1, 2, 0))
        axs[r, 1].imshow(im2[r].permute(1, 2, 0))
        axs[r, 2].imshow(labels[r], cmap="gray", vmin=0, vmax=1)
        err_im = axs[r, 3].imshow(err[r], cmap="inferno", vmin=vmin, vmax=vmax)
        if threshold is not None:
            axs[r, 4].imshow(err[r] > threshold, cmap="gray", vmin=0, vmax=1)
        for c in range(ncols):
            axs[r, c].axis("off")
            if r == 0:
                axs[r, c].set_title(titles[c])
        if indices is not None:
            axs[r, 0].text(-0.1, 0.5, f"#{indices[r]}", transform=axs[r, 0].transAxes, ha="right", va="center")
    fig.colorbar(err_im, ax=axs[:, 3], shrink=0.6, label="MSE per patch")
    if title:
        fig.suptitle(title, fontsize=14)
    return fig


def error_histograms(errors, threshold=None, title=""):
    """errors: dict from metrics.collect_errors. One panel per method, classes overlaid."""
    lab = errors["labels"].astype(bool)
    methods = [m for m in ("model", "raw") if m in errors]
    fig, axs = plt.subplots(1, len(methods), figsize=(6 * len(methods), 4), constrained_layout=True, squeeze=False)
    for ax, m in zip(axs[0], methods):
        s = errors[m]
        bins = np.linspace(np.percentile(s, 0.1), np.percentile(s, 99.9), 101)  # same bins for both classes
        ax.hist(s[~lab], bins=bins, alpha=0.6, density=True, label="no change")
        ax.hist(s[lab], bins=bins, alpha=0.6, density=True, label="change")
        if threshold is not None and m in threshold:
            ax.axvline(threshold[m], color="k", ls="--", label=f"thr = {threshold[m]:.2f}")
        ax.set_yscale("log")
        ax.set_xlabel("error (MSE per patch, upsampled)")
        ax.set_title(f"{title} - {m}")
        ax.legend()
    return fig
