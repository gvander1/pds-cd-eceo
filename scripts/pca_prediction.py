"""PCA view of what the transformer predicts: T1 features (x), predicted T2 (x_hat = transformer(x)) and
actual T2 features (y), for the same val patches, in one PCA plane.

    python scripts/pca_prediction.py --ckpt checkpoints/hiucd-dinov2_vits14-seed0-713e9d4f.pt
    python scripts/pca_prediction.py --ckpt checkpoints/levir-dinov3_vitl16-seed0-1a7ec0f8.pt --samples 0,5,12

The PCA is fitted on TRAIN x + y only (no labels, no x_hat), so the axes are the same as in
pair_pca_figure / pca_rgb_figure and x_hat is projected into the encoder's own feature space.
Writes 2 figures to results/<dataset>/<encoder>/:
    pca_pred_scatter_<run>.png   PC1-PC2 scatter, unchanged | changed patches
    pca_pred_rgb_<run>.png       per crop: T1 | T2 | PCA->RGB of x, x_hat, y | change label
and prints distances computed in the FULL feature space (2D PCA alone can mislead).
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

from pds_cd.config import ROOT
from pds_cd.datasets import build_dataset
from pds_cd.model import build_models
from pds_cd.utils import fixed_rng, get_device


@torch.no_grad()
def collect(cfg, split, enc, tr, device, label_thr=0.5, keep_images=False):
    """Patch-level x, x_hat, y (N, C) float16 + label (N,), rows ordered (sample, row, col).
    Crops drawn like evaluate.py / extract_features.py: no augmentation, torch RNG fixed with eval_seed."""
    ds = build_dataset(cfg, split, augment=False)
    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])
    enc.eval()
    tr.eval()
    out = {k: [] for k in ("x", "x_hat", "y", "label", "im1", "im2", "change")}
    with fixed_rng(cfg["eval_seed"]):
        for batch in loader:
            im1, im2, change = batch[0].to(device), batch[1].to(device), batch[2].float()
            x, y = enc(im1), enc(im2)                     # (B, C, h, w)
            x_hat = tr(x)                                 # (B, C, h, w), predicted T2
            C, grid = x.shape[1], tuple(x.shape[-2:])
            for k, t in (("x", x), ("x_hat", x_hat), ("y", y)):
                out[k].append(t.permute(0, 2, 3, 1).reshape(-1, C).half().cpu().numpy())
            frac = F.adaptive_avg_pool2d(change.unsqueeze(1), grid).reshape(-1)   # fraction of changed pixels
            out["label"].append((frac > label_thr).to(torch.uint8).numpy())
            if keep_images:
                to_u8 = lambda t: (t.permute(0, 2, 3, 1) * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
                out["im1"].append(to_u8(im1))
                out["im2"].append(to_u8(im2))
                out["change"].append(change.to(torch.uint8).numpy())
    d = {k: np.concatenate(v) for k, v in out.items() if v}
    d["grid"] = grid
    return d


def fit_pca(tr_d, n_comp, max_fit=50_000, seed=0):
    """PCA on train T1 + T2 features (same subsample of patches for both dates)."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(tr_d["x"]), min(max_fit, len(tr_d["x"])), replace=False)
    fit = np.concatenate([tr_d["x"][idx], tr_d["y"][idx]]).astype(np.float32)
    return PCA(n_comp, random_state=seed).fit(fit)


def scatter_figure(pca, d, n_pairs=1500, n_lines=100, seed=0, title=""):
    """Rows: no-change patches | changed patches. Columns, for the SAME patches and the SAME PCA axes:
        1 = T1 (x, blue)              vs actual T2 (y, orange)        -> what really changed between the dates
        2 = predicted T2 (x_hat, green) vs actual T2 (y, orange)      -> how close the transformer gets to T2
        3 = T1 (x, blue)              vs predicted T2 (x_hat, green)  -> what the transformer does to T1
    A grey line joins the two points of the same patch for `n_lines` patches."""
    evr = pca.explained_variance_ratio_
    rng = np.random.default_rng(seed)
    rows = []
    for lab, name in ((0, "no change"), (1, "change")):
        pool = np.where(d["label"] == lab)[0]
        idx = rng.choice(pool, min(n_pairs, len(pool)), replace=False)   # same patches in every column
        P = {k: pca.transform(d[k][idx].astype(np.float32))[:, :2] for k in ("x", "x_hat", "y")}
        rows.append((name, len(pool), P))
    allp = np.concatenate([p for _, _, P in rows for p in P.values()])
    lo, hi = np.percentile(allp, 0.5, axis=0), np.percentile(allp, 99.5, axis=0)
    pad = 0.05 * (hi - lo)

    style = {"x": ("tab:blue", "T1  x"), "x_hat": ("tab:green", "T2 predicted  x̂"), "y": ("tab:orange", "T2 actual  y")}
    cols = [("x", "y", "T1 vs actual T2"),                 # (first point, second point, column title)
            ("x_hat", "y", "predicted T2 vs actual T2"),
            ("x", "x_hat", "T1 vs predicted T2")]
    fig, axs = plt.subplots(2, 3, figsize=(19, 12), constrained_layout=True, sharex=True, sharey=True)
    for r, (name, n_tot, P) in enumerate(rows):
        k = min(n_lines, len(P["y"]))
        for c, (ka, kb, col_title) in enumerate(cols):
            ax = axs[r, c]
            for a, b in zip(P[ka][:k], P[kb][:k]):
                ax.plot([a[0], b[0]], [a[1], b[1]], color="grey", lw=0.5, alpha=0.6, zorder=1)
            for key, z in ((kb, 2), (ka, 3)):
                ax.scatter(*P[key].T, s=4, alpha=0.45, color=style[key][0], label=style[key][1], zorder=z)
            dist = np.linalg.norm(P[ka] - P[kb], axis=1).mean()
            ax.set_title(f"{col_title} | {name}: {len(P['y'])} of {n_tot} val patches\n"
                         f"mean distance between the two in this plane {dist:.2f}", fontsize=10)
            ax.set(xlim=(lo[0] - pad[0], hi[0] + pad[0]), ylim=(lo[1] - pad[1], hi[1] + pad[1]))
            ax.legend(markerscale=4, loc="best")
        axs[r, 0].set_ylabel(f"PC2 ({evr[1]:.0%} of x+y variance)")
    for c in range(3):
        axs[1, c].set_xlabel(f"PC1 ({evr[0]:.0%} of x+y variance)")
    fig.suptitle(title + "\nsame patches and same PCA axes in all panels; grey line joins the two points of one patch")
    return fig


def rgb_figure(pca3, tr_d, d, samples, title=""):
    """Per crop: T1 | T2 | PCA->RGB of x | of x_hat | of y | change. Same PCA and colour range for the 3 feature maps,
    so equal colours mean similar features."""
    rng = np.random.default_rng(0)
    idx = rng.choice(len(tr_d["x"]), min(50_000, len(tr_d["x"])), replace=False)
    ref = pca3.transform(np.concatenate([tr_d["x"][idx], tr_d["y"][idx]]).astype(np.float32))
    lo, hi = np.percentile(ref, 1, axis=0), np.percentile(ref, 99, axis=0)
    h, w = d["grid"]
    to_rgb = lambda f: np.clip((pca3.transform(f.astype(np.float32)) - lo) / (hi - lo), 0, 1).reshape(h, w, 3)

    cols = ["T1", "T2", "x  (T1 features)", "x̂  (predicted T2)", "y  (actual T2)", "change"]
    fig, axs = plt.subplots(len(samples), 6, figsize=(18, 3 * len(samples)), constrained_layout=True, squeeze=False)
    for r, i in enumerate(samples):
        sl = slice(i * h * w, (i + 1) * h * w)
        panels = [d["im1"][i], d["im2"][i], to_rgb(d["x"][sl]), to_rgb(d["x_hat"][sl]), to_rgb(d["y"][sl]),
                  d["change"][i]]
        for c, img in enumerate(panels):
            ax = axs[r, c]
            ax.imshow(img, cmap="gray" if c == 5 else None, vmin=0 if c == 5 else None, vmax=1 if c == 5 else None,
                      interpolation="nearest")
            ax.axis("off")
            if r == 0:
                ax.set_title(cols[c])
        axs[r, 0].text(-0.08, 0.5, f"#{i}", transform=axs[r, 0].transAxes, ha="right", va="center")
    fig.suptitle(title)
    return fig

def sq(a, b):
    """Per-patch squared L2 distance divided by C, in the full C-dim feature space
    (= the change score used everywhere else: mean over channels of the squared difference)."""
    return ((a.astype(np.float32) - b.astype(np.float32)) ** 2).mean(1)


def distance_hist_figure(d, bins=80, title=""):
    """Histograms of the distance to the actual T2, BEFORE the transformer (x vs y) and AFTER (x_hat vs y).
    Left: unchanged patches, right: changed patches. Same bins for before/after inside a panel, density scale.
    Panel titles: medians, median per-patch ratio after/before, % of patches whose distance went down."""
    before, after = sq(d["x"], d["y"]), sq(d["x_hat"], d["y"])
    fig, axs = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for ax, (lab, name) in zip(axs, ((0, "no change"), (1, "change"))):
        m = d["label"] == lab
        b, a = before[m], after[m]
        edges = np.linspace(0, np.percentile(np.concatenate([b, a]), 99.5), bins + 1)
        ax.hist(b, bins=edges, density=True, histtype="step", lw=1.8, color="tab:blue",
                label=f"before: ||x − y||²   median {np.median(b):.3g}")
        ax.hist(a, bins=edges, density=True, histtype="step", lw=1.8, color="tab:green",
                label=f"after:  ||x̂ − y||²   median {np.median(a):.3g}")
        ratio = a / np.maximum(b, 1e-12)
        ax.set_title(f"{name}: {m.sum()} val patches\n"
                     f"median after/before ratio {np.median(ratio):.2f} | distance reduced for {(a < b).mean():.0%} of patches",
                     fontsize=10)
        ax.set(xlabel="distance to actual T2 (squared L2 / C)", ylabel="density")
        ax.legend()
    fig.suptitle(title + "\nthe transformer helps detection only if it reduces the distance much more "
                         "for unchanged patches than for changed ones")
    return fig


def print_distances(d):
    """Mean squared distance per patch in the full C-dim space (same quantity as the change scores), by label."""
    rows = {"||x - y||²   (raw score)": sq(d["x"], d["y"]),
            "||x̂ - y||²   (model score)": sq(d["x_hat"], d["y"]),
            "||x̂ - x||²   (how far the transformer moves x)": sq(d["x_hat"], d["x"])}
    keys = ["x_y", "xhat_y", "xhat_x"]   # short names for the W&B summary
    summary = {}
    print(f"\n{'mean over val patches':48s} {'no change':>10s} {'change':>10s} {'ratio':>7s}")
    for key, (name, v) in zip(keys, rows.items()):
        a, b = v[d["label"] == 0].mean(), v[d["label"] == 1].mean()
        print(f"{name:48s} {a:10.4f} {b:10.4f} {b / a:7.2f}")
        summary.update({f"dist/{key}_nochange": float(a), f"dist/{key}_change": float(b),
                        f"dist/{key}_ratio": float(b / a)})
    return summary

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--samples", default="0,1,2,3,4,5", help="val crop indices for the RGB figure")
    p.add_argument("--n_pairs", type=int, default=1500)
    p.add_argument("--n_lines", type=int, default=100)
    p.add_argument("--no_wandb", action="store_true", help="only save the PNG files, log nothing to W&B")
    args = p.parse_args()

    device = get_device()
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt["cfg"]
    enc, tr = build_models(cfg, device)
    tr.load_state_dict(ckpt["transformer"])
    run_name = ckpt["train_run_name"]
    tag = f"{cfg['dataset']} | {cfg['encoder']} | seed {cfg['seed']} | {run_name}"
    run = None
    if not args.no_wandb:   # same project / group as the train and eval runs of this checkpoint
        run = wandb.init(project=cfg["project"], job_type="pca", dir=ROOT,
                         group=f"{cfg['dataset']}-{cfg['encoder']}", name=f"pca-val-{run_name}",
                         tags=[cfg["dataset"], cfg["encoder"], "val"],
                         config={**cfg, "train_run_id": ckpt["train_run_id"], "samples": args.samples,
                                 "n_pairs": args.n_pairs, "n_lines": args.n_lines})

    print("encoding train (PCA fit) ...")
    tr_d = collect(cfg, "train", enc, tr, device)
    print("encoding val ...")
    va_d = collect(cfg, "val", enc, tr, device, keep_images=True)
    print(f"val: {len(va_d['label'])} patches, {va_d['label'].mean():.2%} changed")

    pca2, pca3 = fit_pca(tr_d, 2), fit_pca(tr_d, 3)
    out_dir = ROOT / "results" / cfg["dataset"] / cfg["encoder"]
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = scatter_figure(pca2, va_d, args.n_pairs, args.n_lines, title=f"{tag} | val, PCA fitted on train x + y")
    p1 = out_dir / f"pca_pred_scatter_{run_name}.png"
    fig.savefig(p1, dpi=130)
    plt.close(fig)

    samples = [int(s) for s in args.samples.split(",")]
    fig = rgb_figure(pca3, tr_d, va_d, samples, title=f"{tag} | val, PCA→RGB fitted on train x + y")
    p2 = out_dir / f"pca_pred_rgb_{run_name}.png"
    fig.savefig(p2, dpi=110)
    plt.close(fig)

    fig = distance_hist_figure(va_d, title=f"{tag} | val")
    p3 = out_dir / f"pca_pred_hist_{run_name}.png"
    fig.savefig(p3, dpi=130)
    plt.close(fig)

    summary = print_distances(va_d)
    print(f"\nfigures saved:\n  {p1}\n  {p2}\n  {p3}")

    if run is not None:
        # log the saved PNGs (exactly the files on disk) + the distance table as summary values
        wandb.log({"pca/scatter": wandb.Image(str(p1)), "pca/rgb": wandb.Image(str(p2)),
                   "pca/distance_hist": wandb.Image(str(p3))})
        run.summary.update({**summary, "pca/explained_var_pc1": float(pca2.explained_variance_ratio_[0]),
                            "pca/explained_var_pc2": float(pca2.explained_variance_ratio_[1])})
        run.finish()
    print(f"\nfigures saved:\n  {p1}\n  {p2}\n  {p3}")


if __name__ == "__main__":
    main()