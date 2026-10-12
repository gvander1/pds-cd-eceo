"""Figures for the feature analysis (steps 1-4). Every function returns a matplotlib figure:
plt.show() in a notebook, wandb.Image(fig) or fig.savefig(...) in a script."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_curve, roc_curve

from .analysis import l2_distance
from .features import to_grid


def _balanced(labels, n_per_class, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for c in (0, 1):
        idx = np.where(labels == c)[0]
        out.append(rng.choice(idx, min(n_per_class, len(idx)), replace=False))
    return np.concatenate(out)


# ------------------------------------------------------------------ step 1

def roc_pr_figure(scores, labels, title=""):
    """ROC and precision-recall curves of several scores on the same axes."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for name, s in scores.items():
        fpr, tpr, _ = roc_curve(labels, s)
        a1.plot(fpr, tpr, label=name)
        p, r, _ = precision_recall_curve(labels, s)
        a2.plot(r, p, label=name)
    a1.plot([0, 1], [0, 1], "k:", lw=1, label="chance")
    a2.axhline(labels.mean(), color="k", ls=":", lw=1, label=f"chance ({labels.mean():.3f})")
    a1.set(xlabel="false positive rate", ylabel="true positive rate", title="ROC")
    a2.set(xlabel="recall", ylabel="precision", title="precision-recall", ylim=(0, 1))
    a1.legend()
    a2.legend()
    fig.suptitle(title)
    return fig


def score_histograms(scores, labels, title=""):
    """One panel per score: change vs no-change distributions (same bins, density, log scale)."""
    lab = labels.astype(bool)
    fig, axs = plt.subplots(1, len(scores), figsize=(5 * len(scores), 4), constrained_layout=True, squeeze=False)
    for ax, (name, s) in zip(axs[0], scores.items()):
        bins = np.linspace(np.percentile(s, 0.1), np.percentile(s, 99.9), 101)
        ax.hist(s[~lab], bins=bins, alpha=0.6, density=True, label="no change")
        ax.hist(s[lab], bins=bins, alpha=0.6, density=True, label="change")
        ax.set_yscale("log")
        ax.set_title(name)
        ax.legend()
    fig.suptitle(title)
    return fig


def per_channel_figure(auroc, title=""):
    """AUROC of each channel of |y - x| taken alone, sorted."""
    order = np.argsort(auroc)[::-1]
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    ax.plot(np.arange(len(auroc)), auroc[order])
    ax.axhline(0.5, color="k", ls=":", lw=1)
    for i in range(5):
        ax.annotate(f"#{order[i]}", (i, auroc[order[i]]), textcoords="offset points", xytext=(5, 5 + 10 * i), fontsize=8)
    ax.set(xlabel="channels, sorted", ylabel="AUROC of the channel alone", title=title)
    return fig


def scatter_figure(a, b, labels, names=("L2", "probe"), n_per_class=3000, seed=0, title=""):
    """One point per patch (balanced subsample): which patches one score recovers and the other misses."""
    idx = _balanced(labels, n_per_class, seed)
    fig, ax = plt.subplots(figsize=(6, 5.5), constrained_layout=True)
    for c, name in ((0, "no change"), (1, "change")):
        m = idx[labels[idx] == c]
        ax.scatter(a[m], b[m], s=4, alpha=0.4, label=name)
    ax.set(xlabel=names[0], ylabel=names[1], title=title)
    ax.legend(markerscale=4)
    return fig


def pca_sweep_figure(rows, explained_variance_ratio, title=""):
    ks = [r["k"] for r in rows]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True, constrained_layout=True)
    for m in ("L2", "cosine"):
        a1.plot(ks, [r[f"auroc_{m}"] for r in rows], "o-", label=f"AUROC {m}")
        a1.plot(ks, [r[f"ap_{m}"] for r in rows], "s--", label=f"AP {m}")
    a1.set(ylabel="score on val", title=title)
    a1.legend()
    a2.plot(np.arange(1, len(explained_variance_ratio) + 1), np.cumsum(explained_variance_ratio))
    a2.set(xscale="log", xlabel="number of PCA components k", ylabel="cumulative explained variance")
    return fig


# ------------------------------------------------------------------ step 2 (needs a cache with images)

def fit_rgb_pca(d, max_fit=50_000, seed=0):
    """3-component PCA on T1 + T2 patch features, with percentile ranges to map components to RGB."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(d["x"]), min(max_fit, len(d["x"])), replace=False)
    fit_x = np.concatenate([d["x"][idx], d["y"][idx]]).astype(np.float32)
    pca = PCA(3, random_state=seed).fit(fit_x)
    proj = pca.transform(fit_x)
    lo, hi = np.percentile(proj, 1, axis=0), np.percentile(proj, 99, axis=0)
    return pca, lo, hi


def pca_rgb_figure(d, samples, rgb_pca=None, score=None, score_name="L2", title=""):
    """Per sample: T1 | T2 | PCA-RGB(T1 features) | PCA-RGB(T2 features) | score map | change label.
    The same PCA and colour range are used for T1 and T2, so equal colours mean similar features."""
    pca, lo, hi = rgb_pca if rgb_pca is not None else fit_rgb_pca(d)
    score = l2_distance(d["x"], d["y"]) if score is None else score
    h, w = d["meta"]["grid"]
    score_maps = to_grid(score, d["meta"])
    smin, smax = np.percentile(score, 1), np.percentile(score, 99)
    cols = ["T1", "T2", "features T1 (PCA→RGB)", "features T2 (PCA→RGB)", score_name, "change"]
    fig, axs = plt.subplots(len(samples), 6, figsize=(18, 3 * len(samples)), constrained_layout=True, squeeze=False)
    for r, i in enumerate(samples):
        sl = slice(i * h * w, (i + 1) * h * w)
        rgb = [np.clip((pca.transform(d[k][sl].astype(np.float32)) - lo) / (hi - lo), 0, 1).reshape(h, w, 3)
               for k in ("x", "y")]
        panels = [d["im1"][i], d["im2"][i], rgb[0], rgb[1], score_maps[i], d["change"][i]]
        for c, img in enumerate(panels):
            ax = axs[r, c]
            if c == 4:
                ax.imshow(img, cmap="inferno", vmin=smin, vmax=smax, interpolation="nearest")
            elif c == 5:
                ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            else:
                ax.imshow(img, interpolation="nearest")
            ax.axis("off")
            if r == 0:
                ax.set_title(cols[c])
        axs[r, 0].text(-0.08, 0.5, f"#{i}", transform=axs[r, 0].transAxes, ha="right", va="center")
    fig.suptitle(title)
    return fig


def gallery_figure(d, score, kind="fp", n=8, score_name="L2", title="", picked=None):
    """Worst errors of a score, one per crop: 'fp' = highest-scoring unchanged patches,
    'fn' = lowest-scoring changed patches. The offending patch is outlined in cyan."""
    if picked is None:
        labels = d["label"]
        if kind == "fp":
            cand = np.where(labels == 0)[0]
            cand = cand[np.argsort(score[cand])[::-1]]
        elif kind == "fn":
            cand = np.where(labels == 1)[0]
            cand = cand[np.argsort(score[cand])]
        else:
            raise ValueError("kind must be 'fp' or 'fn'")
        _, first = np.unique(d["sample"][cand], return_index=True)   # keep one patch per crop
        picked = cand[np.sort(first)][:n]


    h, w = d["meta"]["grid"]
    H, W = d["im1"].shape[1:3]
    ph, pw = H / h, W / w
    score_maps = to_grid(score, d["meta"])
    smin, smax = np.percentile(score, 1), np.percentile(score, 99)
    fig, axs = plt.subplots(len(picked), 4, figsize=(12, 3 * len(picked)), constrained_layout=True, squeeze=False)
    for r, k in enumerate(picked):
        i, row, col = d["sample"][k], d["row"][k], d["col"][k]
        panels = [d["im1"][i], d["im2"][i], d["change"][i], score_maps[i]]
        for c, img in enumerate(panels):
            ax = axs[r, c]
            if c == 2:
                ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            elif c == 3:
                ax.imshow(img, cmap="inferno", vmin=smin, vmax=smax, interpolation="nearest")
            else:
                ax.imshow(img)
            if c < 3:
                ax.add_patch(Rectangle((col * pw, row * ph), pw, ph, fill=False, ec="cyan", lw=2))
            else:
                ax.add_patch(Rectangle((col - 0.5, row - 0.5), 1, 1, fill=False, ec="cyan", lw=2))
            ax.axis("off")
            if r == 0:
                ax.set_title(["T1", "T2", "change", score_name][c])
        axs[r, 0].text(-0.08, 0.5, f"#{i}\n{score_name}={score[k]:.3g}\nfrac={float(d['frac'][k]):.2f}",
                       transform=axs[r, 0].transAxes, ha="right", va="center", fontsize=8)
    fig.suptitle(title or f"top {'false positives' if kind == 'fp' else 'false negatives'} ({score_name})")
    return fig


def transition_figure(rows, title=""):
    """Hi-UCD: left = recall per land-cover transition (changed groups),
    right = false-positive rate per land-cover class (unchanged groups)."""
    changed = sorted([r for r in rows if r["lc1"] != r["lc2"]], key=lambda r: r["above_thr"])
    same = sorted([r for r in rows if r["lc1"] == r["lc2"]], key=lambda r: r["above_thr"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, max(4, 0.3 * max(len(changed), len(same)))), constrained_layout=True)
    a1.barh([f"{r['lc1']}→{r['lc2']} (n={r['n']})" for r in changed], [r["above_thr"] for r in changed])
    a1.set(xlabel="fraction above threshold", title="changed transitions (recall)", xlim=(0, 1))
    a2.barh([f"{r['lc1']}→{r['lc2']} (n={r['n']})" for r in same], [r["above_thr"] for r in same], color="C1")
    a2.set(xlabel="fraction above threshold", title="unchanged classes (false-positive rate)", xlim=(0, 1))
    fig.suptitle(title)
    return fig

def pca_scatter_figure(X_tr, X_va, lab_va, extra=None, extra_name="LDA score",
                       n_per_class=3000, max_fit=50_000, seed=0, title=""):
    """Val patches in the plane of the first two PCA components of X (PCA fitted on TRAIN, no labels),
    coloured by label (balanced subsample). If `extra` is given (one score per val patch, e.g. LDA),
    a second panel shows PC1 vs that score, to contrast unsupervised and supervised directions."""
    rng = np.random.default_rng(seed)
    fit_idx = rng.choice(len(X_tr), min(max_fit, len(X_tr)), replace=False)
    pca = PCA(2, random_state=seed).fit(X_tr[fit_idx].astype(np.float32))
    evr = pca.explained_variance_ratio_

    idx = _balanced(lab_va, n_per_class, seed)
    P = pca.transform(X_va[idx].astype(np.float32))
    lab = lab_va[idx]
    panels = [(P[:, 0], P[:, 1], f"PC1 ({evr[0]:.0%} of variance)", f"PC2 ({evr[1]:.0%} of variance)")]
    if extra is not None:
        panels.append((P[:, 0], extra[idx], f"PC1 ({evr[0]:.0%} of variance)", extra_name))

    fig, axs = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5.5), constrained_layout=True, squeeze=False)
    for ax, (a, b, xl, yl) in zip(axs[0], panels):
        for c, name in ((0, "no change"), (1, "change")):
            m = lab == c
            ax.scatter(a[m], b[m], s=4, alpha=0.4, label=name)
        ax.set(xlabel=xl, ylabel=yl)
        ax.legend(markerscale=4)
    axs[0, 0].set_title("PCA (no labels)")
    if extra is not None:
        axs[0, 1].set_title(f"PC1 vs {extra_name} (uses labels)")
    fig.suptitle(title)
    return fig, pca