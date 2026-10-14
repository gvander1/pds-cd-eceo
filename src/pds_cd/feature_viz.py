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
        if f"f1_{m}" in rows[0]:
            a1.plot(ks, [r[f"f1_{m}"] for r in rows], "^:", label=f"F1 {m} (thr from train)")
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

def class_similarity_figure(classes, M_raw, M_centered, counts=None, title=""):
    """Hi-UCD land-cover classes in DINOv2 feature space:
    (1) mean cosine between patches of class i and class j, raw features
    (2) same after subtracting the mean feature (removes what all patches share)."""
    names = [str(c) for c in classes]
    fig, axs = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    panels = [(M_raw, "cosine, raw features", "viridis", None),
              (M_centered, "cosine, centred features", "RdBu_r", (-1, 1))]
    for ax, (M, t, cmap, lim) in zip(axs, panels):
        vmin, vmax = lim if lim else (M.min(), M.max())
        if lim == (-1, 1):
            vmax = np.abs(M).max(); vmin = -vmax
        im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax)
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if (M[i, j] - vmin) / (vmax - vmin + 1e-12) > 0.6 or
                        (cmap == "viridis" and (M[i, j] - vmin) / (vmax - vmin + 1e-12) < 0.4) else "black")
        ax.set_xticks(range(len(names)), names)
        ax.set_yticks(range(len(names)), names)
        ax.set_title(t, fontsize=10)
        ax.set_xlabel("class")
        ax.set_ylabel("class")
        fig.colorbar(im, ax=ax, shrink=0.8)
    if counts is not None:
        axs[0].set_yticks(range(len(names)), [f"{n} (n={k})" for n, k in zip(names, counts)])
    fig.suptitle(title)
    return fig

def class_similarity_pca_figure(rows, ncols=4, title=""):
    """One class x class cosine heatmap per number of PCA components k (output of A.class_cosine_pca).
    Same colour scale for every panel, so panels can be compared directly."""
    n = len(rows)
    nrows = int(np.ceil(n / ncols))
    lim = max(np.abs(r["M"]).max() for r in rows)
    fig, axs = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.4 * nrows), constrained_layout=True, squeeze=False)
    for ax in axs.flat[n:]:
        ax.axis("off")
    for ax, r in zip(axs.flat, rows):
        M, names = r["M"], [str(c) for c in r["classes"]]
        im = ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim)
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if abs(M[i, j]) > 0.6 * lim else "black")
        ax.set_xticks(range(len(names)), names, fontsize=8)
        ax.set_yticks(range(len(names)), names, fontsize=8)
        ax.set_title(f"k = {r['k']}  |  margin {r['margin']:+.2f}", fontsize=10)
    fig.colorbar(im, ax=axs, shrink=0.6, label="mean cosine (centred, in first k PCA components)")
    fig.suptitle(title)
    return fig

def _fit_pca2(arrays, max_fit=50_000, seed=0):
    rng = np.random.default_rng(seed)
    fit = np.concatenate([a[rng.choice(len(a), min(max_fit, len(a)), replace=False)] for a in arrays]).astype(np.float32)
    return PCA(2, random_state=seed).fit(fit)


def class_pca_figure(x_tr, x_va, lc_va, unlabeled=0, n_per_class=1500, max_fit=50_000, seed=0, title=""):
    """T1 patch features in the plane of the first 2 PCA components (PCA fitted on TRAIN T1 features,
    no labels), VAL patches coloured by their land-cover class (majority class of the patch at T1).
    Up to `n_per_class` patches per class so that large classes don't hide small ones; class `unlabeled` is left out."""
    pca = _fit_pca2([x_tr], max_fit, seed)
    evr = pca.explained_variance_ratio_
    rng = np.random.default_rng(seed)
    classes = [c for c in np.unique(lc_va) if c != unlabeled]
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(7.5, 6.5), constrained_layout=True)
    for i, c in enumerate(classes):
        idx = np.where(lc_va == c)[0]
        idx = rng.choice(idx, min(n_per_class, len(idx)), replace=False)
        P = pca.transform(x_va[idx].astype(np.float32))
        ax.scatter(P[:, 0], P[:, 1], s=4, alpha=0.5, color=cmap(i % 10), label=f"class {c} (n={(lc_va == c).sum()})")
    ax.set(xlabel=f"PC1 ({evr[0]:.0%} of variance)", ylabel=f"PC2 ({evr[1]:.0%} of variance)")
    ax.legend(markerscale=4, fontsize=8)
    ax.set_title(title or "T1 patch features, PCA 2D, coloured by land-cover class")
    return fig


def pair_pca_figure(x_tr, y_tr, x_va, y_va, lab_va, n_pairs=1500, n_lines=150, max_fit=50_000, seed=0, title="",
                    by_label=True):
    """T1 and T2 features of the same VAL patches in one PCA plane (PCA fitted on TRAIN T1 + T2, no labels).
    by_label=True : two panels, unchanged patches (left) and changed patches (right).
    by_label=False: one panel with a random sample of ALL val patches, changed and unchanged mixed in their
                    natural proportions, with no visual distinction between them.
    T1 = blue, T2 = orange; grey lines join T1 to T2 for `n_lines` random pairs, so the length of a line is how
    far the patch moved between the two dates."""
    pca = _fit_pca2([x_tr, y_tr], max_fit, seed)
    evr = pca.explained_variance_ratio_
    rng = np.random.default_rng(seed)
    groups = [(0, "no change"), (1, "change")] if by_label else [(None, "all patches")]
    panels = []
    for lab, name in groups:
        pool = np.arange(len(lab_va)) if lab is None else np.where(lab_va == lab)[0]
        idx = rng.choice(pool, min(n_pairs, len(pool)), replace=False)
        panels.append((name, len(pool),
                       pca.transform(x_va[idx].astype(np.float32)), pca.transform(y_va[idx].astype(np.float32))))
    allp = np.concatenate([np.concatenate([a, b]) for _, _, a, b in panels])
    lo, hi = np.percentile(allp, 0.5, axis=0), np.percentile(allp, 99.5, axis=0)
    pad = 0.05 * (hi - lo)
    fig, axs = plt.subplots(1, len(panels), figsize=(6.5 * len(panels), 6), constrained_layout=True,
                            sharex=True, sharey=True, squeeze=False)
    axs = axs[0]
    for ax, (name, n_tot, A, B) in zip(axs, panels):
        k = min(n_lines, len(A))
        for a, b in zip(A[:k], B[:k]):
            ax.plot([a[0], b[0]], [a[1], b[1]], color="grey", lw=0.4, alpha=0.5, zorder=1)
        ax.scatter(A[:, 0], A[:, 1], s=4, alpha=0.5, color="tab:blue", label="T1", zorder=2)
        ax.scatter(B[:, 0], B[:, 1], s=4, alpha=0.5, color="tab:orange", label="T2", zorder=2)
        shift = np.linalg.norm(B - A, axis=1).mean()
        ax.set_title(f"{name}: {len(A)} of {n_tot} patches | mean T1→T2 shift in this plane {shift:.2f}", fontsize=10)
        ax.set(xlabel=f"PC1 ({evr[0]:.0%} of variance)", xlim=(lo[0] - pad[0], hi[0] + pad[0]),
               ylim=(lo[1] - pad[1], hi[1] + pad[1]))
        ax.legend(markerscale=4)
    axs[0].set_ylabel(f"PC2 ({evr[1]:.0%} of variance)")
    fig.suptitle(title or "T1 vs T2 features of the same patches, PCA 2D")
    return fig