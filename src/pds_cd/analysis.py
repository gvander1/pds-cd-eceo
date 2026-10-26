"""Steps 1, 2 and 4: change scores computed on cached patch features (see features.py).

Every score is "higher = more likely change" and is defined per patch:
    L2            mean over channels of (y - x)^2      (same as the raw error of evaluate.py)
    cosine        1 - cos(x, y)                         (ignores the feature norm)
    best channel  |y - x| of the single best channel, chosen on TRAIN
"""
import numpy as np
from scipy.stats import rankdata
from sklearn.decomposition import PCA

from .metrics import best_f1_threshold, scores_at_threshold, separability

CHUNK = 65536  # rows processed at once, keeps memory low on large splits


def _chunks(n, size=CHUNK):
    for s in range(0, n, size):
        yield slice(s, min(s + size, n))


def _subsample(n, max_n, seed=0):
    if n <= max_n:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, max_n, replace=False))


# ------------------------------------------------------------------ unsupervised distances

def l2_distance(x, y):
    out = np.empty(len(x), np.float32)
    for sl in _chunks(len(x)):
        d = y[sl].astype(np.float32) - x[sl].astype(np.float32)
        out[sl] = (d * d).mean(1)
    return out


def cosine_distance(x, y, eps=1e-8):
    out = np.empty(len(x), np.float32)
    for sl in _chunks(len(x)):
        a, b = x[sl].astype(np.float32), y[sl].astype(np.float32)
        out[sl] = 1 - (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + eps)
    return out


# ------------------------------------------------------------------ supervised diagnostics

def abs_difference(x, y):
    """|y - x| per patch and channel (float32). Used for the PCA figure."""
    return np.abs(y.astype(np.float32) - x.astype(np.float32))


def per_channel_auroc(X, labels, max_n=100_000, seed=0):
    """AUROC of every column of X taken alone (Mann-Whitney formula, all channels at once)."""
    idx = _subsample(len(labels), max_n, seed)
    Xs, ys = X[idx], labels[idx].astype(bool)
    n1 = ys.sum()
    n0 = len(ys) - n1
    ranks = rankdata(Xs, axis=0)
    return (ranks[ys].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)


# ------------------------------------------------------------------ evaluation

def evaluate_score(s_tr, lab_tr, s_va, lab_va):
    """Same protocol as evaluate.py: best-F1 threshold on TRAIN, applied to VAL, + AUROC / AP on VAL."""
    thr, f1_tr = best_f1_threshold(s_tr, lab_tr)
    res = {"threshold": thr, "train_f1": f1_tr}
    res.update(separability(s_va, lab_va))
    res.update(scores_at_threshold(s_va, lab_va, thr))
    return res


def pca_sweep(x_tr, y_tr, x_va, y_va, lab_va, lab_tr=None,
              ks=(2, 5, 10, 20, 50, 100, 200, 384), max_fit=50_000, seed=0):
    """Step 4: L2 / cosine distance computed in the first k PCA components (PCA fitted on TRAIN, no labels).
    AUROC / AP on VAL. If lab_tr is given, also F1 on VAL with the best-F1 threshold chosen on TRAIN
    (same protocol as the results table)."""
    idx = _subsample(len(x_tr), max_fit, seed)
    fit_x = np.concatenate([x_tr[idx], y_tr[idx]]).astype(np.float32)
    k_max = min(fit_x.shape)  # number of feature channels (384 ViT-S, 768 ViT-B, 1024 ResNet-50 layer3...)
    ks = sorted({min(k, k_max) for k in ks} | {k_max})  # always include "all components"
    pca = PCA(n_components=k_max, random_state=seed).fit(fit_x)

    def project(a):
        return np.concatenate([pca.transform(a[sl].astype(np.float32)) for sl in _chunks(len(a))]).astype(np.float32)

    px, py = project(x_va), project(y_va)
    if lab_tr is not None:
        tx, ty = project(x_tr), project(y_tr)
    rows = []
    for k in sorted({min(k, k_max) for k in ks}):
        r = {"k": k}
        for name, dist in (("L2", l2_distance), ("cosine", cosine_distance)):
            s_va = dist(px[:, :k], py[:, :k])
            if lab_tr is None:
                r.update({f"{m}_{name}": v for m, v in separability(s_va, lab_va).items()})
            else:
                res = evaluate_score(dist(tx[:, :k], ty[:, :k]), lab_tr, s_va, lab_va)
                r.update({f"{m}_{name}": res[m] for m in ("auroc", "ap", "f1", "precision", "recall")})
        rows.append(r)
    return rows, pca.explained_variance_ratio_

def per_transition(score, labels, lc1, lc2, thr, min_count=50):
    """Hi-UCD: behaviour of a score for each land-cover transition (lc1 -> lc2) at patch level.

    above_thr          fraction of the group's patches above the threshold
                       (= recall for changed groups, false-positive rate for unchanged groups)
    auroc_vs_unchanged for groups containing changed patches: probability that one of their changed
                       patches scores higher than a random unchanged patch of the whole split
    """
    neg_sorted = np.sort(score[labels == 0])
    rows = []
    pairs = np.unique(np.stack([lc1, lc2], 1), axis=0)
    for a, b in pairs:
        m = (lc1 == a) & (lc2 == b)
        n = int(m.sum())
        if n < min_count:
            continue
        s, lab = score[m], labels[m]
        row = {"lc1": int(a), "lc2": int(b), "n": n, "frac_changed": float(lab.mean()),
               "mean_score": float(s.mean()), "above_thr": float((s > thr).mean())}
        ch = s[lab == 1]
        if len(ch) >= min_count:
            lo = np.searchsorted(neg_sorted, ch, side="left")
            hi = np.searchsorted(neg_sorted, ch, side="right")
            row["auroc_vs_unchanged"] = float(((lo + hi) / 2).mean() / len(neg_sorted))
        rows.append(row)
    return rows

# ------------------------------------------------------------------ Hi-UCD: are land-cover classes separated in feature space?

def class_patches(d, unlabeled=0):
    """Pool T1 and T2 patches with their own land-cover class: T1 features get lc1, T2 features get lc2.
    Returns (features (M, C) float16, classes (M,)) without the unlabelled class."""
    feats = np.concatenate([d["x"], d["y"]])
    cls = np.concatenate([d["lc1"], d["lc2"]])
    keep = cls != unlabeled
    return feats[keep], cls[keep]


def _normalize(X, mean=None):
    X = X.astype(np.float32)
    if mean is not None:
        X = X - mean
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)


def class_cosine_matrix(feats, cls, mean=None, max_per_class=1000, seed=0):
    """Mean cosine similarity between individual patches of class i and class j (random subsample per class).
    Diagonal = within-class similarity (self-pairs excluded), off-diagonal = between classes.
    `mean`: vector subtracted from every feature before the cosine (None = raw features)."""
    rng = np.random.default_rng(seed)
    classes = np.unique(cls)
    groups = []
    for c in classes:
        idx = np.where(cls == c)[0]
        idx = rng.choice(idx, min(max_per_class, len(idx)), replace=False)
        groups.append(_normalize(feats[idx], mean))
    K = len(classes)
    M = np.zeros((K, K), np.float32)
    for i in range(K):
        for j in range(i, K):
            S = groups[i] @ groups[j].T
            if i == j:
                n = len(S)
                M[i, i] = (S.sum() - np.trace(S)) / max(n * (n - 1), 1)
            else:
                M[i, j] = M[j, i] = S.mean()
    counts = np.array([(cls == c).sum() for c in classes])
    return classes, M, counts


def nearest_prototype(feats_tr, cls_tr, feats_va, cls_va, mean=None):
    """Prototype of each class = mean of its (normalised) TRAIN features. Every VAL patch is assigned to the
    prototype with the highest cosine. Returns classes, row-normalised confusion matrix, balanced accuracy."""
    classes = np.unique(cls_tr)
    protos = np.stack([_normalize(feats_tr[cls_tr == c], mean).mean(0) for c in classes])
    protos /= np.linalg.norm(protos, axis=1, keepdims=True)
    pred = np.empty(len(feats_va), np.int64)
    for sl in _chunks(len(feats_va)):
        pred[sl] = (_normalize(feats_va[sl], mean) @ protos.T).argmax(1)
    pred = classes[pred]
    conf = np.zeros((len(classes), len(classes)))
    for a, c in enumerate(classes):
        m = cls_va == c
        if m.any():
            conf[a] = [(pred[m] == p).mean() for p in classes]
    present = [a for a, c in enumerate(classes) if (cls_va == c).any()]
    return classes, conf, float(np.mean([conf[a, a] for a in present]))

def class_cosine_pca(feats_tr, feats_va, cls_va, ks=(2, 5, 10, 20, 50, 100, 384), max_fit=50_000,
                     max_per_class=1000, seed=0):
    """Class x class cosine matrix (as class_cosine_matrix) computed in the first k PCA components.
    PCA is fitted on TRAIN class patches without labels; PCA centres the data, so k = 384 equals the
    'centred' matrix. Returns a list of dicts {k, classes, M, counts, within, max_between, margin}:
        within       mean of the diagonal (same-class similarity)
        max_between  mean over classes of the highest similarity to ANOTHER class
        margin       mean over classes of (own-class similarity - highest other-class similarity);
                     > 0 means classes are, on average, closer to themselves than to their nearest other class."""
    idx = _subsample(len(feats_tr), max_fit, seed)
    k_max = min(feats_tr.shape[1], len(idx))
    ks = sorted({min(k, k_max) for k in ks} | {k_max})  # always include "all components"
    pca = PCA(n_components=k_max, random_state=seed).fit(feats_tr[idx].astype(np.float32))
    P = np.concatenate([pca.transform(feats_va[sl].astype(np.float32)) for sl in _chunks(len(feats_va))]).astype(np.float32)
    out = []
    for k in sorted({min(k, k_max) for k in ks}):
        classes, M, counts = class_cosine_matrix(P[:, :k], cls_va, max_per_class=max_per_class, seed=seed)
        off = M.copy()
        np.fill_diagonal(off, -np.inf)
        best_other = off.max(1)
        out.append({"k": k, "classes": classes, "M": M, "counts": counts,
                    "within": float(np.diag(M).mean()), "max_between": float(best_other.mean()),
                    "margin": float((np.diag(M) - best_other).mean())})
    return out, pca.explained_variance_ratio_