"""Steps 1, 2 and 4: change scores computed on cached patch features (see features.py).

Every score is "higher = more likely change" and is defined per patch:
    L2            mean over channels of (y - x)^2      (same as the raw error of evaluate.py)
    cosine        1 - cos(x, y)                         (ignores the feature norm)
    probe         logistic regression on |y - x|        (uses labels: diagnostic, not a method)
    LDA           best linear direction for 2 classes   (uses labels: diagnostic)
    best channel  |y - x| of the single best channel, chosen on TRAIN
"""
import numpy as np
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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

def probe_input(x, y, kind="absdiff"):
    """Patch representation given to the probe / LDA."""
    x32, y32 = x.astype(np.float32), y.astype(np.float32)
    if kind == "absdiff":
        return np.abs(y32 - x32)
    if kind == "diff":
        return y32 - x32
    if kind == "concat":
        return np.concatenate([x32, y32], axis=1)
    raise ValueError(f"Unknown probe input {kind!r}")


def balanced_index(labels, neg_ratio=1.0, seed=0):
    """All positive patches + neg_ratio times as many random negatives."""
    pos = np.where(labels == 1)[0]
    neg = np.where(labels == 0)[0]
    if len(pos) == 0:
        raise ValueError("No changed patch in the training labels: check the patch label threshold (label_thr)")
    n_neg = min(len(neg), int(round(neg_ratio * len(pos))))
    neg = np.random.default_rng(seed).choice(neg, n_neg, replace=False)
    return np.sort(np.concatenate([pos, neg]))


def fit_probe(X_tr, lab_tr, neg_ratio=1.0, C=1.0, seed=0):
    """Linear probe = standardisation + logistic regression, trained on a class-balanced subset of TRAIN."""
    idx = balanced_index(lab_tr, neg_ratio, seed)
    model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
    return model.fit(X_tr[idx], lab_tr[idx])


def fit_lda(X_tr, lab_tr, neg_ratio=1.0, seed=0):
    idx = balanced_index(lab_tr, neg_ratio, seed)
    return make_pipeline(StandardScaler(), LinearDiscriminantAnalysis()).fit(X_tr[idx], lab_tr[idx])


def model_scores(model, X):
    """decision_function in chunks (same ranking as the predicted probability)."""
    out = np.empty(len(X), np.float32)
    for sl in _chunks(len(X)):
        out[sl] = model.decision_function(X[sl])
    return out


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


def pca_sweep(x_tr, y_tr, x_va, y_va, lab_va, ks=(2, 5, 10, 20, 50, 100, 200, 384), max_fit=50_000, seed=0):
    """Step 4: L2 / cosine distance computed in the first k PCA components (PCA fitted on TRAIN, no labels)."""
    idx = _subsample(len(x_tr), max_fit, seed)
    fit_x = np.concatenate([x_tr[idx], y_tr[idx]]).astype(np.float32)
    k_max = min(max(ks), fit_x.shape[1])
    pca = PCA(n_components=k_max, random_state=seed).fit(fit_x)
    px = np.concatenate([pca.transform(x_va[sl].astype(np.float32)) for sl in _chunks(len(x_va))]).astype(np.float32)
    py = np.concatenate([pca.transform(y_va[sl].astype(np.float32)) for sl in _chunks(len(y_va))]).astype(np.float32)
    rows = []
    for k in sorted({min(k, k_max) for k in ks}):
        l2 = l2_distance(px[:, :k], py[:, :k])
        cos = cosine_distance(px[:, :k], py[:, :k])
        r = {"k": k}
        r.update({f"{m}_L2": v for m, v in separability(l2, lab_va).items()})
        r.update({f"{m}_cosine": v for m, v in separability(cos, lab_va).items()})
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