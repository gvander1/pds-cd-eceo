"""Steps 1 (+ 4): compare change scores on cached patch features and log everything to W&B.

    python scripts/analyze_features.py --data levir
    python scripts/analyze_features.py --data hiucd --pca_sweep
    python scripts/analyze_features.py --data levir --probe_input concat --no_wandb

Needs features/<dataset>/<encoder>/{train,val}.npz  (python scripts/extract_features.py --data ...).
Protocol for every score: threshold = best F1 on TRAIN patches, metrics on VAL patches.
Figures are also saved to results/<dataset>/<encoder>/.
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from pds_cd import analysis as A  # noqa: E402
from pds_cd import feature_viz as V  # noqa: E402
from pds_cd.config import ROOT, load_config  # noqa: E402
from pds_cd.features import load_features  # noqa: E402

METRICS = ["auroc", "ap", "f1", "precision", "recall", "pred_pos_frac", "threshold"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--probe_input", default="absdiff", choices=["absdiff", "diff", "concat"])
    p.add_argument("--neg_ratio", type=float, default=1.0, help="unchanged/changed ratio for probe & LDA training")
    p.add_argument("--pca_sweep", action="store_true", help="step 4: AUROC/AP vs number of PCA components")
    p.add_argument("--no_wandb", action="store_true")
    args, rest = p.parse_known_args()
    cfg = load_config(rest)
    ds, enc = cfg["dataset"], cfg["encoder"]
    tag = f"{ds} | {enc}"

    tr = load_features(ds, enc, "train", with_images=False)
    va = load_features(ds, enc, "val", with_images=False)
    lab_tr, lab_va = tr["label"], va["label"]
    print(f"train: {len(lab_tr)} patches ({lab_tr.mean():.2%} changed) | val: {len(lab_va)} patches ({lab_va.mean():.2%} changed)")

    s_tr, s_va = {}, {}
    s_tr["L2"], s_va["L2"] = A.l2_distance(tr["x"], tr["y"]), A.l2_distance(va["x"], va["y"])
    s_tr["cosine"], s_va["cosine"] = A.cosine_distance(tr["x"], tr["y"]), A.cosine_distance(va["x"], va["y"])

    X_tr, X_va = A.probe_input(tr["x"], tr["y"], args.probe_input), A.probe_input(va["x"], va["y"], args.probe_input)
    probe = A.fit_probe(X_tr, lab_tr, neg_ratio=args.neg_ratio)
    s_tr["probe"], s_va["probe"] = A.model_scores(probe, X_tr), A.model_scores(probe, X_va)
    lda = A.fit_lda(X_tr, lab_tr, neg_ratio=args.neg_ratio)
    s_tr["LDA"], s_va["LDA"] = A.model_scores(lda, X_tr), A.model_scores(lda, X_va)

    if args.probe_input != "absdiff":
        X_tr, X_va = A.probe_input(tr["x"], tr["y"]), A.probe_input(va["x"], va["y"])
    ch_tr = A.per_channel_auroc(X_tr, lab_tr)
    ch_va = A.per_channel_auroc(X_va, lab_va)
    best = int(np.argmax(ch_tr))  # chosen on TRAIN, reported on VAL
    s_tr["best_channel"], s_va["best_channel"] = X_tr[:, best], X_va[:, best]
    del X_tr, X_va

    results = {name: A.evaluate_score(s_tr[name], lab_tr, s_va[name], lab_va) for name in s_va}
    print(f"\n{tag} — patch level, val (threshold from train); best channel = #{best}")
    print(f"{'score':14s}" + "".join(f"{m:>14s}" for m in METRICS))
    for name, r in results.items():
        print(f"{name:14s}" + "".join(f"{r[m]:14.4f}" for m in METRICS))

    figs = {
        "roc_pr": V.roc_pr_figure(s_va, lab_va, f"{tag} | val"),
        "histograms": V.score_histograms(s_va, lab_va, f"{tag} | val"),
        "per_channel_auroc": V.per_channel_figure(ch_va, f"{tag} | val | AUROC of each channel of |y - x|"),
        "scatter_L2_probe": V.scatter_figure(s_va["L2"], s_va["probe"], lab_va, ("L2", "probe"), title=f"{tag} | val"),
    }

    transitions = None
    if va["meta"]["has_landcover"]:
        transitions = A.per_transition(s_va["L2"], lab_va, va["lc1"], va["lc2"], results["L2"]["threshold"])
        figs["transitions_L2"] = V.transition_figure(transitions, f"{tag} | val | L2, threshold from train")

    sweep = None
    if args.pca_sweep:
        sweep, evr = A.pca_sweep(tr["x"], tr["y"], va["x"], va["y"], lab_va)
        figs["pca_sweep"] = V.pca_sweep_figure(sweep, evr, f"{tag} | distance in the first k PCA components")
        for r in sweep:
            print(f"k={r['k']:4d}  AUROC L2 {r['auroc_L2']:.4f}  AP L2 {r['ap_L2']:.4f}  "
                  f"AUROC cos {r['auroc_cosine']:.4f}  AP cos {r['ap_cosine']:.4f}")

    out_dir = ROOT / "results" / ds / enc
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, fig in figs.items():
        fig.savefig(out_dir / f"{name}.png", dpi=120, bbox_inches="tight")
    print(f"\nfigures saved to {out_dir}")

    if not args.no_wandb:
        import wandb
        run = wandb.init(project=cfg["project"], job_type="analysis", dir=ROOT, group=f"{ds}-{enc}",
                         name=f"analysis-{ds}-{enc}-{args.probe_input}", tags=[ds, enc, "analysis"],
                         config={**cfg, **vars(args), "best_channel": best,
                                 "n_patches_train": len(lab_tr), "n_patches_val": len(lab_va)})
        run.summary.update({f"val/{m}_{name}": r[m] for name, r in results.items() for m in METRICS})
        log = {k: wandb.Image(f) for k, f in figs.items()}
        log["results"] = wandb.Table(columns=["score"] + METRICS,
                                     data=[[name] + [r[m] for m in METRICS] for name, r in results.items()])
        if transitions:
            cols = ["lc1", "lc2", "n", "frac_changed", "mean_score", "above_thr", "auroc_vs_unchanged"]
            log["transitions_L2_table"] = wandb.Table(columns=cols, data=[[r.get(c) for c in cols] for r in transitions])
        if sweep:
            cols = list(sweep[0].keys())
            log["pca_sweep_table"] = wandb.Table(columns=cols, data=[[r[c] for c in cols] for r in sweep])
        wandb.log(log)
        run.finish()
    plt.close("all")


if __name__ == "__main__":
    main()