"""Steps 1 (+ 4): compare change scores on cached patch features and log everything to W&B.

    python scripts/analyze_features.py --data levir
    python scripts/analyze_features.py --data hiucd --pca_sweep
    
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
    # p.add_argument("--probe_input", default="absdiff", choices=["absdiff", "diff", "concat"])
    # p.add_argument("--neg_ratio", type=float, default=1.0, help="unchanged/changed ratio for probe & LDA training")
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

    # |y - x| per patch, only for the PCA figure (no labels used to compute it)
    D_tr, D_va = A.abs_difference(tr["x"], tr["y"]), A.abs_difference(va["x"], va["y"])
    pca_fig, _ = V.pca_scatter_figure(D_tr, D_va, lab_va, title=f"{tag} | val | PCA of |y - x|")
    del D_tr, D_va

    results = {name: A.evaluate_score(s_tr[name], lab_tr, s_va[name], lab_va) for name in s_va}
    print(f"\n{tag} — patch level, val (threshold from train)")
    print(f"{'score':14s}" + "".join(f"{m:>14s}" for m in METRICS))
    for name, r in results.items():
        print(f"{name:14s}" + "".join(f"{r[m]:14.4f}" for m in METRICS))

    figs = {
        "roc_pr": V.roc_pr_figure(s_va, lab_va, f"{tag} | val"),
        "histograms": V.score_histograms(s_va, lab_va, f"{tag} | val"),
        # "per_channel_auroc": V.per_channel_figure(ch_va, f"{tag} | val | AUROC of each channel of |y - x|"),
        # "scatter_L2_probe": V.scatter_figure(s_va["L2"], s_va["probe"], lab_va, ("L2", "probe"), title=f"{tag} | val"),
        "pca_scatter": pca_fig,
        "pca_pairs_T1_T2": V.pair_pca_figure(tr["x"], tr["y"], va["x"], va["y"], lab_va,
                                             title=f"{tag} | val | T1 (blue) vs T2 (orange) features, PCA 2D"),
        "pca_pairs_T1_T2_all": V.pair_pca_figure(tr["x"], tr["y"], va["x"], va["y"], lab_va, by_label=False,
                                                 title=f"{tag} | val | T1 (blue) vs T2 (orange), all patches (changed + unchanged), PCA 2D"),
    }
    if va["meta"]["has_landcover"]:
        figs["pca_classes_T1"] = V.class_pca_figure(tr["x"], va["x"], va["lc1"],
                                                    title=f"{tag} | val | T1 patch features, PCA 2D, coloured by land-cover class")
    
    class_rows = class_pca_rows = None
    if va["meta"]["has_landcover"]:
        f_tr, c_tr = A.class_patches(tr)
        f_va, c_va = A.class_patches(va)
        mu = f_tr.astype(np.float32).mean(0)
        classes, M_raw, counts = A.class_cosine_matrix(f_va, c_va)
        _, M_cen, _ = A.class_cosine_matrix(f_va, c_va, mean=mu)
        # _, conf, bal_acc = A.nearest_prototype(f_tr, c_tr, f_va, c_va, mean=mu)
        figs["class_similarity"] = V.class_similarity_figure(classes, M_raw, M_cen, counts,
                                                             f"{tag} | land-cover classes in feature space (T1 + T2 patches)")
        cls_pca, _ = A.class_cosine_pca(f_tr, f_va, c_va)
        figs["class_similarity_pca"] = V.class_similarity_pca_figure(
            cls_pca, title=f"{tag} | land-cover classes: cosine in the first k PCA components (PCA fitted on train, no labels)")
        class_pca_rows = [[r["k"], r["within"], r["max_between"], r["margin"]] for r in cls_pca]
        for r in cls_pca:
            print(f"class similarity k={r['k']:4d}  within {r['within']:.3f}  max_between {r['max_between']:.3f}  margin {r['margin']:+.3f}")
        class_rows = [[int(c), int(n), float(M_raw[i, i]), float(M_cen[i, i]),
                       float(np.delete(M_cen[i], i).max()), int(classes[np.delete(np.arange(len(classes)), i)[np.argmax(np.delete(M_cen[i], i))]])] for i, (c, n) in enumerate(zip(classes, counts))]
        del f_tr, f_va

    transitions = None
    if va["meta"]["has_landcover"]:
        transitions = A.per_transition(s_va["L2"], lab_va, va["lc1"], va["lc2"], results["L2"]["threshold"])
        figs["transitions_L2"] = V.transition_figure(transitions, f"{tag} | val | L2, threshold from train")

    sweep = None
    if args.pca_sweep:
        sweep, evr = A.pca_sweep(tr["x"], tr["y"], va["x"], va["y"], lab_va, lab_tr=lab_tr)
        figs["pca_sweep"] = V.pca_sweep_figure(sweep, evr, f"{tag} | distance in the first k PCA components")
        for r in sweep:
            print(f"k={r['k']:4d}  L2: AUROC {r['auroc_L2']:.4f} AP {r['ap_L2']:.4f} F1 {r['f1_L2']:.4f}  |  "
                  f"cos: AUROC {r['auroc_cosine']:.4f} AP {r['ap_cosine']:.4f} F1 {r['f1_cosine']:.4f}")

    out_dir = ROOT / "results" / ds / enc
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, fig in figs.items():
        fig.savefig(out_dir / f"{name}.png", dpi=120, bbox_inches="tight")
    print(f"\nfigures saved to {out_dir}")

    if not args.no_wandb:
        import wandb
        run = wandb.init(project=cfg["project"], job_type="analysis", dir=ROOT, group=f"{ds}-{enc}",
                         name=f"analysis-{ds}-{enc}", tags=[ds, enc, "analysis"],
                         config={**cfg, **vars(args),
                                 "n_patches_train": len(lab_tr), "n_patches_val": len(lab_va)})
        run.summary.update({f"val/{m}_{name}": r[m] for name, r in results.items() for m in METRICS})
        log = {k: wandb.Image(f) for k, f in figs.items()}
        log["results"] = wandb.Table(columns=["score"] + METRICS,
                                     data=[[name] + [r[m] for m in METRICS] for name, r in results.items()])
        if transitions:
            cols = ["lc1", "lc2", "n", "frac_changed", "mean_score", "above_thr", "auroc_vs_unchanged"]
            log["transitions_L2_table"] = wandb.Table(columns=cols, data=[[r.get(c) for c in cols] for r in transitions])
        if class_pca_rows:
            log["class_similarity_pca_table"] = wandb.Table(columns=["k", "within", "max_between", "margin"],
                                                            data=class_pca_rows)
        if class_rows:
            # run.summary["val/class_nearest_proto_bal_acc"] = bal_acc
                        log["class_similarity_table"] = wandb.Table(
                columns=["class", "n_val_patches", "within_cos_raw", "within_cos_centred",
                         "max_between_cos_centred", "closest_class"], data=class_rows)
        if sweep:
            cols = list(sweep[0].keys())
            log["pca_sweep_table"] = wandb.Table(columns=cols, data=[[r[c] for c in cols] for r in sweep])
        wandb.log(log)
        run.finish()
    plt.close("all")


if __name__ == "__main__":
    main()