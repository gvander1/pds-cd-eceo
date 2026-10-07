"""Evaluate a saved checkpoint. Never trains.

    python scripts/evaluate.py --ckpt checkpoints/<run-name>.pt
    python scripts/evaluate.py --ckpt checkpoints/<run-name>.pt --split test

Protocol (same as the notebook): for each signal (model / raw), the threshold maximising F1
on the TRAIN set (without augmentation) is applied to the evaluated split.
Logs to W&B: thresholds, F1 / precision / recall / AUROC / AP, predicted vs true positive
fractions, error histograms and example predictions.
"""
import argparse

import torch
import wandb
from torch.utils.data import DataLoader

from pds_cd.config import ROOT
from pds_cd.datasets import build_dataset
from pds_cd.metrics import METHODS, best_f1_threshold, collect_errors, scores_at_threshold, separability
from pds_cd.model import build_models
from pds_cd.utils import fixed_rng, get_device
from pds_cd.viz import error_histograms, pick_indices, predict_samples, visualize_predictions


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split", default="val", choices=["val", "test"])
    args = p.parse_args()

    device = get_device()
    ckpt = torch.load(args.ckpt, map_location=device) 
    cfg = ckpt["cfg"]  # exactly the config used for training
    enc, tr = build_models(cfg, device) #rebuilds the encoder and transformer models with the same architecture as used in training
    tr.load_state_dict(ckpt["transformer"]) #replaces the transformer's random initial weights with the trained weights

    run = wandb.init(project=cfg["project"], job_type="eval", dir=ROOT,
                     group=f"{cfg['dataset']}-{cfg['encoder']}", #same group as the training so they sit together in wandb
                     name=f"eval-{args.split}-{ckpt['train_run_name']}",
                     tags=[cfg["dataset"], cfg["encoder"], args.split],
                     config={**cfg, "eval_split": args.split, "train_run_id": ckpt["train_run_id"]})

    #this part builds the two data source needed for the evaluation: the trainset to choose the threshold and the val to measure performance
    #loader: given a split name (train, val, test) and optionally an augmentation flag, it creates a dataset (access one sample with ds[6]) and a dataloader and returns both
    def loader(split, augment=None): 
        ds = build_dataset(cfg, split, augment=augment)
        return ds, DataLoader(ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])

    _, train_loader = loader("train", augment=False)  # threshold chosen on non-augmented train
    eval_ds, eval_loader = loader(args.split)

    with fixed_rng(cfg["eval_seed"]):
        tr_err = collect_errors(train_loader, enc, tr, device) 
    with fixed_rng(cfg["eval_seed"]):
        ev_err = collect_errors(eval_loader, enc, tr, device)

    summary, thresholds = {}, {}
    for m in METHODS:
        thr, f1_train = best_f1_threshold(tr_err[m], tr_err["labels"])
        thresholds[m] = thr
        summary[f"train/threshold_{m}"] = thr
        summary[f"train/f1_{m}"] = f1_train
        for k, v in separability(tr_err[m], tr_err["labels"]).items():
            summary[f"train/{k}_{m}"] = v
        for k, v in scores_at_threshold(ev_err[m], ev_err["labels"], thr).items():
            summary[f"{args.split}/{k}_{m}"] = v
        for k, v in separability(ev_err[m], ev_err["labels"]).items():
            summary[f"{args.split}/{k}_{m}"] = v
    summary[f"{args.split}/feat_loss"] = ev_err["feat_loss"]
    run.summary.update(summary)

    for k, v in sorted(summary.items()):
        print(f"{k:32s} {v:.4f}")

    # --- figures ---  (every title says which dataset / encoder / seed it comes from)
    tag = f"{cfg['dataset']} | {cfg['encoder']} | seed {cfg['seed']}"
    figs = {"hist/train": error_histograms(tr_err, thresholds, f"{tag} | train"),
            f"hist/{args.split}": error_histograms(ev_err, thresholds, f"{tag} | {args.split}")}

    with fixed_rng(cfg["eval_seed"]):
        idx = pick_indices(eval_ds, cfg["n_vis"], seed=cfg["eval_seed"])
        im1, im2, labels, err_model, err_raw = predict_samples(enc, tr, eval_ds, device, idx)
    for m, err in (("model", err_model), ("raw", err_raw)):
        figs[f"examples/{m}"] = visualize_predictions(im1, im2, labels, err, threshold=thresholds[m],
                                                      indices=idx, title=f"{tag} | {args.split} | {m}")

    class_labels = {0: "no change", 1: "change"}
    overlays = [wandb.Image(im2[i].permute(1, 2, 0).numpy(), caption=f"{cfg['dataset']} #{idx[i]}", masks={
        "ground_truth": {"mask_data": labels[i].numpy().astype("uint8"), "class_labels": class_labels},
        "prediction": {"mask_data": (err_model[i] > thresholds["model"]).numpy().astype("uint8"),
                       "class_labels": class_labels},
    }) for i in range(len(idx))]

    wandb.log({**{k: wandb.Image(f) for k, f in figs.items()}, "examples/overlay_model": overlays})
    run.finish()


if __name__ == "__main__":
    main()