"""Train the FeatureTransformer (encoder frozen) and save a checkpoint. No thresholds, no F1.

    python scripts/train.py --data hiucd
    python scripts/train.py --data levir --epochs 30 --seed 1 --augment false

Logs to W&B every epoch: train loss, lr, val feature loss, val AUROC/AP (model and raw,
on a fixed pixel subsample). Saves checkpoints/<run-name>.pt with the weights AND the config.
"""
import secrets
import torch
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader

from pds_cd.config import ROOT, load_config
from pds_cd.datasets import build_dataset
from pds_cd.metrics import METHODS, collect_errors, separability
from pds_cd.model import build_models
from pds_cd.utils import fixed_rng, get_device, seed_everything


def main():
    cfg = load_config() #builds the config dict
    run_id = secrets.token_hex(4)  # short unique id, also used in the checkpoint file name
    run = wandb.init(project=cfg["project"], config=cfg, job_type="train", dir=ROOT,
                     group=f"{cfg['dataset']}-{cfg['encoder']}", id=run_id,
                     name=f"{cfg['dataset']}-{cfg['encoder']}-seed{cfg['seed']}-{run_id}")
    cfg = dict(wandb.config)  # a sweep may have overridden values --> reads the config from WandB
    seed_everything(cfg["seed"]) #fixies the transformers initial weights, the dropout mask and the random train crops so the same seed gives the same run 
    device = get_device()

    train_ds = build_dataset(cfg, "train") #creates the datasets, train has augmentaition if augment=true, val never does, for HiUCD both use everything=true so each sample is (im1, im2, changemap, lc1, lc2)
    val_ds = build_dataset(cfg, "val")
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=True,
                              num_workers=cfg["num_workers"],
                              generator=torch.Generator().manual_seed(cfg["seed"])) #so that shuffle order and worker (eugmentation and crops) seeds depend only on seed
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False,
                            num_workers=cfg["num_workers"])
    print(f"train: {len(train_ds)} samples, {len(train_loader)} batches | val: {len(val_ds)} samples")

    enc, tr = build_models(cfg, device)
    opt = torch.optim.AdamW(tr.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], pct_start=cfg["pct_start"],
                                                total_steps=cfg["epochs"] * len(train_loader)) #learning rate schedule, learning rate raises up to 10% of the 

    for ep in range(cfg["epochs"]):
        tr.train() #turns on trainnf mode : dropout active
        enc.eval() #keeps the encoder in inference mode because the evaluation at the end of each epoch also calls .eval(), and the next epoch must switch the transformer back to .train()
        tot = 0.0 #accumulates the loss to average it over the epoch 
        for batch in train_loader:
            im1, im2 = batch[0].to(device), batch[1].to(device) #takes t1 and t2 and moves them to the GPU
            with torch.no_grad(): #encodes both dates --> each image put into the feature embedding space, no gradient computed for the encoder
                x, y = enc(im1), enc(im2)
            loss = F.mse_loss(tr(x), y) #transformer takes the t1 embedding and tries to predict the t2 embedding, loss is the mean squared error between the predicted and actual t2 embeddings
            opt.zero_grad() #clears the gradients of the previous batch
            loss.backward() #compute the gradients of the loss w.r.t. the transformer parameters
            opt.step() #updates the weights 
            sched.step() #move the learning rate one step alorg the OneCycle schedule
            tot += loss.item() #.item() turns the 1-element tensor into a plain Python float

        with fixed_rng(cfg["eval_seed"]):  # same val crops every epoch, training RNG untouched
            val = collect_errors(val_loader, enc, tr, device,
                                 max_pixels=cfg["epoch_eval_pixels"], seed=cfg["eval_seed"])
        #collect_errors runs the whole val set and retiurns val["model"]: per-pixel error ‖x̂ − y‖² (with transformer), val["raw"]: per-pixel error ‖x − y‖² (without transformer), val["labels"]: ground truth labels, val["feat_loss"]: mean of val["model"] over all pixels
        log = {"epoch": ep, "train/loss": tot / len(train_loader),
               "lr": sched.get_last_lr()[0], "val/feat_loss": val["feat_loss"]}
        #log builds the dictionary of metrics to log to WandB, including the average training loss, the current learning rate, and the validation feature loss
        for m in METHODS:
            for k, v in separability(val[m], val["labels"]).items():
                log[f"val/{k}_{m}"] = v
        wandb.log(log)
        print(f"epoch {ep:3d} | train {log['train/loss']:.4f} | val {log['val/feat_loss']:.4f} | "
              f"AUROC model {log['val/auroc_model']:.3f} raw {log['val/auroc_raw']:.3f}")

    ckpt_dir = ROOT / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    run_name = run.name or run.id  # run.name is None in offline mode
    ckpt_path = ckpt_dir / f"{run_name}.pt"
    torch.save({"transformer": tr.state_dict(), "cfg": cfg, "train_run_id": run.id,
                "train_run_name": run_name}, ckpt_path)
    run.summary["checkpoint"] = str(ckpt_path)
    if cfg["log_model"]:
        art = wandb.Artifact(f"transformer-{cfg['dataset']}-{cfg['encoder']}", type="model",
                             metadata={"seed": cfg["seed"]})
        art.add_file(str(ckpt_path))
        run.log_artifact(art)
    print(f"checkpoint saved: {ckpt_path}\nnext: python scripts/evaluate.py --ckpt {ckpt_path}")
    run.finish()


if __name__ == "__main__":
    main()
