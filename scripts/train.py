#!/usr/bin/env python3
"""Train the 3D U-Net motion correction model."""

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

# Allow running as a script from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset import MRIMotionDataset
from data.transforms import build_train_transforms, build_val_transforms
from losses.combined import CombinedLoss
from metrics.image_quality import MetricTracker
from models.unet3d import UNet3D
from training.trainer import Trainer


def parse_args():
    p = argparse.ArgumentParser(description="Train MRI motion correction model")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--output-dir", default=None, help="Override output directory")
    p.add_argument("--resume", default=None, help="Checkpoint path to resume from")
    return p.parse_args()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seed_everything(cfg["experiment"]["seed"])

    output_dir = args.output_dir or os.path.join(
        cfg["experiment"]["output_dir"], cfg["experiment"]["name"]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"  {props.name} — {props.total_memory // 1024**3} GB VRAM")

    # ---- Data ---------------------------------------------------------------
    data_cfg = cfg["data"]
    patch_size = tuple(data_cfg["patch_size"])

    train_ds = MRIMotionDataset(
        data_cfg["train_dir"],
        patch_size=patch_size,
        patches_per_volume=data_cfg.get("patches_per_volume_train", 16),
        transform=build_train_transforms(cfg),
        normalize="zscore",
    )
    val_ds = MRIMotionDataset(
        data_cfg["val_dir"],
        patch_size=patch_size,
        patches_per_volume=data_cfg.get("patches_per_volume_val", 4),
        transform=build_val_transforms(cfg),
        normalize="zscore",
    )
    print(f"Train: {len(train_ds.pairs)} volumes  ({len(train_ds)} patches)")
    print(f"Val:   {len(val_ds.pairs)} volumes  ({len(val_ds)} patches)")

    num_workers = data_cfg.get("num_workers", 4)
    pin_memory = data_cfg.get("pin_memory", True) and device.type == "cuda"
    batch_size = cfg["training"]["batch_size"]

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    # ---- Model --------------------------------------------------------------
    model_cfg = cfg["model"]
    model = UNet3D(
        in_channels=model_cfg["in_channels"],
        out_channels=model_cfg["out_channels"],
        base_features=model_cfg["base_features"],
        depth=model_cfg["depth"],
        dropout=model_cfg.get("dropout", 0.1),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    # ---- Loss / optimiser ---------------------------------------------------
    loss_cfg = cfg.get("loss", {})
    criterion = CombinedLoss(
        l1_weight=loss_cfg.get("l1_weight", 1.0),
        ssim_weight=loss_cfg.get("ssim_weight", 0.5),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    # ---- Trainer ------------------------------------------------------------
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        metric_tracker=MetricTracker(),
        cfg=cfg,
        output_dir=output_dir,
        device=device,
    )

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        trainer.scheduler.load_state_dict(ckpt["scheduler_state"])
        trainer.scaler.load_state_dict(ckpt["scaler_state"])
        print(f"Resumed from {args.resume} (epoch {ckpt['epoch']})")

    trainer.fit()


if __name__ == "__main__":
    main()
