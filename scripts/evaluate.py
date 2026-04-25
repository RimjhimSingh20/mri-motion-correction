#!/usr/bin/env python3
"""Evaluate a trained model on the test set with sliding-window inference."""

import argparse
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset import MRIMotionDataset
from evaluation.evaluator import Evaluator
from models.unet3d import UNet3D


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate motion correction model")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    p.add_argument("--data-dir", default=None, help="Override test data directory")
    p.add_argument("--output-dir", default="outputs/eval")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(args.checkpoint, map_location=device)
    model_cfg = cfg["model"]
    model = UNet3D(
        in_channels=model_cfg["in_channels"],
        out_channels=model_cfg["out_channels"],
        base_features=model_cfg["base_features"],
        depth=model_cfg["depth"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint (epoch {ckpt.get('epoch', '?')})")

    data_dir = args.data_dir or cfg["data"]["test_dir"]
    test_ds = MRIMotionDataset(data_dir, normalize="zscore")

    evaluator = Evaluator(
        model=model,
        dataset=test_ds,
        cfg=cfg,
        output_dir=args.output_dir,
        device=device,
    )
    evaluator.evaluate()


if __name__ == "__main__":
    main()
