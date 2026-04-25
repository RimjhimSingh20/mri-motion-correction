#!/usr/bin/env python3
"""Single-volume inference with sliding-window patch aggregation."""

import argparse
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.evaluator import sliding_window_inference
from models.unet3d import UNet3D
from utils.io import load_nifti, save_nifti, normalize_volume, volume_to_tensor, tensor_to_volume


def parse_args():
    p = argparse.ArgumentParser(description="Correct a single motion-degraded MRI volume")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    p.add_argument("--input", required=True, help="Path to motion-corrupted NIfTI")
    p.add_argument("--output", required=True, help="Path for corrected NIfTI output")
    p.add_argument("--overlap", type=float, default=0.5, help="Sliding-window overlap fraction")
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
    print(f"Loaded checkpoint: {args.checkpoint}")

    vol, ref_img = load_nifti(args.input)
    print(f"Input shape: {vol.shape}  dtype: {vol.dtype}")

    inp = volume_to_tensor(normalize_volume(vol, method="zscore")).to(device)

    patch_size = tuple(cfg["data"]["patch_size"])
    pred = sliding_window_inference(
        model, inp, patch_size=patch_size, overlap=args.overlap, device=device
    )

    pred_np = tensor_to_volume(pred).astype("float32")
    save_nifti(pred_np, args.output, reference_img=ref_img)
    print(f"Saved corrected volume → {args.output}")


if __name__ == "__main__":
    main()
