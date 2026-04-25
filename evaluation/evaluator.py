import json
from pathlib import Path
from typing import Tuple

import numpy as np
import nibabel as nib
import pandas as pd
import torch
import torch.nn as nn

from metrics.image_quality import MetricTracker, ssim3d, psnr, nrmse


def sliding_window_inference(
    model: nn.Module,
    volume: torch.Tensor,
    patch_size: Tuple[int, int, int] = (64, 64, 64),
    overlap: float = 0.5,
    batch_size: int = 4,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Patch-based inference with Gaussian-weighted blending.

    Args:
        volume:     [1, C, D, H, W] — single volume (no batch dim flattening).
        patch_size: spatial size of each patch fed to the model.
        overlap:    fraction of patch size used as stride overlap.
        batch_size: how many patches to process in one model call.

    Returns:
        Corrected volume [1, C, D, H, W] on CPU.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    _, C, D, H, W = volume.shape
    pd, ph, pw = patch_size

    step_d = max(1, int(pd * (1.0 - overlap)))
    step_h = max(1, int(ph * (1.0 - overlap)))
    step_w = max(1, int(pw * (1.0 - overlap)))

    # Gaussian weight map for smooth seam blending
    def _gaussian_weight(size):
        coords = [np.linspace(-1.0, 1.0, s) for s in size]
        grids = np.meshgrid(*coords, indexing="ij")
        w = np.exp(-0.5 * sum(g ** 2 for g in grids) / 0.5)
        return torch.from_numpy(w.astype(np.float32))

    patch_weight = _gaussian_weight(patch_size).to(device)

    output = torch.zeros(1, C, D, H, W, device=device)
    weight_sum = torch.zeros(1, 1, D, H, W, device=device)

    d_starts = sorted(set(list(range(0, max(1, D - pd), step_d)) + [max(0, D - pd)]))
    h_starts = sorted(set(list(range(0, max(1, H - ph), step_h)) + [max(0, H - ph)]))
    w_starts = sorted(set(list(range(0, max(1, W - pw), step_w)) + [max(0, W - pw)]))

    pending_patches, pending_coords = [], []

    def _flush():
        if not pending_patches:
            return
        batch = torch.cat(pending_patches, dim=0).to(device)
        with torch.no_grad():
            preds = model(batch)
        for pred, (dd, hh, ww) in zip(preds, pending_coords):
            output[0, :, dd : dd + pd, hh : hh + ph, ww : ww + pw] += pred * patch_weight
            weight_sum[0, :, dd : dd + pd, hh : hh + ph, ww : ww + pw] += patch_weight
        pending_patches.clear()
        pending_coords.clear()

    for d0 in d_starts:
        for h0 in h_starts:
            for w0 in w_starts:
                patch = volume[:, :, d0 : d0 + pd, h0 : h0 + ph, w0 : w0 + pw]
                if patch.shape[2:] != tuple(patch_size):
                    continue
                pending_patches.append(patch)
                pending_coords.append((d0, h0, w0))
                if len(pending_patches) >= batch_size:
                    _flush()

    _flush()

    return (output / (weight_sum + 1e-8)).cpu()


class Evaluator:
    """Runs sliding-window inference over the test set and computes metrics."""

    def __init__(
        self,
        model: nn.Module,
        dataset,
        cfg: dict,
        output_dir: str,
        device: torch.device,
    ):
        self.model = model
        self.dataset = dataset
        self.cfg = cfg
        self.output_dir = Path(output_dir)
        self.device = device
        self.output_dir.mkdir(parents=True, exist_ok=True)

        eval_cfg = cfg.get("evaluation", {})
        self.patch_size = tuple(cfg["data"].get("patch_size", [64, 64, 64]))
        self.overlap = eval_cfg.get("sliding_window_overlap", 0.5)
        self.save_preds = eval_cfg.get("save_predictions", True)
        self.tracker = MetricTracker(eval_cfg.get("metrics", ["ssim", "psnr", "nrmse"]))

    def evaluate(self) -> dict:
        self.tracker.reset()
        records = []

        for vol_idx in range(len(self.dataset.pairs)):
            corrupted, clean = self.dataset.get_volume_pair(vol_idx)

            pred = sliding_window_inference(
                self.model,
                corrupted,
                patch_size=self.patch_size,
                overlap=self.overlap,
                device=self.device,
            )

            pred_dev = pred.to(self.device)
            clean_dev = clean.to(self.device)
            self.tracker.update(pred_dev, clean_dev)

            vol_name = self.dataset.pairs[vol_idx][0].name
            rec = {
                "volume": vol_name,
                "ssim": ssim3d(pred_dev, clean_dev).item(),
                "psnr": psnr(pred_dev, clean_dev).item(),
                "nrmse": nrmse(pred_dev, clean_dev).item(),
            }
            records.append(rec)
            print(
                f"  {vol_name:<40s}  "
                f"SSIM={rec['ssim']:.4f}  "
                f"PSNR={rec['psnr']:.2f} dB  "
                f"NRMSE={rec['nrmse']:.4f}"
            )

            if self.save_preds:
                self._save_nifti(pred, vol_idx)

        summary = self.tracker.compute()

        pd.DataFrame(records).to_csv(
            self.output_dir / "per_volume_metrics.csv", index=False
        )
        with open(self.output_dir / "summary_metrics.json", "w") as f:
            json.dump(summary, f, indent=2)

        print("\nSummary:")
        for m, s in summary.items():
            print(f"  {m.upper():<8s}: {s['mean']:.4f} ± {s['std']:.4f}")

        return summary

    def _save_nifti(self, pred: torch.Tensor, vol_idx: int):
        arr = pred.squeeze().numpy().astype(np.float32)
        stem = self.dataset.pairs[vol_idx][0].name.replace(".nii.gz", "").replace(".nii", "")
        out_path = self.output_dir / f"{stem}_corrected.nii.gz"
        nib.save(nib.Nifti1Image(arr, np.eye(4)), str(out_path))
