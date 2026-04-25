from typing import Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch


def _as_np(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.squeeze().cpu().numpy()
    return np.squeeze(x)


def plot_slice_comparison(
    corrupted,
    corrected,
    clean,
    axis: int = 1,
    slice_idx: Optional[int] = None,
    title: str = "",
    save_path: Optional[str] = None,
    vrange: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """
    Four-panel figure: corrupted | corrected | ground truth | |residual|.

    Args:
        axis:      anatomical axis to slice (0=axial, 1=coronal, 2=sagittal).
        slice_idx: slice index; defaults to the mid-plane.
    """
    corrupted, corrected, clean = _as_np(corrupted), _as_np(corrected), _as_np(clean)

    if slice_idx is None:
        slice_idx = corrupted.shape[axis] // 2

    def _take(vol):
        return np.take(vol, slice_idx, axis=axis)

    slc_corrupted = _take(corrupted)
    slc_corrected = _take(corrected)
    slc_clean = _take(clean)
    residual = np.abs(slc_clean - slc_corrected)

    vmin, vmax = vrange if vrange else (clean.min(), clean.max())

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    panels = [
        (slc_corrupted, "Corrupted",     "gray", vmin,         vmax),
        (slc_corrected, "Corrected",     "gray", vmin,         vmax),
        (slc_clean,     "Ground Truth",  "gray", vmin,         vmax),
        (residual,      "|GT − Pred|",   "hot",  0.0,  residual.max() + 1e-8),
    ]
    for ax, (img, ttl, cmap, mn, mx) in zip(axes, panels):
        im = ax.imshow(img.T, origin="lower", cmap=cmap, vmin=mn, vmax=mx)
        ax.set_title(ttl)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if title:
        fig.suptitle(title, fontsize=13)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_training_curves(history: Dict, save_path: Optional[str] = None) -> plt.Figure:
    """Loss, SSIM, and PSNR curves from the training history dict."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    ax.plot(history.get("train_loss", []), label="Train")
    ax.plot(history.get("val_loss", []), label="Val")
    ax.set_title("Loss")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(True)

    axes[1].plot(history.get("val_ssim", []))
    axes[1].set_title("Val SSIM")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(True)

    axes[2].plot(history.get("val_psnr", []))
    axes[2].set_title("Val PSNR (dB)")
    axes[2].set_xlabel("Epoch")
    axes[2].grid(True)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_metric_comparison(
    baseline_metrics: Dict,
    model_metrics: Dict,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Bar chart comparing baseline vs DL model metrics."""
    metrics = list(model_metrics.keys())
    x = np.arange(len(metrics))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        x - w / 2,
        [baseline_metrics.get(m, {}).get("mean", 0) for m in metrics],
        w,
        label="Baseline",
        alpha=0.8,
    )
    ax.bar(
        x + w / 2,
        [model_metrics[m]["mean"] for m in metrics],
        w,
        yerr=[model_metrics[m]["std"] for m in metrics],
        label="DL Model",
        alpha=0.8,
        capsize=4,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.legend()
    ax.set_title("Baseline vs DL Model")
    ax.grid(True, axis="y")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
