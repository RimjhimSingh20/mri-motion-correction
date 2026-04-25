import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List


# ---------------------------------------------------------------------------
# Kernel helpers
# ---------------------------------------------------------------------------

def _gaussian_kernel_3d(
    kernel_size: int,
    sigma: float,
    device: torch.device,
    n_channels: int,
) -> torch.Tensor:
    coords = torch.arange(kernel_size, dtype=torch.float32, device=device)
    coords = coords - kernel_size // 2
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    kernel = g[:, None, None] * g[None, :, None] * g[None, None, :]
    kernel = kernel / kernel.sum()
    # Shape: [n_channels, 1, K, K, K] — used with groups=n_channels
    return (
        kernel.view(1, 1, kernel_size, kernel_size, kernel_size)
        .expand(n_channels, 1, kernel_size, kernel_size, kernel_size)
        .contiguous()
    )


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def ssim3d(
    pred: torch.Tensor,
    target: torch.Tensor,
    kernel_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    3-D Structural Similarity Index (SSIM).

    Args:
        pred:       [B, C, D, H, W]
        target:     [B, C, D, H, W]
        data_range: peak signal value (1.0 for normalised, 255 for uint8, etc.)
        reduction:  "mean" → scalar | "batch" → [B] | "none" → [B, C, D, H, W]

    Returns:
        SSIM value(s) in [-1, 1].
    """
    assert pred.shape == target.shape, "pred and target must have identical shape"
    B, C = pred.shape[:2]

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    padding = kernel_size // 2

    kernel = _gaussian_kernel_3d(kernel_size, sigma, pred.device, C)

    mu_x = F.conv3d(pred, kernel, padding=padding, groups=C)
    mu_y = F.conv3d(target, kernel, padding=padding, groups=C)

    mu_x2 = mu_x.pow(2)
    mu_y2 = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv3d(pred * pred, kernel, padding=padding, groups=C) - mu_x2
    sigma_y2 = F.conv3d(target * target, kernel, padding=padding, groups=C) - mu_y2
    sigma_xy = F.conv3d(pred * target, kernel, padding=padding, groups=C) - mu_xy

    sigma_x2 = sigma_x2.clamp(min=0.0)
    sigma_y2 = sigma_y2.clamp(min=0.0)

    num = (2.0 * mu_xy + C1) * (2.0 * sigma_xy + C2)
    den = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)

    ssim_map = (num / (den + 1e-10)).clamp(-1.0, 1.0)

    if reduction == "mean":
        return ssim_map.mean()
    if reduction == "batch":
        return ssim_map.mean(dim=[1, 2, 3, 4])
    return ssim_map


def psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Peak Signal-to-Noise Ratio (dB).

    Returns:
        PSNR value(s).  Higher is better.
        Returns 100 dB for numerically identical inputs.
    """
    mse = F.mse_loss(pred, target, reduction="none").mean(dim=[1, 2, 3, 4])
    val = 10.0 * torch.log10(data_range ** 2 / (mse + 1e-10))
    if reduction == "mean":
        return val.mean()
    return val


def nrmse(
    pred: torch.Tensor,
    target: torch.Tensor,
    normalization: str = "euclidean",
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Normalised Root Mean-Square Error.

    Args:
        normalization: "euclidean" (RMS of target) | "min_max" | "mean"
        reduction:     "mean" → scalar | "none" → [B]

    Returns:
        NRMSE value(s).  Lower is better.
    """
    rmse = torch.sqrt(F.mse_loss(pred, target, reduction="none").mean(dim=[1, 2, 3, 4]))

    flat = target.reshape(target.shape[0], -1)
    if normalization == "euclidean":
        norm = torch.sqrt((flat ** 2).mean(dim=1))
    elif normalization == "min_max":
        norm = flat.max(dim=1).values - flat.min(dim=1).values
    else:  # mean
        norm = flat.mean(dim=1).abs()

    val = rmse / (norm + 1e-10)
    if reduction == "mean":
        return val.mean()
    return val


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class MetricTracker:
    """Accumulates per-batch metric values; computes mean ± std at the end."""

    def __init__(self, metrics: List[str] = ("ssim", "psnr", "nrmse")):
        self.metrics = list(metrics)
        self.reset()

    def reset(self):
        self._values: Dict[str, List[float]] = {m: [] for m in self.metrics}

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        data_range: float = 1.0,
    ):
        with torch.no_grad():
            if "ssim" in self.metrics:
                vals = ssim3d(pred, target, data_range=data_range, reduction="batch")
                self._values["ssim"].extend(vals.cpu().tolist())
            if "psnr" in self.metrics:
                vals = psnr(pred, target, data_range=data_range, reduction="none")
                self._values["psnr"].extend(vals.cpu().tolist())
            if "nrmse" in self.metrics:
                vals = nrmse(pred, target, reduction="none")
                self._values["nrmse"].extend(vals.cpu().tolist())

    def compute(self) -> Dict[str, Dict[str, float]]:
        result = {}
        for m, vals in self._values.items():
            if vals:
                arr = np.array(vals)
                result[m] = {"mean": float(arr.mean()), "std": float(arr.std())}
            else:
                result[m] = {"mean": 0.0, "std": 0.0}
        return result

    def summary(self) -> str:
        stats = self.compute()
        return " | ".join(
            f"{m.upper()}: {v['mean']:.4f}±{v['std']:.4f}" for m, v in stats.items()
        )
