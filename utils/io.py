from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import nibabel as nib
import torch
import yaml


def load_nifti(path: str) -> Tuple[np.ndarray, nib.Nifti1Image]:
    img = nib.load(str(path))
    return img.get_fdata(dtype=np.float32), img


def save_nifti(
    data: np.ndarray,
    path: str,
    reference_img: Optional[nib.Nifti1Image] = None,
    affine: Optional[np.ndarray] = None,
):
    if reference_img is not None:
        out = nib.Nifti1Image(data, reference_img.affine, reference_img.header)
    else:
        out = nib.Nifti1Image(data, affine if affine is not None else np.eye(4))
    nib.save(out, str(path))


def normalize_volume(vol: np.ndarray, method: str = "zscore") -> np.ndarray:
    if method == "zscore":
        mask = vol > 0
        if mask.any():
            mu, sigma = vol[mask].mean(), vol[mask].std()
            return (vol - mu) / (sigma + 1e-8)
        return vol
    if method == "minmax":
        lo, hi = vol.min(), vol.max()
        return (vol - lo) / (hi - lo + 1e-8)
    return vol


def volume_to_tensor(vol: np.ndarray) -> torch.Tensor:
    """(D, H, W) ndarray → [1, 1, D, H, W] float tensor."""
    return torch.from_numpy(vol).float().unsqueeze(0).unsqueeze(0)


def tensor_to_volume(t: torch.Tensor) -> np.ndarray:
    """Any shape tensor with squeezable leading dims → (D, H, W) ndarray."""
    return t.squeeze().cpu().numpy()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
