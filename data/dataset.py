import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple


class MRIMotionDataset(Dataset):
    """
    Paired (motion-corrupted, ground-truth) 3D MRI dataset.

    Expected directory layout::

        root/
          motion/   *.nii or *.nii.gz   — degraded volumes
          clean/    *.nii or *.nii.gz   — paired ground-truth volumes

    Accepted subdirectory names (first match wins):
        corrupted side: motion | corrupted | degraded | input
        clean side:     clean  | reference | ground_truth | target
    """

    _CORRUPTED = ("motion", "corrupted", "degraded", "input")
    _CLEAN = ("clean", "reference", "ground_truth", "target")

    def __init__(
        self,
        root_dir: str,
        patch_size: Tuple[int, int, int] = (64, 64, 64),
        patches_per_volume: int = 16,
        transform: Optional[Callable] = None,
        normalize: str = "zscore",
        cache: bool = False,
    ):
        self.root_dir = Path(root_dir)
        self.patch_size = patch_size
        self.patches_per_volume = patches_per_volume
        self.transform = transform
        self.normalize = normalize
        self._cache: Dict[Path, np.ndarray] = {} if cache else None

        self.pairs = self._find_pairs()
        if not self.pairs:
            raise RuntimeError(f"No paired NIfTI volumes found under {root_dir}")

    # ------------------------------------------------------------------
    def _find_pairs(self) -> List[Tuple[Path, Path]]:
        corrupted_dir = next(
            (self.root_dir / s for s in self._CORRUPTED if (self.root_dir / s).is_dir()),
            None,
        )
        clean_dir = next(
            (self.root_dir / s for s in self._CLEAN if (self.root_dir / s).is_dir()),
            None,
        )
        if corrupted_dir is None or clean_dir is None:
            raise RuntimeError(
                f"Could not locate subdirectory pair in {self.root_dir}.\n"
                f"  corrupted side: {self._CORRUPTED}\n"
                f"  clean side:     {self._CLEAN}"
            )

        pairs = []
        for cf in sorted(corrupted_dir.glob("*.nii*")):
            stem = cf.name.replace(".nii.gz", "").replace(".nii", "")
            for ext in (".nii.gz", ".nii"):
                gf = clean_dir / (stem + ext)
                if gf.exists():
                    pairs.append((cf, gf))
                    break
        return pairs

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.pairs) * self.patches_per_volume

    def _load(self, path: Path) -> np.ndarray:
        if self._cache is not None and path in self._cache:
            return self._cache[path]
        vol = nib.load(str(path)).get_fdata(dtype=np.float32)
        if self._cache is not None:
            self._cache[path] = vol
        return vol

    def _normalize(self, vol: np.ndarray) -> np.ndarray:
        if self.normalize == "zscore":
            mask = vol > 0
            if mask.any():
                mu, sigma = vol[mask].mean(), vol[mask].std()
                return (vol - mu) / (sigma + 1e-8)
            return vol
        if self.normalize == "minmax":
            lo, hi = vol.min(), vol.max()
            return (vol - lo) / (hi - lo + 1e-8)
        return vol

    def _random_crop(self, vol: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int]]:
        D, H, W = vol.shape
        pd, ph, pw = self.patch_size
        d0 = np.random.randint(0, max(D - pd, 1))
        h0 = np.random.randint(0, max(H - ph, 1))
        w0 = np.random.randint(0, max(W - pw, 1))
        patch = vol[d0 : d0 + pd, h0 : h0 + ph, w0 : w0 + pw].copy()
        # Reflect-pad if volume is smaller than patch size
        if patch.shape != (pd, ph, pw):
            pad = [(0, max(0, t - a)) for t, a in zip((pd, ph, pw), patch.shape)]
            patch = np.pad(patch, pad, mode="reflect")
        return patch, (d0, h0, w0)

    def _crop_at(self, vol: np.ndarray, origin: Tuple[int, int, int]) -> np.ndarray:
        d0, h0, w0 = origin
        pd, ph, pw = self.patch_size
        patch = vol[d0 : d0 + pd, h0 : h0 + ph, w0 : w0 + pw].copy()
        if patch.shape != (pd, ph, pw):
            pad = [(0, max(0, t - a)) for t, a in zip((pd, ph, pw), patch.shape)]
            patch = np.pad(patch, pad, mode="reflect")
        return patch

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        vol_idx = idx // self.patches_per_volume
        corrupted_path, clean_path = self.pairs[vol_idx]

        corrupted_vol = self._normalize(self._load(corrupted_path))
        clean_vol = self._normalize(self._load(clean_path))

        corrupted_patch, origin = self._random_crop(corrupted_vol)
        clean_patch = self._crop_at(clean_vol, origin)

        sample = {"input": corrupted_patch, "target": clean_patch}
        if self.transform is not None:
            sample = self.transform(sample)

        return {
            "input": torch.from_numpy(sample["input"]).unsqueeze(0).float(),
            "target": torch.from_numpy(sample["target"]).unsqueeze(0).float(),
        }

    # ------------------------------------------------------------------
    def get_volume_pair(self, vol_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return full (un-patched) normalized volumes for evaluation."""
        corrupted_path, clean_path = self.pairs[vol_idx]
        corrupted = torch.from_numpy(self._normalize(self._load(corrupted_path)))
        clean = torch.from_numpy(self._normalize(self._load(clean_path)))
        # shape: [1, 1, D, H, W]
        return corrupted.unsqueeze(0).unsqueeze(0), clean.unsqueeze(0).unsqueeze(0)
