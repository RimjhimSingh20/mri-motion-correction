import numpy as np
from scipy.ndimage import rotate, shift
from typing import Dict, List, Optional


class RandomFlip3D:
    def __init__(self, axes=(0, 1, 2), p: float = 0.5):
        self.axes = axes
        self.p = p

    def __call__(self, sample: Dict) -> Dict:
        for ax in self.axes:
            if np.random.random() < self.p:
                sample = {k: np.flip(v, axis=ax).copy() for k, v in sample.items()}
        return sample


class RandomRotation3D:
    """Small random rotation around the three axis planes."""

    def __init__(self, max_angle: float = 5.0, p: float = 0.5):
        self.max_angle = max_angle
        self.p = p

    def __call__(self, sample: Dict) -> Dict:
        if np.random.random() > self.p:
            return sample
        for ax_pair in ((0, 1), (0, 2), (1, 2)):
            angle = np.random.uniform(-self.max_angle, self.max_angle)
            sample = {
                k: rotate(v, angle, axes=ax_pair, reshape=False, order=1)
                for k, v in sample.items()
            }
        return sample


class SimulateMotion:
    """
    K-space motion simulation: replaces a fraction of phase-encode lines with
    the k-space of a rigidly displaced version of the volume.

    This approximates intra-scan bulk motion during 3D Cartesian acquisition.
    Applied only to the 'input' key — 'target' is left untouched (it becomes
    the clean reference).
    """

    def __init__(
        self,
        corruption_fraction: float = 0.10,
        max_translation: float = 3.0,
        max_rotation_deg: float = 3.0,
        p: float = 0.5,
    ):
        self.corruption_fraction = corruption_fraction
        self.max_translation = max_translation
        self.max_rotation_deg = max_rotation_deg
        self.p = p

    def __call__(self, sample: Dict) -> Dict:
        if np.random.random() > self.p:
            return sample

        vol = sample["input"].copy()
        _, H, _ = vol.shape

        tx, ty, tz = np.random.uniform(-self.max_translation, self.max_translation, 3)
        angle = np.random.uniform(-self.max_rotation_deg, self.max_rotation_deg)

        moved = rotate(vol, angle, axes=(0, 1), reshape=False, order=1)
        moved = shift(moved, [tz, ty, tx], order=1)

        kspace_orig = np.fft.fftn(vol)
        kspace_moved = np.fft.fftn(moved)

        n_corrupt = max(1, int(H * self.corruption_fraction))
        corrupt_lines = np.random.choice(H, n_corrupt, replace=False)

        kspace_corrupted = kspace_orig.copy()
        kspace_corrupted[:, corrupt_lines, :] = kspace_moved[:, corrupt_lines, :]

        corrupted = np.real(np.fft.ifftn(kspace_corrupted)).astype(np.float32)
        sample["input"] = corrupted
        return sample


class Compose:
    def __init__(self, transforms: List):
        self.transforms = transforms

    def __call__(self, sample: Dict) -> Dict:
        for t in self.transforms:
            sample = t(sample)
        return sample


def build_train_transforms(cfg: dict) -> Compose:
    return Compose([
        RandomFlip3D(axes=(0, 1, 2), p=0.5),
        RandomRotation3D(max_angle=5.0, p=0.3),
    ])


def build_val_transforms(cfg: dict) -> Optional[Compose]:
    return None
