"""
Rigid-body intra-scan motion simulation via k-space corruption.

For a 3-D Cartesian acquisition the phase-encode (PE) lines are acquired
sequentially.  If the subject moves between groups of lines the k-space
is inconsistent, which causes ghosting / blurring in the reconstructed image.

This module models that by:
  1. Taking a clean 3-D volume.
  2. Generating one or more randomly-displaced copies (rigid transform).
  3. Replacing a fraction of PE lines in the original k-space with those of
     the displaced copy.
  4. Reconstructing with IFFT to obtain the corrupted volume.

Severity presets
----------------
  mild:     rot 0–2°, trans 0–1 mm,  1 motion event,  ~8 % lines corrupted
  moderate: rot 0–4°, trans 0–2 mm,  2 motion events, ~20 % lines corrupted
  severe:   rot 0–6°, trans 0–4 mm,  3 motion events, ~40 % lines corrupted
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
from scipy.ndimage import rotate, shift

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------

SEVERITY_PRESETS: Dict[str, Dict] = {
    "mild": {
        "rot_max_deg":  2.0,
        "trans_max_mm": 1.0,
        "n_events":     1,
        "corrupt_frac": 0.08,
    },
    "moderate": {
        "rot_max_deg":  4.0,
        "trans_max_mm": 2.0,
        "n_events":     2,
        "corrupt_frac": 0.20,
    },
    "severe": {
        "rot_max_deg":  6.0,
        "trans_max_mm": 4.0,
        "n_events":     3,
        "corrupt_frac": 0.40,
    },
}


def _rigid_transform(
    volume: np.ndarray,
    angles_deg: Tuple[float, float, float],
    shifts_mm: Tuple[float, float, float],
    voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """Apply a 3-D rigid body transform (ZXY Euler + translation)."""
    moved = volume.astype(np.float32)
    # Three sequential in-plane rotations
    ax_pairs = [(0, 1), (0, 2), (1, 2)]
    for angle, ax in zip(angles_deg, ax_pairs):
        if abs(angle) > 0.01:
            moved = rotate(moved, angle, axes=ax, reshape=False, order=1, mode="nearest")
    # Translation in voxels = mm / voxel_size
    shifts_vox = [s / v for s, v in zip(shifts_mm, voxel_size)]
    if any(abs(s) > 0.01 for s in shifts_vox):
        moved = shift(moved, shifts_vox, order=1, mode="nearest")
    return moved


def simulate_motion(
    volume: np.ndarray,
    severity: str = "moderate",
    voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    pe_axis: int = 1,
    seed: Optional[int] = None,
    return_params: bool = False,
) -> np.ndarray | Tuple[np.ndarray, List[Dict]]:
    """
    Corrupt a clean 3-D volume with simulated rigid-body intra-scan motion.

    Args:
        volume:       (D, H, W) float32 clean volume.
        severity:     One of "mild" | "moderate" | "severe", or pass a custom
                      dict with keys rot_max_deg, trans_max_mm, n_events,
                      corrupt_frac.
        voxel_size:   (dz, dy, dx) voxel spacing in mm — used to convert mm
                      translations to voxel offsets.
        pe_axis:      Axis along which PE lines are indexed (default=1 / coronal).
        seed:         Optional random seed for reproducibility.
        return_params: If True, also return the list of motion parameters used.

    Returns:
        Corrupted volume (same shape and dtype as input), or a (volume, params) tuple.
    """
    if isinstance(severity, str):
        if severity not in SEVERITY_PRESETS:
            raise ValueError(f"severity must be one of {list(SEVERITY_PRESETS)}, got '{severity}'")
        params = SEVERITY_PRESETS[severity].copy()
    else:
        params = dict(severity)

    rng = np.random.default_rng(seed)

    rot_max: float  = params["rot_max_deg"]
    trans_max: float = params["trans_max_mm"]
    n_events: int   = params["n_events"]
    corrupt_frac: float = params["corrupt_frac"]

    n_pe_lines = volume.shape[pe_axis]
    k_orig = np.fft.fftn(volume)
    k_corrupted = k_orig.copy()

    lines_per_event = max(1, int(n_pe_lines * corrupt_frac / n_events))
    all_lines = np.arange(n_pe_lines)
    # Reserve the central k-space (low-spatial-freq) for the clean state
    central_start = int(n_pe_lines * 0.35)
    central_end   = int(n_pe_lines * 0.65)
    peripheral_lines = np.concatenate([
        all_lines[:central_start], all_lines[central_end:]
    ])

    event_records: List[Dict] = []
    available = peripheral_lines.copy()
    rng.shuffle(available)

    for i in range(n_events):
        angles = (
            float(rng.uniform(-rot_max, rot_max)),
            float(rng.uniform(-rot_max, rot_max)),
            float(rng.uniform(-rot_max, rot_max)),
        )
        translations = (
            float(rng.uniform(-trans_max, trans_max)),
            float(rng.uniform(-trans_max, trans_max)),
            float(rng.uniform(-trans_max, trans_max)),
        )

        moved = _rigid_transform(volume, angles, translations, voxel_size)
        k_moved = np.fft.fftn(moved)

        start = i * lines_per_event
        end   = start + lines_per_event
        corrupt_lines = available[start:end]

        idx: List = [slice(None)] * volume.ndim
        idx[pe_axis] = corrupt_lines
        k_corrupted[tuple(idx)] = k_moved[tuple(idx)]

        event_records.append({
            "event":        i,
            "angles_deg":   angles,
            "translations_mm": translations,
            "n_lines_corrupted": len(corrupt_lines),
        })
        log.debug("  Event %d: rot=%s deg, trans=%s mm, %d lines corrupted",
                  i, angles, translations, len(corrupt_lines))

    corrupted = np.real(np.fft.ifftn(k_corrupted)).astype(np.float32)

    if return_params:
        return corrupted, event_records
    return corrupted


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

class MotionSimulator:
    """
    Process a directory of clean NIfTI volumes and produce paired
    clean/corrupted datasets.

    Parameters
    ----------
    clean_dir:
        Directory containing source *.nii or *.nii.gz volumes.
    output_clean_dir:
        Destination for the (copied) clean volumes.
    output_corrupted_dir:
        Destination for the synthesised corrupted volumes.
    severity:
        "mild" | "moderate" | "severe" or a custom params dict.
    n_volumes:
        Maximum number of volumes to process (None = all).
    seed:
        Global RNG seed; each volume gets an incrementing sub-seed.
    """

    def __init__(
        self,
        clean_dir: str,
        output_clean_dir: str,
        output_corrupted_dir: str,
        severity: str = "moderate",
        n_volumes: Optional[int] = None,
        seed: int = 42,
    ):
        self.clean_dir = Path(clean_dir)
        self.out_clean = Path(output_clean_dir)
        self.out_corrupted = Path(output_corrupted_dir)
        self.severity = severity
        self.n_volumes = n_volumes
        self.seed = seed

        self.out_clean.mkdir(parents=True, exist_ok=True)
        self.out_corrupted.mkdir(parents=True, exist_ok=True)

    def _voxel_size(self, img: nib.Nifti1Image) -> Tuple[float, float, float]:
        zooms = img.header.get_zooms()[:3]
        return tuple(float(z) if z > 0 else 1.0 for z in zooms)

    def run(self) -> List[Dict]:
        files = sorted(
            list(self.clean_dir.glob("*.nii")) + list(self.clean_dir.glob("*.nii.gz"))
        )
        if not files:
            raise RuntimeError(f"No NIfTI files found in {self.clean_dir}")

        if self.n_volumes is not None:
            files = files[: self.n_volumes]

        records = []
        print(f"Processing {len(files)} volume(s) with severity='{self.severity}' ...")

        for vol_idx, fpath in enumerate(files):
            print(f"  [{vol_idx + 1}/{len(files)}] {fpath.name}", end="  ", flush=True)

            img = nib.load(str(fpath))
            volume = img.get_fdata(dtype=np.float32)
            vsize = self._voxel_size(img)

            corrupted, params = simulate_motion(
                volume,
                severity=self.severity,
                voxel_size=vsize,
                seed=self.seed + vol_idx,
                return_params=True,
            )

            stem = fpath.name.replace(".nii.gz", "").replace(".nii", "")

            # Save clean (preserve original affine/header)
            clean_path = self.out_clean / f"{stem}.nii.gz"
            nib.save(nib.Nifti1Image(volume, img.affine, img.header), str(clean_path))

            # Save corrupted
            corrupted_path = self.out_corrupted / f"{stem}.nii.gz"
            nib.save(nib.Nifti1Image(corrupted, img.affine, img.header), str(corrupted_path))

            # Quick quality check
            psnr = _psnr(volume, corrupted)
            print(f"shape={volume.shape}  PSNR={psnr:.1f} dB")

            records.append({
                "file": fpath.name,
                "shape": volume.shape,
                "voxel_size_mm": vsize,
                "severity": self.severity,
                "psnr_db": psnr,
                "events": params,
            })

        print(f"\nDone.  Saved to:\n  clean     → {self.out_clean}\n  corrupted → {self.out_corrupted}")
        return records


def _psnr(ref: np.ndarray, pred: np.ndarray) -> float:
    mse = np.mean((ref.astype(np.float64) - pred.astype(np.float64)) ** 2)
    if mse < 1e-10:
        return 100.0
    data_range = ref.max() - ref.min()
    return float(10.0 * np.log10(data_range ** 2 / mse))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Simulate rigid-body motion artifacts in 3-D MRI volumes"
    )
    p.add_argument("clean_dir",            help="Directory with clean NIfTI volumes")
    p.add_argument("output_clean_dir",     help="Output directory for clean copies")
    p.add_argument("output_corrupted_dir", help="Output directory for corrupted volumes")
    p.add_argument("--severity", default="moderate",
                   choices=["mild", "moderate", "severe"],
                   help="Motion severity preset (default: moderate)")
    p.add_argument("--n-volumes", type=int, default=None,
                   help="Limit number of volumes (default: all)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (default: 42)")
    p.add_argument("--verbose", action="store_true")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    sim = MotionSimulator(
        clean_dir=args.clean_dir,
        output_clean_dir=args.output_clean_dir,
        output_corrupted_dir=args.output_corrupted_dir,
        severity=args.severity,
        n_volumes=args.n_volumes,
        seed=args.seed,
    )
    sim.run()
