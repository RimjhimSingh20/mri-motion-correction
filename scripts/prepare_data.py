#!/usr/bin/env python3
"""
prepare_data.py — Extract IXI-T1.tar and run motion simulation.

Usage (full pipeline):
  python scripts/prepare_data.py --n-volumes 10 --severity moderate

Usage (if already extracted):
  python scripts/prepare_data.py --extracted-dir data/raw/IXI-T1 \
      --n-volumes 10 --severity moderate

Steps:
  1. Extract data/raw/IXI-T1.tar  (skipped if already done)
  2. Run MotionSimulator on up to --n-volumes volumes
  3. Print per-volume PSNR summary
"""

import argparse
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.motion_simulator import MotionSimulator

# Canonical paths relative to project root
PROJECT_ROOT  = Path(__file__).parent.parent
RAW_DIR       = PROJECT_ROOT / "data" / "raw"
EXTRACTED_DIR = RAW_DIR / "IXI-T1"
CLEAN_DIR     = PROJECT_ROOT / "data" / "processed" / "clean"
CORRUPTED_DIR = PROJECT_ROOT / "data" / "processed" / "corrupted"
TAR_PATH      = RAW_DIR / "IXI-T1.tar"


def parse_args():
    p = argparse.ArgumentParser(description="Extract IXI-T1 and generate motion-corrupted pairs")
    p.add_argument("--tar-path",      default=str(TAR_PATH),      help="Path to IXI-T1.tar")
    p.add_argument("--extracted-dir", default=str(EXTRACTED_DIR), help="Directory for extracted files")
    p.add_argument("--clean-dir",     default=str(CLEAN_DIR),     help="Output clean dir")
    p.add_argument("--corrupted-dir", default=str(CORRUPTED_DIR), help="Output corrupted dir")
    p.add_argument("--n-volumes",     type=int, default=10,        help="Number of volumes to process")
    p.add_argument("--severity",      default="moderate",
                   choices=["mild", "moderate", "severe"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-extract",  action="store_true",
                   help="Skip extraction if files already exist")
    return p.parse_args()


def extract_tar(tar_path: Path, dest_dir: Path, n_files: int):
    """Extract up to n_files NIfTI files from the tar archive."""
    dest_dir.mkdir(parents=True, exist_ok=True)

    already = list(dest_dir.glob("*.nii.gz")) + list(dest_dir.glob("*.nii"))
    if already:
        print(f"  Found {len(already)} file(s) already extracted in {dest_dir} — skipping.")
        return

    if not tar_path.exists():
        raise FileNotFoundError(
            f"Tar archive not found at {tar_path}.\n"
            "  Run the download first:\n"
            "    curl -L http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-T1.tar"
            f" -o {tar_path}"
        )

    print(f"Extracting up to {n_files} volumes from {tar_path} → {dest_dir} ...")
    extracted = 0
    with tarfile.open(str(tar_path), "r") as tf:
        for member in tf:
            if extracted >= n_files:
                break
            name = member.name
            if not (name.endswith(".nii") or name.endswith(".nii.gz")):
                continue
            # Flatten directory structure
            member.name = Path(member.name).name
            tf.extract(member, path=str(dest_dir))
            print(f"  Extracted: {member.name}")
            extracted += 1

    print(f"  Extracted {extracted} file(s).")


def main():
    args = parse_args()

    tar_path      = Path(args.tar_path)
    extracted_dir = Path(args.extracted_dir)

    # ---- Step 1: extract ------------------------------------------------
    if not args.skip_extract:
        extract_tar(tar_path, extracted_dir, n_files=args.n_volumes)
    else:
        print(f"Skipping extraction (--skip-extract).  Using {extracted_dir}")

    # ---- Step 2: simulate -----------------------------------------------
    sim = MotionSimulator(
        clean_dir=str(extracted_dir),
        output_clean_dir=args.clean_dir,
        output_corrupted_dir=args.corrupted_dir,
        severity=args.severity,
        n_volumes=args.n_volumes,
        seed=args.seed,
    )
    records = sim.run()

    # ---- Step 3: summary ------------------------------------------------
    psnr_values = [r["psnr_db"] for r in records]
    if psnr_values:
        print(
            f"\nSummary ({len(psnr_values)} volumes, severity={args.severity}):\n"
            f"  PSNR  mean={sum(psnr_values)/len(psnr_values):.2f} dB"
            f"  min={min(psnr_values):.2f}  max={max(psnr_values):.2f}"
        )

    print("\nPipeline complete.  Paired dataset ready at:")
    print(f"  clean     → {args.clean_dir}")
    print(f"  corrupted → {args.corrupted_dir}")
    print("\nNext: train the model with")
    print("  python scripts/train.py --config configs/default.yaml")


if __name__ == "__main__":
    main()
