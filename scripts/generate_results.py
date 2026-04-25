#!/usr/bin/env python3
"""
Generate summary charts, tables, and sample visualisations
for the IXI-T1 motion simulation pipeline.

Outputs (written to results/):
  metrics_table.csv          — per-volume PSNR/NRMSE across severities
  metrics_table.md           — same, as Markdown
  fig1_psnr_by_severity.png  — grouped bar chart
  fig2_nrmse_by_severity.png — grouped bar chart
  fig3_sample_slices.png     — clean vs corrupted slices (3 severities)
  fig4_kspace_corruption.png — k-space magnitude before/after
  fig5_severity_boxplot.png  — PSNR distribution boxplot
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import importlib.util, numpy as np, nibabel as nib, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from skimage.metrics import structural_similarity as ssim2d
import csv, math

# ── load motion_simulator without triggering torch __init__ ──────────────────
spec = importlib.util.spec_from_file_location(
    "motion_simulator",
    str(Path(__file__).parent.parent / "data" / "motion_simulator.py"),
)
ms = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ms)

RESULTS = Path(__file__).parent.parent / "results"
CLEAN_DIR = Path(__file__).parent.parent / "data" / "processed" / "clean"
SEVERITIES = ["mild", "moderate", "severe"]
PALETTE = {"mild": "#4CAF50", "moderate": "#FF9800", "severe": "#F44336"}

# ── helpers ──────────────────────────────────────────────────────────────────

def load_vol(path):
    img = nib.load(str(path))
    return img.get_fdata(dtype=np.float32)

def psnr(ref, pred):
    mse = np.mean((ref.astype(np.float64) - pred.astype(np.float64)) ** 2)
    if mse < 1e-10:
        return 100.0
    return 10.0 * np.log10((ref.max() - ref.min()) ** 2 / mse)

def nrmse(ref, pred):
    rmse = np.sqrt(np.mean((ref - pred) ** 2))
    norm = np.sqrt(np.mean(ref ** 2))
    return float(rmse / (norm + 1e-10))

def ssim_mid_slice(ref, pred, axis=1):
    sl = ref.shape[axis] // 2
    r = np.take(ref,  sl, axis=axis)
    p = np.take(pred, sl, axis=axis)
    drange = float(r.max() - r.min())
    return ssim2d(r, p, data_range=drange)

# ── collect metrics ──────────────────────────────────────────────────────────
files = sorted(CLEAN_DIR.glob("*.nii.gz"))
print(f"Scoring {len(files)} volumes × {len(SEVERITIES)} severities …")

rows = []
for i, fpath in enumerate(files):
    vol = load_vol(fpath)
    row = {"volume": fpath.name.replace("-T1.nii.gz", "")}
    for sev in SEVERITIES:
        corrupted, _ = ms.simulate_motion(vol, severity=sev, seed=i, return_params=True)
        row[f"psnr_{sev}"]  = round(psnr(vol, corrupted), 2)
        row[f"nrmse_{sev}"] = round(nrmse(vol, corrupted), 4)
        row[f"ssim_{sev}"]  = round(ssim_mid_slice(vol, corrupted), 4)
    rows.append(row)
    print(f"  {i+1:2d}/{len(files)}  {row['volume']}")

# ── CSV ──────────────────────────────────────────────────────────────────────
csv_path = RESULTS / "metrics_table.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader(); w.writerows(rows)
print(f"Saved {csv_path}")

# ── Markdown table ────────────────────────────────────────────────────────────
md_lines = []
header = ["Volume"]
for sev in SEVERITIES:
    header += [f"PSNR {sev} (dB)", f"SSIM {sev}", f"NRMSE {sev}"]
md_lines.append("| " + " | ".join(header) + " |")
md_lines.append("| " + " | ".join(["---"] * len(header)) + " |")
for r in rows:
    cols = [r["volume"]]
    for sev in SEVERITIES:
        cols += [str(r[f"psnr_{sev}"]), str(r[f"ssim_{sev}"]), str(r[f"nrmse_{sev}"])]
    md_lines.append("| " + " | ".join(cols) + " |")

# Summary row
md_lines.append("| " + " | ".join(["---"] * len(header)) + " |")
means = ["**Mean**"]
for sev in SEVERITIES:
    means += [
        f"**{np.mean([r[f'psnr_{sev}'] for r in rows]):.2f}**",
        f"**{np.mean([r[f'ssim_{sev}'] for r in rows]):.4f}**",
        f"**{np.mean([r[f'nrmse_{sev}'] for r in rows]):.4f}**",
    ]
md_lines.append("| " + " | ".join(means) + " |")

md_path = RESULTS / "metrics_table.md"
md_path.write_text("\n".join(md_lines))
print(f"Saved {md_path}")

# ── Fig 1: PSNR grouped bar chart ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5))
x = np.arange(len(rows))
w = 0.26
for k, sev in enumerate(SEVERITIES):
    vals = [r[f"psnr_{sev}"] for r in rows]
    bars = ax.bar(x + (k - 1) * w, vals, w, label=sev.capitalize(),
                  color=PALETTE[sev], alpha=0.85, edgecolor="white")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{v:.1f}", ha="center", va="bottom", fontsize=6.5)

ax.set_xticks(x)
ax.set_xticklabels([r["volume"].split("-")[0] for r in rows], rotation=30, ha="right")
ax.set_ylabel("PSNR (dB)  ↑ better")
ax.set_title("PSNR by Volume and Motion Severity  —  IXI-T1 Dataset")
ax.legend(title="Severity")
ax.grid(axis="y", alpha=0.4)
ax.set_ylim(0, max(r[f"psnr_mild"] for r in rows) + 8)
plt.tight_layout()
fig.savefig(RESULTS / "fig1_psnr_by_severity.png", dpi=150)
plt.close(); print("Saved fig1")

# ── Fig 2: NRMSE grouped bar chart ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5))
for k, sev in enumerate(SEVERITIES):
    vals = [r[f"nrmse_{sev}"] for r in rows]
    bars = ax.bar(x + (k - 1) * w, vals, w, label=sev.capitalize(),
                  color=PALETTE[sev], alpha=0.85, edgecolor="white")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f"{v:.3f}", ha="center", va="bottom", fontsize=6.5)

ax.set_xticks(x)
ax.set_xticklabels([r["volume"].split("-")[0] for r in rows], rotation=30, ha="right")
ax.set_ylabel("NRMSE  ↓ better")
ax.set_title("NRMSE by Volume and Motion Severity  —  IXI-T1 Dataset")
ax.legend(title="Severity")
ax.grid(axis="y", alpha=0.4)
plt.tight_layout()
fig.savefig(RESULTS / "fig2_nrmse_by_severity.png", dpi=150)
plt.close(); print("Saved fig2")

# ── Fig 3: Sample slices — one volume, 3 severities ─────────────────────────
sample_vol = load_vol(files[0])
sl_idx = sample_vol.shape[1] // 2

fig = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.05, wspace=0.05)

def show(ax, arr, title="", cmap="gray", vmin=None, vmax=None):
    ax.imshow(arr.T, origin="lower", cmap=cmap,
              vmin=vmin or arr.min(), vmax=vmax or arr.max())
    ax.set_title(title, fontsize=9, pad=3)
    ax.axis("off")

clean_sl = np.take(sample_vol, sl_idx, axis=1)
vmin, vmax = clean_sl.min(), clean_sl.max()

for row_idx, sev in enumerate(SEVERITIES):
    corrupted, _ = ms.simulate_motion(sample_vol, severity=sev, seed=0, return_params=True)
    corr_sl = np.take(corrupted, sl_idx, axis=1)
    residual = np.abs(clean_sl - corr_sl)

    show(fig.add_subplot(gs[row_idx, 0]), clean_sl,
         "Clean" if row_idx == 0 else "", vmin=vmin, vmax=vmax)
    show(fig.add_subplot(gs[row_idx, 1]), corr_sl,
         f"Corrupted ({sev})", vmin=vmin, vmax=vmax)
    show(fig.add_subplot(gs[row_idx, 2]), residual,
         "|Residual|" if row_idx == 0 else "", cmap="hot", vmin=0, vmax=residual.max())

    # PSNR annotation
    p = psnr(sample_vol, corrupted)
    s = ssim_mid_slice(sample_vol, corrupted)
    ax_info = fig.add_subplot(gs[row_idx, 3])
    ax_info.axis("off")
    ax_info.text(0.5, 0.5,
        f"Severity: {sev.upper()}\n\nPSNR:  {p:.1f} dB\nSSIM:  {s:.4f}\nNRMSE: {nrmse(sample_vol, corrupted):.4f}",
        ha="center", va="center", fontsize=11,
        bbox=dict(boxstyle="round,pad=0.6", facecolor=PALETTE[sev], alpha=0.25))

fig.suptitle(f"Motion Artifact Simulation  —  {files[0].name.replace('-T1.nii.gz','')}  (coronal mid-slice)",
             fontsize=12, y=1.01)
plt.tight_layout()
fig.savefig(RESULTS / "fig3_sample_slices.png", dpi=150, bbox_inches="tight")
plt.close(); print("Saved fig3")

# ── Fig 4: K-space magnitude (log scale) ─────────────────────────────────────
sample_small = sample_vol[::2, ::2, sample_vol.shape[2]//2]  # 2D slice
k_clean = np.fft.fftshift(np.fft.fft2(sample_small))

corrupted_mod, _ = ms.simulate_motion(sample_vol, severity="moderate", seed=0, return_params=True)
sample_corr = corrupted_mod[::2, ::2, corrupted_mod.shape[2]//2]
k_corr = np.fft.fftshift(np.fft.fft2(sample_corr))

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
kw = dict(cmap="inferno", origin="lower")
im0 = axes[0].imshow(np.log1p(np.abs(k_clean)).T, **kw)
axes[0].set_title("K-space: Clean"); axes[0].axis("off")
plt.colorbar(im0, ax=axes[0], fraction=0.046)

im1 = axes[1].imshow(np.log1p(np.abs(k_corr)).T, **kw)
axes[1].set_title("K-space: Moderate Corruption"); axes[1].axis("off")
plt.colorbar(im1, ax=axes[1], fraction=0.046)

diff = np.log1p(np.abs(k_corr - k_clean))
im2 = axes[2].imshow(diff.T, cmap="hot", origin="lower")
axes[2].set_title("|K-space Difference|"); axes[2].axis("off")
plt.colorbar(im2, ax=axes[2], fraction=0.046)

fig.suptitle("K-space Magnitude (log scale)  —  Axial Slice", fontsize=12)
plt.tight_layout()
fig.savefig(RESULTS / "fig4_kspace_corruption.png", dpi=150)
plt.close(); print("Saved fig4")

# ── Fig 5: Boxplot of PSNR / SSIM / NRMSE across volumes ─────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 5))
metrics_info = [
    ("psnr",  "PSNR (dB)  ↑ better",  True),
    ("ssim",  "SSIM  ↑ better",       True),
    ("nrmse", "NRMSE  ↓ better",      False),
]
for ax, (key, ylabel, higher_better) in zip(axes, metrics_info):
    data = [[r[f"{key}_{sev}"] for r in rows] for sev in SEVERITIES]
    bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                    medianprops=dict(color="black", linewidth=2))
    for patch, sev in zip(bp["boxes"], SEVERITIES):
        patch.set_facecolor(PALETTE[sev]); patch.set_alpha(0.75)
    ax.set_xticklabels([s.capitalize() for s in SEVERITIES])
    ax.set_ylabel(ylabel); ax.set_xlabel("Severity")
    ax.grid(axis="y", alpha=0.4)
    means = [np.mean(d) for d in data]
    ax.plot(range(1, 4), means, "D--", color="navy", markersize=5, label="Mean")
    ax.legend(fontsize=8)

fig.suptitle("Image Quality Metrics Across 10 IXI-T1 Volumes  —  All Severities", fontsize=12)
plt.tight_layout()
fig.savefig(RESULTS / "fig5_severity_boxplot.png", dpi=150)
plt.close(); print("Saved fig5")

print("\n✓ All results written to results/")
