"""Unit tests for image-quality metrics (ssim3d, psnr, nrmse, MetricTracker)."""

import pytest
import torch

from metrics.image_quality import MetricTracker, nrmse, psnr, ssim3d


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def vol():
    torch.manual_seed(0)
    return torch.rand(2, 1, 32, 32, 32)


# ---------------------------------------------------------------------------
# SSIM
# ---------------------------------------------------------------------------

def test_ssim_identical(vol):
    score = ssim3d(vol, vol).item()
    assert score > 0.99, f"SSIM of identical volumes should be ~1, got {score:.4f}"


def test_ssim_noisy_lower_than_identical(vol):
    noisy = (vol + 0.5 * torch.randn_like(vol)).clamp(0, 1)
    score_clean = ssim3d(vol, vol).item()
    score_noisy = ssim3d(noisy, vol).item()
    assert score_noisy < score_clean


def test_ssim_range(vol):
    noisy = vol + 0.3 * torch.randn_like(vol)
    score = ssim3d(vol, noisy).item()
    assert -1.0 <= score <= 1.0, f"SSIM out of [-1, 1]: {score}"


def test_ssim_batch_reduction(vol):
    noisy = vol + 0.1 * torch.randn_like(vol)
    scores = ssim3d(vol, noisy, reduction="batch")
    assert scores.shape == (vol.shape[0],), f"Expected shape ({vol.shape[0]},)"


def test_ssim_none_reduction(vol):
    noisy = vol + 0.1 * torch.randn_like(vol)
    map_ = ssim3d(vol, noisy, reduction="none")
    assert map_.shape == vol.shape


# ---------------------------------------------------------------------------
# PSNR
# ---------------------------------------------------------------------------

def test_psnr_identical(vol):
    score = psnr(vol, vol).item()
    assert score > 60.0, f"PSNR of identical volumes should be >60 dB, got {score:.2f}"


def test_psnr_decreases_with_noise(vol):
    low = vol + 0.01 * torch.randn_like(vol)
    high = vol + 0.3 * torch.randn_like(vol)
    assert psnr(low, vol).item() > psnr(high, vol).item()


def test_psnr_per_sample_shape(vol):
    scores = psnr(vol, vol + 0.1 * torch.randn_like(vol), reduction="none")
    assert scores.shape == (vol.shape[0],)


# ---------------------------------------------------------------------------
# NRMSE
# ---------------------------------------------------------------------------

def test_nrmse_identical(vol):
    score = nrmse(vol, vol).item()
    assert score < 1e-5, f"NRMSE of identical volumes should be ~0, got {score}"


def test_nrmse_increases_with_noise(vol):
    low = vol + 0.01 * torch.randn_like(vol)
    high = vol + 0.3 * torch.randn_like(vol)
    assert nrmse(low, vol).item() < nrmse(high, vol).item()


def test_nrmse_normalization_modes(vol):
    noisy = vol + 0.1 * torch.randn_like(vol)
    for mode in ("euclidean", "min_max", "mean"):
        val = nrmse(noisy, vol, normalization=mode).item()
        assert val >= 0, f"NRMSE should be non-negative for mode={mode}"


# ---------------------------------------------------------------------------
# MetricTracker
# ---------------------------------------------------------------------------

def test_tracker_accumulates():
    tracker = MetricTracker(["ssim", "psnr", "nrmse"])
    for _ in range(4):
        a = torch.rand(2, 1, 16, 16, 16)
        b = a + 0.1 * torch.randn_like(a)
        tracker.update(a, b)

    result = tracker.compute()
    for m in ("ssim", "psnr", "nrmse"):
        assert m in result
        assert result[m]["mean"] != 0.0


def test_tracker_reset():
    tracker = MetricTracker()
    a = torch.rand(1, 1, 16, 16, 16)
    tracker.update(a, a)
    tracker.reset()
    # After reset, lists are empty
    assert all(len(v) == 0 for v in tracker._values.values())


def test_tracker_summary_str():
    tracker = MetricTracker()
    a = torch.rand(1, 1, 16, 16, 16)
    tracker.update(a, a + 0.05 * torch.randn_like(a))
    s = tracker.summary()
    assert "SSIM" in s and "PSNR" in s and "NRMSE" in s


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
