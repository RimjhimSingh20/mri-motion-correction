import torch
import torch.nn as nn
from metrics.image_quality import ssim3d


class CombinedLoss(nn.Module):
    """
    Weighted combination of L1 and SSIM losses.

    L_total = w_l1 * L1 + w_ssim * (1 - SSIM)

    SSIM loss is (1 - SSIM) so it is minimised along with L1.
    """

    def __init__(self, l1_weight: float = 1.0, ssim_weight: float = 0.5):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.tensor(0.0, device=pred.device)
        if self.l1_weight > 0:
            loss = loss + self.l1_weight * self.l1(pred, target)
        if self.ssim_weight > 0:
            loss = loss + self.ssim_weight * (1.0 - ssim3d(pred, target))
        return loss
