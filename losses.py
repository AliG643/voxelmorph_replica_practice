"""Differentiable similarity and displacement regularization."""
import torch
from torch import nn
from torch.nn import functional as F


def mse_loss(fixed, warped):
    return F.mse_loss(warped, fixed)


def smoothness_loss(flow):
    return flow.diff(dim=2).square().mean() + flow.diff(dim=3).square().mean()


class NCCLoss(nn.Module):
    def __init__(self, window_size=9, eps=1e-5):
        super().__init__()
        if window_size < 1 or window_size % 2 != 1:
            raise ValueError("NCC window size must be a positive odd number.")
        self.window_size, self.eps = window_size, eps

    def forward(self, fixed, warped):
        window = torch.ones(1, 1, self.window_size, self.window_size, device=fixed.device, dtype=fixed.dtype)
        def total(x):
            return F.conv2d(x, window, padding=self.window_size // 2)
        count = self.window_size ** 2
        f, w = total(fixed), total(warped)
        covariance = total(fixed * warped) - f * w / count
        vf = (total(fixed.square()) - f.square() / count).clamp_min(0)
        vw = (total(warped.square()) - w.square() / count).clamp_min(0)
        return -(covariance.square() / (vf * vw + self.eps)).mean()
