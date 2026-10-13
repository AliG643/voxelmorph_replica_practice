"""Differentiable 2D sampling using pixel displacement in x,y order."""
import torch
from torch import nn
from torch.nn import functional as F


class SpatialTransformer(nn.Module):
    def __init__(self, interpolation_mode="linear", align_corners=True, device=None):
        super().__init__()
        if interpolation_mode not in {"linear", "nearest"}:
            raise ValueError("Interpolation must be linear or nearest.")
        self.mode = "bilinear" if interpolation_mode == "linear" else "nearest"
        self.align_corners = align_corners

    def forward(self, moving_image, deformation_field):
        if moving_image.ndim != 4 or deformation_field.ndim != 4:
            raise ValueError("Expected image [B,C,H,W] and flow [B,2,H,W].")
        b, _, h, w = moving_image.shape
        if deformation_field.shape != (b, 2, h, w) or min(h, w) < 2:
            raise ValueError("Flow shape must match image; H and W must be at least 2.")
        if moving_image.device != deformation_field.device or moving_image.dtype != deformation_field.dtype:
            raise ValueError("Image and flow must share device and floating-point dtype.")
        y = torch.arange(h, device=moving_image.device, dtype=moving_image.dtype)
        x = torch.arange(w, device=moving_image.device, dtype=moving_image.dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        sample_x = xx[None] + deformation_field[:, 0]
        sample_y = yy[None] + deformation_field[:, 1]
        if self.align_corners:
            sample_x = 2 * sample_x / (w - 1) - 1
            sample_y = 2 * sample_y / (h - 1) - 1
        else:
            sample_x = 2 * (sample_x + 0.5) / w - 1
            sample_y = 2 * (sample_y + 0.5) / h - 1
        grid = torch.stack((sample_x, sample_y), dim=-1)
        return F.grid_sample(moving_image, grid, mode=self.mode,
                             padding_mode="border", align_corners=self.align_corners)
