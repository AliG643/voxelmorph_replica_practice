"""Small U-Net for direct-displacement or integrated-velocity registration."""
import torch
from torch import nn
from torch.nn import functional as F
from integration import IntegrateVelocityField


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.act(self.conv(x))


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = ConvBlock(2, 16, 2)
        self.block2 = ConvBlock(16, 32, 2)
        self.block3 = ConvBlock(32, 32, 2)

    def forward(self, x):
        f1 = self.block1(x)
        f2 = self.block2(f1)
        return f1, f2, self.block3(f2)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = ConvBlock(64, 32)
        self.block2 = ConvBlock(48, 16)

    def forward(self, features, output_size):
        f1, f2, f3 = features
        x = F.interpolate(f3, size=f2.shape[-2:], mode="nearest")
        x = self.block1(torch.cat((x, f2), dim=1))
        x = F.interpolate(x, size=f1.shape[-2:], mode="nearest")
        x = self.block2(torch.cat((x, f1), dim=1))
        return F.interpolate(x, size=output_size, mode="nearest")


class FlowHead(nn.Module):
    def __init__(self, in_channels=16):
        super().__init__()
        self.flow = nn.Conv2d(in_channels, 2, 3, padding=1)
        nn.init.normal_(self.flow.weight, mean=0, std=1e-5)
        nn.init.zeros_(self.flow.bias)

    def forward(self, x):
        return self.flow(x)


class Voxelmorph(nn.Module):
    def __init__(self, integration_steps=0):
        super().__init__()
        self.integration_steps = integration_steps
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.flow_head = FlowHead(16)
        self.integrator = IntegrateVelocityField(integration_steps)

    def forward(self, moving_image, fixed):
        if moving_image.ndim != 4 or moving_image.shape != fixed.shape or moving_image.shape[1] != 1:
            raise ValueError("Moving and fixed must have matching [B,1,H,W] shapes.")
        if min(moving_image.shape[-2:]) < 8:
            raise ValueError("Image dimensions must be at least 8.")
        features = self.encoder(torch.cat((moving_image, fixed), dim=1))
        field = self.flow_head(self.decoder(features, moving_image.shape[-2:]))
        return self.integrator(field)
