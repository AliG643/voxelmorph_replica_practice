"""Scaling-and-squaring integration of stationary velocity."""
from torch import nn
from spatial_transformer import SpatialTransformer


class IntegrateVelocityField(nn.Module):
    def __init__(self, steps=5):
        super().__init__()
        if not isinstance(steps, int) or steps < 0:
            raise ValueError("Integration steps must be a nonnegative integer.")
        self.steps = steps
        self.transformer = SpatialTransformer()

    def forward(self, velocity):
        displacement = velocity / (2 ** self.steps)
        for _ in range(self.steps):
            displacement = displacement + self.transformer(displacement, displacement)
        return displacement
