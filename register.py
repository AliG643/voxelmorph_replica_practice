"""Predict and export one image-pair registration."""
import torch
from checkpoint import choose_device, load_model
from image_io import load_image, save_registration
from spatial_transformer import SpatialTransformer
from visualize import plot_registration, plot_flow_magnitude, plot_warped_grid
from pathlib import Path


def predict_pair(model, moving, fixed, device):
    moving, fixed = moving[None].to(device), fixed[None].to(device)
    with torch.no_grad():
        flow = model(moving, fixed)
        warped = SpatialTransformer().to(device)(moving, flow)
    return warped[0, 0].cpu().numpy(), flow[0].cpu().numpy()


def save_pair_figures(moving, fixed, warped, flow, output_dir):
    output_dir = Path(output_dir)
    plot_registration(moving, fixed, warped, output_dir/"comparison.png")
    plot_flow_magnitude(flow, output_dir/"displacement.png")
    yy, xx = torch.meshgrid(torch.arange(moving.shape[0]), torch.arange(moving.shape[1]), indexing="ij")
    grid = ((yy % 12 == 0) | (xx % 12 == 0)).float()
    with torch.no_grad():
        bent = SpatialTransformer()(grid[None, None], torch.from_numpy(flow)[None])[0, 0].numpy()
    plot_warped_grid(bent, output_dir/"grid.png")


def register(moving_path, fixed_path, checkpoint_path, output_dir, device="auto"):
    device = choose_device(device)
    model, _ = load_model(checkpoint_path, device)
    moving, fixed = load_image(moving_path), load_image(fixed_path)
    warped, flow = predict_pair(model, moving, fixed, device)
    m, f = moving[0].numpy(), fixed[0].numpy()
    save_registration(output_dir, m, f, warped, flow, fixed_path)
    save_pair_figures(m, f, warped, flow, output_dir)
    print(f"Saved registration to {Path(output_dir).resolve()}")
