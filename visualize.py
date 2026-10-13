"""Unbranded scientific figures; transpose consistently for OASIS slices."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_figure(fig, save_path):
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_registration(moving, fixed, warped, save_path):
    if np.ndim(moving) != 2 or moving.shape != fixed.shape or fixed.shape != warped.shape:
        raise ValueError("Images must have matching [H,W] shapes.")
    errors = [np.abs(moving-fixed), np.abs(warped-fixed)]
    limit = max(*(float(error.max()) for error in errors), 1e-8)
    fig, axes = plt.subplots(1, 5, figsize=(17, 4), layout="constrained")
    titles = ["Moving", "Fixed", "Warped moving", "Absolute error before", "Absolute error after"]
    for i, (ax, array, title) in enumerate(zip(axes, [moving, fixed, warped, *errors], titles)):
        ax.imshow(array.T, origin="lower", cmap="gray", vmin=0, vmax=1 if i < 3 else limit)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    save_figure(fig, save_path)


def plot_flow_magnitude(flow, save_path):
    magnitude = np.sqrt(np.sum(np.asarray(flow) ** 2, axis=0))
    fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
    image = ax.imshow(magnitude.T, origin="lower", cmap="viridis", vmin=0)
    ax.set(title="Displacement magnitude", xlabel="Array axis 0 (pixels)", ylabel="Array axis 1 (pixels)")
    fig.colorbar(image, ax=ax, label="Displacement (pixels)")
    save_figure(fig, save_path)


def plot_warped_grid(warped_grid, save_path):
    fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
    ax.imshow(warped_grid.T, origin="lower", cmap="gray", vmin=0, vmax=1)
    ax.set(title="Warped coordinate grid", xlabel="Array axis 0 (pixels)", ylabel="Array axis 1 (pixels)")
    save_figure(fig, save_path)


plot_warpd_grid = plot_warped_grid  # Preserve the earlier function spelling.


def plot_training_history(training_losses, save_path, validation_losses=None):
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    epochs = np.arange(1, len(training_losses) + 1)
    ax.plot(epochs, training_losses, label="Training")
    if validation_losses is not None:
        if len(training_losses) != len(validation_losses):
            raise ValueError("Expected one validation loss per epoch.")
        ax.plot(epochs, validation_losses, label="Validation")
    ax.set(title="Registration objective by epoch", xlabel="Epoch", ylabel="Similarity + weighted smoothness")
    ax.legend()
    save_figure(fig, save_path)


def plot_segmentations(moving, fixed, warped, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(11, 4), layout="constrained")
    limit = max(float(np.max(a)) for a in [moving, fixed, warped])
    for ax, array, title in zip(axes, [moving, fixed, warped], ["Moving labels", "Fixed labels", "Warped labels"]):
        ax.imshow(array.T, origin="lower", cmap="tab20", vmin=0, vmax=max(limit, 1))
        ax.set_title(title)
        ax.axis("off")
    save_figure(fig, save_path)
