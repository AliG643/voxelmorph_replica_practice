"""Construct the correct model from saved settings."""
import torch
from model import Voxelmorph


def choose_device(name="auto"):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available in this environment.")
    return torch.device(name)


def load_model(path, device):
    saved = torch.load(path, map_location=device, weights_only=True)
    if "model_state" in saved:
        settings = saved.get("settings", {})
        weights = saved["model_state"]
    else:
        # Older runner checkpoints or plain state_dict use direct displacement.
        settings = saved.get("settings", {}) if "weights" in saved else {}
        weights = saved["weights"] if "weights" in saved else saved
    model = Voxelmorph(settings.get("integration_steps", 0)).to(device)
    model.load_state_dict(weights)
    model.eval()
    return model, settings
