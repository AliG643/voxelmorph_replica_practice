"""Training, held-out validation, checkpoint selection and resuming."""
import json
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from checkpoint import choose_device
from dataset import RegistrationDataset
from model import Voxelmorph
from spatial_transformer import SpatialTransformer
from losses import mse_loss, smoothness_loss, NCCLoss
from visualize import plot_training_history


def epoch_loss(model, transformer, loader, device, smoothness_weight, optimizer=None, similarity="mse"):
    training = optimizer is not None
    model.train(training)
    image_loss_fn = mse_loss if similarity == "mse" else NCCLoss()
    totals = {"loss": 0.0, "similarity": 0.0, "smoothness": 0.0}
    count = 0
    with torch.set_grad_enabled(training):
        for moving, fixed in loader:
            moving, fixed = moving.to(device), fixed.to(device)
            flow = model(moving, fixed)
            warped = transformer(moving, flow)
            match = image_loss_fn(fixed, warped)
            smooth = smoothness_loss(flow)
            loss = match + smoothness_weight * smooth
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss; check images and learning rate.")
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            batch = moving.shape[0]
            for key, value in [("loss", loss), ("similarity", match), ("smoothness", smooth)]:
                totals[key] += value.item() * batch
            count += batch
    return {key: value / count for key, value in totals.items()}


def train(data_dir, output_dir, epochs=10, learning_rate=0.001, batch_size=1,
          smoothness_weight=0.005, seed=42, integration_steps=0, similarity="mse",
          device="auto", resume=None):
    if epochs < 1 or batch_size < 1 or learning_rate <= 0 or smoothness_weight < 0:
        raise ValueError("Invalid training parameters.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    device = choose_device(device)
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "last.pt").exists() and resume is None:
        raise ValueError("Output already has weights. Use a new output folder or --resume.")
    loaders = {group: DataLoader(RegistrationDataset(data_dir/group/"moving", data_dir/group/"fixed"),
                 batch_size=batch_size, shuffle=(group == "train")) for group in ["train", "validation"]}
    model = Voxelmorph(integration_steps).to(device)
    transformer = SpatialTransformer().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    settings = dict(learning_rate=learning_rate, batch_size=batch_size, smoothness_weight=smoothness_weight,
                    seed=seed, integration_steps=integration_steps, similarity=similarity, flow_order="x,y")
    history, best, start = [], float("inf"), 1
    if resume:
        saved = torch.load(resume, map_location=device, weights_only=True)
        if saved["settings"] != settings:
            raise ValueError("Resume requires the checkpoint's same training settings.")
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        history, best, start = saved["history"], saved["best_validation_loss"], saved["epoch"] + 1
        torch.set_rng_state(saved["torch_rng_state"].cpu())
        if device.type == "cuda" and "cuda_rng_state" in saved:
            torch.cuda.set_rng_state_all([state.cpu() for state in saved["cuda_rng_state"]])
    if start > epochs:
        raise ValueError("--epochs must exceed the last completed epoch when resuming.")
    for epoch in range(start, epochs + 1):
        training = epoch_loss(model, transformer, loaders["train"], device, smoothness_weight, optimizer, similarity)
        validation = epoch_loss(model, transformer, loaders["validation"], device, smoothness_weight, similarity=similarity)
        history.append(dict(epoch=epoch, training=training, validation=validation))
        improved = validation["loss"] < best
        if improved:
            best = validation["loss"]
        saved = dict(model_state=model.state_dict(), optimizer_state=optimizer.state_dict(), epoch=epoch,
                     best_validation_loss=best, settings=settings, history=history, torch_rng_state=torch.get_rng_state())
        if device.type == "cuda":
            saved["cuda_rng_state"] = torch.cuda.get_rng_state_all()
        torch.save(saved, output_dir/"last.pt")
        if improved:
            torch.save(saved, output_dir/"best.pt")
        (output_dir/"history.json").write_text(json.dumps(history, indent=2))
        print(f'Epoch {epoch}/{epochs}: training={training["loss"]:.6f}, validation={validation["loss"]:.6f}', flush=True)
    plot_training_history([r["training"]["loss"] for r in history], output_dir/"training.png",
                          [r["validation"]["loss"] for r in history])
    return output_dir/"best.pt"
