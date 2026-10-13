"""Test-set evaluation with optional existing anatomical labels."""
import json
from pathlib import Path
import numpy as np
import torch
from checkpoint import choose_device, load_model
from dataset import RegistrationDataset
from image_io import load_array, save_registration
from metrics import image_mse, fold_fraction, mean_dice, dice_per_label
from register import predict_pair, save_pair_figures
from spatial_transformer import SpatialTransformer
from visualize import plot_segmentations


def evaluate(data_dir, checkpoint_path, output_dir, device="auto", save_examples=3):
    device = choose_device(device)
    model, settings = load_model(checkpoint_path, device)
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = RegistrationDataset(data_dir/"test"/"moving", data_dir/"test"/"fixed")
    nearest = SpatialTransformer("nearest").to(device)
    rows = []
    for index in range(len(dataset)):
        moving, fixed = dataset[index]
        warped, flow = predict_pair(model, moving, fixed, device)
        m, f = moving[0].numpy(), fixed[0].numpy()
        name = dataset.pairs[index][0].name
        row = dict(pair=name, mse_before=image_mse(f, m), mse_after=image_mse(f, warped), fold_fraction=fold_fraction(flow))
        example = output_dir/f"pair_{index:03}"
        if index < save_examples:
            save_registration(example, m, f, warped, flow, dataset.pairs[index][1])
            save_pair_figures(m, f, warped, flow, example)
        # OASIS preparation stores original integer labels as .npy sidecars.
        stem = name.removesuffix(".nii.gz").rsplit(".", 1)[0]
        mp, fp = data_dir/"test"/"moving_seg"/f"{stem}.npy", data_dir/"test"/"fixed_seg"/f"{stem}.npy"
        if mp.exists() != fp.exists():
            raise ValueError("Both moving and fixed segmentations are required for Dice.")
        if mp.exists():
            ms, fs = load_array(mp, labels=True), load_array(fp, labels=True)
            if ms.shape != m.shape or fs.shape != f.shape:
                raise ValueError("Segmentation shapes must match the corresponding images.")
            with torch.no_grad():
                ws = nearest(torch.from_numpy(ms)[None, None].to(device), torch.from_numpy(flow)[None].to(device))[0, 0].cpu().numpy()
            row.update(dice_before=mean_dice(fs, ms), dice_after=mean_dice(fs, ws), dice_per_label=dice_per_label(fs, ws))
            if index < save_examples:
                np.savez_compressed(example/"labels.npz", moving=ms, fixed=fs, warped=ws)
                plot_segmentations(ms, fs, ws, example/"labels.png")
        rows.append(row)
    keys = ["mse_before", "mse_after", "fold_fraction", "dice_before", "dice_after"]
    summary = {key:float(np.mean([row[key] for row in rows if key in row])) for key in keys if any(key in row for row in rows)}
    report = dict(checkpoint=str(Path(checkpoint_path).resolve()), settings=settings, test_pairs=len(rows), summary=summary, pairs=rows)
    # JSON null is used for undefined scores, rather than nonstandard NaN.
    def clean(value):
        if isinstance(value, dict): return {k:clean(v) for k,v in value.items()}
        if isinstance(value, list): return [clean(v) for v in value]
        if isinstance(value, float) and not np.isfinite(value): return None
        return value
    (output_dir/"metrics.json").write_text(json.dumps(clean(report), indent=2, allow_nan=False))
    print(json.dumps(clean(summary), indent=2), flush=True)
    return report
