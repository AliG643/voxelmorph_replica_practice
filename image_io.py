"""2D PNG/NIfTI loading. All deformation units are pixels, not millimeters."""
from pathlib import Path
import numpy as np
from PIL import Image
import torch


def load_array(path, labels=False):
    path = Path(path)
    if path.name.endswith((".nii", ".nii.gz")):
        import nibabel as nib
        array = nib.load(path).get_fdata().squeeze()
    elif path.suffix == ".npy":
        array = np.load(path, allow_pickle=False)
    else:
        with Image.open(path) as image:
            array = np.array(image if labels else image.convert("L"))
        if not labels:
            array = array.astype(np.float32) / 255
    array = np.asarray(array, dtype=np.float32)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError(f"{path}: expected a finite 2D array, not a full 3D volume.")
    if labels:
        if not np.equal(array, np.rint(array)).all():
            raise ValueError("Segmentation IDs must be integers.")
    elif array.min() < 0 or array.max() > 1:
        low, high = array.min(), array.max()
        array = (array - low) / max(float(high - low), 1e-8)
    return array


def load_image(path):
    return torch.from_numpy(load_array(path)).unsqueeze(0)


def save_registration(output_dir, moving, fixed, warped, flow, fixed_path=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "registration.npz", moving=moving, fixed=fixed,
                        warped=warped, flow_xy=flow)
    Image.fromarray(np.rint(np.clip(warped, 0, 1) * 255).astype(np.uint8)).save(output_dir / "warped.png")
    if fixed_path and str(fixed_path).endswith((".nii", ".nii.gz")):
        import nibabel as nib
        reference = nib.load(fixed_path)
        nib.save(nib.Nifti1Image(warped.astype(np.float32), reference.affine), output_dir / "warped.nii.gz")
