"""Explicit filename pairing; subject splitting is done during preparation."""
from pathlib import Path
from torch.utils.data import Dataset
from image_io import load_image


class RegistrationDataset(Dataset):
    def __init__(self, moving_dir, fixed_dir):
        def files(folder):
            return {p.name: p for p in Path(folder).iterdir()
                    if p.is_file() and p.name.lower().endswith((".png", ".nii", ".nii.gz", ".npy"))}
        moving, fixed = files(moving_dir), files(fixed_dir)
        if not moving or moving.keys() != fixed.keys():
            raise ValueError("Moving and fixed folders must contain matching nonempty filenames.")
        self.pairs = [(moving[name], fixed[name]) for name in sorted(moving)]
        if any(a.resolve() == b.resolve() for a, b in self.pairs):
            raise ValueError("Moving and fixed paths must differ.")

    def __len__(self):
        return len(self.pairs)

    load_image = staticmethod(load_image)

    def __getitem__(self, index):
        moving, fixed = (load_image(path) for path in self.pairs[index])
        if moving.shape != fixed.shape:
            raise ValueError("Paired images must share shape and already be approximately aligned.")
        return moving, fixed
