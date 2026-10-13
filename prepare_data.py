"""Prepare public OASIS 2D MRI with reproducible subject-level splits."""
import gzip
import hashlib
import json
from pathlib import Path
import tarfile
import urllib.request
import nibabel as nib
import numpy as np
from PIL import Image

URL = "https://surfer.nmr.mgh.harvard.edu/ftp/data/neurite/data/neurite-oasis.2d.v1.0.tar"
MD5 = "c9ae5864f250c7e4b8d83a104e51ae8e"


def prepare(data_dir, archive=None, seed=42):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    if archive is None:
        archive = data_dir/"neurite-oasis.2d.v1.0.tar"
        if not archive.exists():
            print("Downloading public OASIS 2D slices...", flush=True)
            urllib.request.urlretrieve(URL, archive)
    archive = Path(archive)
    if hashlib.md5(archive.read_bytes()).hexdigest() != MD5:
        raise ValueError("Dataset checksum does not match the published archive.")
    images, labels = {}, {}
    with tarfile.open(archive) as tar:
        names = sorted({m.name.split("/")[0] for m in tar if m.name.endswith("/slice_norm.nii.gz")})
        for name in names:
            for filename, destination in [("slice_norm.nii.gz", images), ("slice_seg24.nii.gz", labels)]:
                raw = gzip.decompress(tar.extractfile(f"{name}/{filename}").read())
                destination[name] = nib.Nifti1Image.from_bytes(raw).get_fdata().squeeze()
    indices = np.random.default_rng(seed).permutation(len(names))
    split = {group:[names[i] for i in chosen] for group,chosen in
             [("train",indices[:-60]), ("validation",indices[-60:-40]), ("test",indices[-40:])]}
    expected = dict(source=URL, archive_md5=MD5, seed=seed, subjects=split)
    manifest_path = data_dir/"manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != expected:
        raise ValueError("This data folder has a different split; choose a new folder.")
    pairs = {}
    for group, subjects in split.items():
        for side in ["moving", "fixed", "moving_seg", "fixed_seg"]:
            (data_dir/group/side).mkdir(parents=True, exist_ok=True)
        entries = []
        step = 1 if group == "train" else 2
        for p, i in enumerate(range(0, len(subjects), step)):
            a, b = subjects[i], subjects[(i + 1) % len(subjects)]
            name = f"pair_{p:03}"
            entries.append(dict(filename=name+".png", moving=a, fixed=b))
            for side, subject in [("moving",a), ("fixed",b)]:
                array = np.rint(np.clip(images[subject],0,1)*255).astype(np.uint8)
                Image.fromarray(array).save(data_dir/group/side/f"{name}.png")
                np.save(data_dir/group/f"{side}_seg"/f"{name}.npy", labels[subject].astype(np.int16))
        pairs[group] = entries
    manifest_path.write_text(json.dumps(expected, indent=2))
    (data_dir/"pairs.json").write_text(json.dumps(pairs, indent=2))
    print(f"Prepared {len(split['train'])} training subjects, 20 validation subjects, 40 test subjects.")
    return data_dir
