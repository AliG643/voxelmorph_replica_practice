# 2D deformable image registration

A small, independent implementation of the core VoxelMorph registration idea: a U-Net predicts spatial deformation, a differentiable sampler warps the moving image, and image similarity plus smoothness trains the network. Optional stationary-velocity integration is supported. This is an educational 2D implementation, not a reproduction of the published 3D benchmark or the full original library.

## Setup

The existing environment already contains the required dependencies:

```powershell
conda activate voxelmorph
cd D:\vmorph\replica
python run.py --help
```

For a different environment, install `requirements.txt` and follow your hardware's PyTorch installation requirements. Python 3.10 or newer is recommended. No original VoxelMorph or Neurite package is required.

## One command for the full experiment

Use the OASIS archive already downloaded on this machine:

```powershell
python run.py pipeline --archive ../voxelmorph/demo/data/neurite-oasis.2d.v1.0.tar --epochs 10 --output outputs/baseline
```

This prepares data if absent, trains, selects the best checkpoint using validation, evaluates test pairs, and saves figures and metrics. If no archive is supplied and data is absent, it downloads the public 24 MB 2D archive. A matching existing data folder is reused. Weights in an existing training output are protected; choose a different output or resume.

Change parameters from one place:

```powershell
python run.py pipeline --epochs 10 --learning-rate 0.001 --batch-size 1 --smoothness-weight 0.005 --integration-steps 5 --output outputs/integrated
```

`--epochs` counts full passes over 354 training pairs. It is not the same as the earlier helper's `--steps`. `--integration-steps 0` is direct displacement; positive steps integrate velocity. `--similarity ncc` changes the image loss and needs its own regularization tuning. `--device cpu`, `cuda`, or `auto` selects hardware. Defaults come from `config.py`; command-line values override them.

## Separate commands

```powershell
python run.py prepare --archive ../voxelmorph/demo/data/neurite-oasis.2d.v1.0.tar
python run.py train --epochs 10 --output outputs/baseline
python run.py evaluate --checkpoint outputs/baseline/best.pt --output outputs/baseline/evaluation
python run.py register --moving data/oasis/test/moving/pair_000.png --fixed data/oasis/test/fixed/pair_000.png --checkpoint outputs/baseline/best.pt --output outputs/single_pair
```

Resume to a larger total epoch count, using the same training settings and output directory:

```powershell
python run.py train --epochs 20 --resume outputs/baseline/last.pt --output outputs/baseline
```

For an integrated experiment, also supply its original `--integration-steps`, smoothness, learning rate, batch size and similarity settings. Changing settings while resuming is rejected. Checkpoints retain optimizer state, history and PyTorch random state. GPU runs may remain nondeterministic across devices/software.

## Inputs and outputs

The prepared directory has `train`, `validation`, and `test`, each with `moving` and `fixed` folders. Corresponding files have matching names. Optional `moving_seg` and `fixed_seg` folders contain matching `.npy` segmentation sidecars. Image batches require consistent shapes; pair shapes must always match.

The loader accepts grayscale PNG, 2D NIfTI (including singleton third axes), and 2D NumPy arrays. Unsupported full 3D inputs are rejected. Images must already be in compatible spatial grids and approximately aligned. There is no automatic skull stripping, affine alignment, resampling, or multimodal contrast handling. Shape equality alone does not prove compatible physical space.

PNG intensities are converted to [0,1]. NIfTI/NumPy intensities already in [0,1] are preserved; otherwise per-image min-max normalization is applied. Constant-valued images outside that range become zero. For other modalities, design preprocessing explicitly rather than assuming this normalization is sufficient.

Training writes `best.pt`, `last.pt`, `history.json` and `training.png`. Checkpoints store architecture mode and training settings; evaluation recreates the correct integration mode automatically. Older direct-displacement plain state_dict checkpoints and helper `weights` checkpoints can also be loaded.

Evaluation writes `metrics.json` and example folders containing:
- `comparison.png`: moving/fixed/warped and error before/after, with shared error scales.
- `displacement.png`: field magnitude in pixels.
- `grid.png`: bent coordinate grid.
- `registration.npz`: full-precision input/output arrays and `flow_xy`.
- `warped.png`: quantized preview.
- `labels.png` and `labels.npz` when segmentations are available.

Registration accepts one new pair and produces the same intensity/field artifacts. If the fixed input is NIfTI, the warped NIfTI uses its affine. The displacement remains in array-index pixels, not millimeters; it is saved as NumPy data, not advertised as a world-coordinate vector field. Plotting transposes every panel consistently for the OASIS coronal display; this does not modify model inputs.

## Metrics and evaluation discipline

MSE is squared normalized-intensity error; lower is better. Dice measures anatomical label overlap; higher is better. Mean Dice averages over the union of non-background labels present in each pair and gives each label equal weight. The labels are provided automated segmentations, not outputs of this model or manual ground truth. Empty-label scores are undefined and written as JSON null.

Fold fraction is the fraction of finite-difference Jacobian determinants of identity plus displacement that are nonpositive. Multiply by 100 for a percentage. Integration encourages invertibility, but discrete sampling can still fold. Do not rank registration solely by overlap.

Data preparation splits by subject before making pairs: 354 training, 20 validation, 40 test subjects, with seed 42 by default. Training pairs link neighboring subjects in the shuffled training list; validation and test use 10 and 20 disjoint pairs. `manifest.json` records the split and archive source; `pairs.json` maps filenames to subject IDs. PNG conversion quantizes normalized intensities; original segmentation IDs are retained. A different split cannot overwrite an existing manifest.

Choose settings on validation data. Test data is used by evaluation after training; repeated exploratory test runs should not be used to tune parameters. No published accuracy is promised by these small experiments.

## Tests and learning guide

```powershell
python -m unittest discover -s tests -v
```

Tests check sampling direction, identity, integer labels, integration, gradients, dataset pairing, known metrics, training versus validation updates, arbitrary spatial sizes, and checkpoint reloads. See `DESIGN.md` for the reasoning and every file/function.

## Sources and data attribution

- [VoxelMorph framework paper](https://arxiv.org/abs/1809.05231): direct-displacement, differentiable warping and unsupervised similarity/smoothness objective.
- [Diffeomorphic extension](https://arxiv.org/abs/1805.04605): velocity-based registration. This project implements deterministic velocity integration, not the full probabilistic model.
- [Neurite OASIS data documentation](https://github.com/adalca/medical-datasets/blob/master/neurite-oasis.md): preprocessed public MRI slices. Follow its OASIS data use agreement and cite OASIS and HyperMorph when using these data. Retain third-party attribution.
