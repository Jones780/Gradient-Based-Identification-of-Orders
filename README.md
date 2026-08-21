# Noise-Robust Gradient-Based Identification of Fractional Orders

Code associated with the manuscript **“Noise-Robust Gradient-Based Identification of Fractional Orders via Operator Learning and Integral Reconstruction.”**

## Overview

This repository contains reproducibility code for a continuous fractional-order identification framework based on:

- a DeepONet-style surrogate for noise-robust fractional-derivative evaluation;
- a differentiable fractional-integral reconstruction objective;
- projected gradient refinement of the unknown fractional order.

The manuscript assumes that the structural form of the governing right-hand side is specified and focuses on estimating the unknown fractional order from noisy trajectory observations.

## Repository structure

```text
src/                    Core numerical routines, models, and order estimation
train/                  Surrogate-training script
experiments/            Scripts for the principal numerical experiments
figures/                Generated figures (not tracked by default)
results/                Generated result tables/files (not tracked by default)
models/                 Local trained checkpoints (not tracked by default)
```

## Installation

Create a Python environment and install the dependencies:

```bash
pip install -r requirements.txt
```

## Training the surrogate

Train a surrogate for one of the three systems with:

```bash
python train/train_surrogates.py --system linear
```

Available systems are `linear`, `polynomial`, and `brusselator`. The default protocol uses fractional orders `{0.25, 0.50, 0.75, 0.95}`, 100 trajectories per order, 5% relative Gaussian noise, and 150 training epochs.

## Reproducing the main experiments

After training the corresponding checkpoint, the principal experiments can be run with:

```bash
python experiments/run_main_benchmarks.py --system linear --checkpoint models/deeponet_linear.pt
python experiments/run_offgrid.py --checkpoint models/deeponet_linear.pt
python experiments/run_ablation.py --checkpoint models/deeponet_linear.pt
python experiments/loss_landscapes.py --checkpoint models/deeponet_linear.pt
```

The scripts use deterministic seeds where applicable, but numerical output can still vary slightly with the PyTorch version, hardware, and numerical libraries.

## Reproducibility and checkpoints

Model checkpoints (`*.pt`), generated figures, and result files are excluded from the initial repository by `.gitignore`. This keeps the source repository lightweight. Checkpoints can be added later through a release or an archival service such as Zenodo if a citable binary record is desired.

## Citation

If you use this code, please cite the associated manuscript. A `CITATION.cff` file is included for GitHub citation support.

## License

This repository is distributed under the MIT License. See `LICENSE`.
