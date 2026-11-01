# Noise-Robust Gradient-Based Identification of Fractional Orders

Code associated with the manuscript **“Noise-Robust Gradient-Based Identification of Fractional Orders via Operator Learning and Integral Reconstruction.”**

## Overview

This repository contains the numerical routines and experiment scripts for a continuous fractional-order identification framework based on:
- a DeepONet-style surrogate for fractional-derivative estimation from noisy state observations;
- a differentiable fractional-integral reconstruction objective;
- projected-gradient refinement of the unknown order;
- two differentiable quadrature choices: the original left-rectangle rule and piecewise-constant product integration.

The benchmark assumes that the structural form of the governing right-hand side is specified. The proposed order estimator uses the observed noisy trajectory increment during test-stage optimization; it does not receive the clean test trajectory as its reconstruction target and does not evaluate the governing right-hand side in the surrogate-based optimization loop.

## Final three-system benchmark (1 November 2026)

The final observed-target benchmark evaluates three systems at true order 0.85 and 5% observation noise. Each system uses five independently trained checkpoints (seeds 42, 123, 456, 789, 2026) and 100 common test paths. Both quadrature rules are compared on the same paths.

| System | Rectangular MAE | Product-integration MAE | Relative reduction | Paired Wilcoxon p |
|---|---:|---:|---:|---:|
| Linear | 0.005787 | 0.005581 | 3.56% | 6.73e-17 |
| Nonlinear polynomial | 0.026264 | 0.025665 | 2.28% | 2.81e-12 |
| Fractional Brusselator | 0.019476 | 0.019146 | 1.70% | 0.1330 |

MAE is computed by first averaging each path's error across the five checkpoints and then averaging across 100 paths. Path SD and the SD across five seed-specific MAEs are reported separately in `results/final_three_system_summary.csv`. Paired tests use 100 path-level pairs per system; the five checkpoints are not treated as 500 independent training replicates. Product integration's reduction is statistically supported for the linear and polynomial systems, but not for the Brusselator.

The linear-system product-integration sensitivity experiments use 20 common paths per condition. MAE increases from 0.003659 at 0% noise to 0.007924 at 10% noise. Initial guesses 0.4, 0.6, and 0.8 give nearly identical errors (about 0.005525).

## Repository structure

```text
src/                    Core numerical routines, models, solver, and order estimation
train/                  Surrogate-training script
experiments/            Main, ablation, and off-grid scripts
results/                Final aggregate tables and provenance notes
tests/                  Unit tests for the fractional-integral implementations
models/                 Local trained checkpoints (not tracked)
figures/                Generated figures (not tracked by default)
```

## Installation

```bash
pip install -r requirements.txt
```

## Train a surrogate

Train a separate surrogate for one benchmark system:

```bash
python train/train_surrogates.py --system linear --seed 42
```

Available systems are `linear`, `polynomial`, and `brusselator`. The default protocol uses fractional orders `{0.25, 0.50, 0.75, 0.95}`, 100 trajectories per order (400 total), 5% relative Gaussian noise, and 150 epochs.

## Run a main benchmark repeat

After training the corresponding checkpoint:

```bash
python experiments/run_main_benchmarks.py --system linear --checkpoint models/deeponet_linear.pt --quadrature-rule product
```

The default `--quadrature-rule` is `rectangular` for backward compatibility. Use `product` to select piecewise-constant product integration. The final five-seed/100-path experiment is documented in the results records; the binary checkpoints and raw Kaggle archives are preserved in the companion manuscript revision package rather than duplicated in Git history.

## Quadrature API

`src.fractional.fractional_integral_torch(F, alpha, h, Nt, rule="rectangular")` accepts `rule="rectangular"` or `rule="product"`. The latter implements the piecewise-constant product-integration weights and remains differentiable with respect to the scalar tensor `alpha`.

The estimation routines accept the optional `quadrature_rule` argument. Existing calls remain backward-compatible and default to the rectangular rule.

## Tests

Run the integral checks with:

```bash
python -m unittest discover -s tests -v
```

The tests check constant-input exactness of product integration, finite gradients with respect to the order, and backward-compatible rectangular-rule behavior.

## Reproducibility artifacts

- `results/final_three_system_summary.csv`: MAE, path SD, and across-seed SD.
- `results/final_three_system_paired_tests.csv`: paired reductions, raw and Holm-adjusted p-values, and bootstrap intervals.
- `results/linear_sensitivity_summary.csv`: noise and initial-guess sensitivity.
- `results/final_benchmark_manifest.json`: protocol, data-unit, and source/checkpoint provenance notes.
- `results/README.md`: interpretation and archive-location caveats.

The complete raw trial data, run manifests, checkpoints, and training logs are retained in the companion revision package. Binary checkpoints are intentionally not committed to the Git repository.

## Citation

Please cite the associated manuscript. A `CITATION.cff` file is included for GitHub citation support.

## License

This repository is distributed under the MIT License. See `LICENSE`.
