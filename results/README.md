# Final benchmark results

These files contain the aggregate numerical results used in the manuscript revision dated 1 November 2026.

## Statistical unit

The main benchmark contains five trained checkpoints and 100 common test paths for each of three systems. Each test path is evaluated by every checkpoint and by both quadrature rules. For the reported path-level summary, absolute errors are averaged across the five checkpoints within each fixed path, then summarized across 100 paths. The path SD is the SD across those path averages. The seed SD is computed separately across the five seed-specific MAEs. Do not interpret the 5 × 100 combinations as 500 independent training replicates.

Paired quadrature tests are performed on 100 path-level pairs per system after checkpoint averaging. Reported p-values are two-sided Wilcoxon signed-rank values, with Holm adjustment across the three systems. Bootstrap intervals are percentile intervals for the mean paired MAE reduction (rectangular minus product integration).

## Main interpretation

Product integration reduces MAE in all three systems, but gains are modest. The paired comparison supports reductions in the linear and nonlinear-polynomial systems; the fractional Brusselator reduction is not statistically significant. The linear noise- and initialization-sensitivity experiments each use 20 paths per condition and should not be generalized to the nonlinear systems.

## Provenance

The complete raw CSVs, manifests, checkpoint weights, and training logs are preserved in the companion manuscript revision package. The primary archive is named `final_three_system_benchmark.zip`; it is accompanied by `training_seed_sensitivity_outputs.zip` and `test_set_sensitivity_outputs.zip`. The full package records SHA-256 identifiers for the source snapshots and checkpoints. Binary checkpoint files are intentionally not duplicated in this source repository.

The historical Table-1 checkpoints and precise test paths were not recovered. A historical notebook audit found clean trajectory increments used as optimization targets in several legacy estimator cells. The exact provenance of every old table value remains unauthenticated, and the old values are superseded by the observed-target benchmark.
