#!/bin/bash
set -e
OUTFILE="seed_results.txt"
echo "=== Started at $(date) ===" > $OUTFILE
for s in $(seq 0 30); do
    echo "" | tee -a $OUTFILE
    echo "=== Seed: $s ===" | tee -a $OUTFILE
    PYTHONPATH=. python train/train_surrogates.py --system linear --seed $s 2>&1 | tail -1 | tee -a $OUTFILE
    PYTHONPATH=. python experiments/run_main_benchmarks.py --system linear --checkpoint models/deeponet_linear.pt --seed $s 2>&1 | tail -1 | tee -a $OUTFILE
    PYTHONPATH=. python experiments/run_offgrid.py --checkpoint models/deeponet_linear.pt --seed $s 2>&1 | tail -3 | tee -a $OUTFILE
done
echo "" | tee -a $OUTFILE
echo "=== Finished at $(date) ===" | tee -a $OUTFILE
