#!/bin/bash

# Example: Compare time windows (-0.5 to 0.0 vs 0.0 to 0.5)
# Find pairs significantly increased after stimulus
# Then plot these pairs across different conditions (stimulus_c, motor)

cd /hpc/home/rl330/coganlab/rl330/GlobalLocal/src/analysis/pac

python sig_test.py \
  --compare_windows \
  --baseline_window "-0.5:0.0" \
  --test_window "0.0:0.5" \
  --subj D0057 D0059 D0063 \
  --region lpfc \
  --conditions stimulus_c motor \
  --perm_trials \
  --n_perm 1000 \
  --alpha 0.05 \
  -o timewindow_comparison_results.csv \
  -O timewindow_overall.csv \
  -f timewindow_comparison.png

echo "Done! Check outputs:"
echo "  - timewindow_comparison_results.csv (detailed per-pair results)"
echo "  - timewindow_comparison_sig_pairs_across_conds.png (visualization)"
