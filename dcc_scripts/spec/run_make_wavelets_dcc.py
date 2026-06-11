
#!/usr/bin/env python
"""
Submit script for spectrogram (wavelet or multitaper) analysis.
This sets up input args for and calls make_wavelets_dcc.py
Should be wrapped in an sbatch script for cluster submission.
"""
import sys
import os
import numpy as np
from types import SimpleNamespace
from datetime import datetime

# ============================================================================
# PATH SETUP
# ============================================================================
# Detect if we're on cluster or local machine
if os.path.exists("/hpc/home"):
    # We're on the cluster
    USER = os.environ.get('USER')
    sys.path.append(f"/hpc/home/{USER}/coganlab/{USER}/GlobalLocal/IEEG_Pipelines/")
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
else:
    # Local machine (Windows)
    sys.path.append("C:/Users/jz421/Desktop/GlobalLocal/IEEG_Pipelines/")
    try:
        current_file_path = os.path.abspath(__file__)
        current_script_dir = os.path.dirname(current_file_path)
    except NameError:
        current_script_dir = os.getcwd()

project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..'))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import after path is set up
from dcc_scripts.spec.make_wavelets_dcc import main

# ============================================================================
# ANALYSIS PARAMETERS
# ============================================================================
LAB_ROOT = None # Will be determined automatically in main()

SUBJECT_ID = os.environ.get('SUBJECT_ID')
if SUBJECT_ID is None:
    raise ValueError("SUBJECT_ID environment variable not set. "
                     "Set it via sbatch --export=ALL,SUBJECT_ID=...")
# task
TASK = 'GlobalLocal'

# ============================================================================
# Condition selection - NOW FROM SUBMIT SCRIPT
# Pass a condition_label (string key into condition_registry.CONDITION_REGISTRY).
# The registry resolves: conditions_obj, comparisons, subtraction_pairs,
# anova_factors, anova_interactions, etc.
# Example labels: 'stimulus_err_corr_conditions', 'stimulus_lwpc_conditions',
# 'stimulus_experiment_conditions' (16-cell ANOVA).
CONDITION_LABEL = os.environ.get('CONDITION_LABEL')
if CONDITION_LABEL is None:
    raise ValueError("CONDITION_LABEL environment variable not set. "
                     "Set it via sbatch --export=ALL,CONDITION_LABEL=...")

SPEC_TYPE = 'wavelet' # wavelet or multitaper

BASELINE_TIMES = [-0.5, 0]
SIGNAL_TIMES = [-0.5, 1.5]
    
# multitaper params
FREQS = np.arange(10, 200, 2)
N_CYCLES = FREQS / 2  
TIME_BANDWIDTH = 10  
RETURN_ITC = False

def run_analysis():
    """Make wavelets or multitaper."""
    
    # Create argument namespace
    args = SimpleNamespace(
        LAB_root=LAB_ROOT,
        task=TASK,
        subject_id=SUBJECT_ID,
        condition_label=CONDITION_LABEL,
        spec_type=SPEC_TYPE,
        baseline_times=BASELINE_TIMES,
        signal_times=SIGNAL_TIMES,
        freqs=FREQS,
        n_cycles=N_CYCLES,
        time_bandwidth=TIME_BANDWIDTH,
        return_itc=RETURN_ITC
    )

    # Print configuration summary
    print("=" * 70)
    print("MAKE WAVELETS ANALYSIS")
    print("=" * 70)
    print(f"Subject id:          {SUBJECT_ID}")
    print(f"Condition label:        {CONDITION_LABEL}")
    print(f"spec type:        {SPEC_TYPE}")
    print(f"baseline times:        {BASELINE_TIMES}")
    print(f"signal times:        {SIGNAL_TIMES}")
    print(f"multitaper freqs:        {FREQS}")
    print(f"multitaper n_cycles:        {N_CYCLES}")
    print(f"multitaper time bandwidth:        {TIME_BANDWIDTH}")
    print(f"multitaper return_itc:        {RETURN_ITC}")

    
    # Run the analysis
    print("\nStarting analysis...")
    try:
        main(args)
        print("\n✓ Analysis completed successfully!")
    except Exception as e:
        print(f"\n✗ Analysis failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_analysis()


