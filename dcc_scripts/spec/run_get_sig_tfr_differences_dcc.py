#!/usr/bin/env python
"""
Submit script for sig TFR differences analysis.
This sets up input args for and calls get_sig_tfr_differences.py
Should be wrapped in an sbatch script for cluster submission.
"""
import sys
import os
import numpy as np
from functools import partial
from scipy.stats import ttest_ind
from types import SimpleNamespace
import argparse

# ============================================================================
# PATH SETUP
# ============================================================================
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
from dcc_scripts.spec.get_sig_tfr_differences_dcc import main
from src.analysis.config import experiment_conditions

# ============================================================================
# ANALYSIS PARAMETERS
# ============================================================================
LAB_ROOT = None

# Subject configuration
SUBJECTS = ['D0059']  # So need to manually update this, then run sbatch sbatch_get_sig_tfr_differences.sh. Not ideal, it would be better to have a submit script that loops over the subjects, but whatever!

# Time and task parameters
SIGNAL_TIMES = [-1.0, 1.5]
TASK = 'GlobalLocal'

# Trial selection
ACC_TRIALS_ONLY = True
ERROR_TRIALS_ONLY = False

# Frequency parameters
FREQS = np.arange(2, 200., 2.)  # Full spectrum
N_CYCLES = FREQS / 2

# For specific frequency bands, uncomment one:
# FREQS = np.arange(4, 8, 1)     # Theta
# FREQS = np.arange(8, 13, 1)    # Alpha
# FREQS = np.arange(13, 30, 2)   # Beta
# FREQS = np.arange(30, 70, 2)   # Low gamma
# FREQS = np.arange(70, 150, 2)  # High gamma

# Spectral analysis parameters
SPEC_METHOD = 'multitaper'
TIME_BANDWIDTH = 10
RETURN_ITC = False
AVERAGE = False

# Statistical parameters
STAT_FUNC = partial(ttest_ind, equal_var=False, nan_policy='omit')
P_THRESH = 0.05
N_PERM = 200  # Increase to 1000 for publication-quality results
IGNORE_ADJACENCY = 1  # Ignore channels dimension for clusters
TAILS = 2 # one tailed or two tailed t-test, currently doing a single two-tailed but i think should do two separate one-tailed - do that later..
SEED = None  # Set to integer for reproducibility

# Parallel processing
N_JOBS = 1

# Decoding parameters (for future functionality)
N_SPLITS = 2
N_REPEATS = 1
RANDOM_STATE = 42

# Condition selection
CONDITIONS = experiment_conditions.stimulus_big_letter_conditions
# Alternative conditions (uncomment to use):
# CONDITIONS = experiment_conditions.stimulus_small_letter_conditions
# CONDITIONS = experiment_conditions.stimulus_task_conditions
# CONDITIONS = experiment_conditions.stimulus_congruency_conditions
# CONDITIONS = experiment_conditions.response_big_letter_conditions

# Stimulus/Response locking
STIMULUS_LOCKED = True
RESPONSE_LOCKED = not STIMULUS_LOCKED

# Epochs file selection
if STIMULUS_LOCKED:
    EPOCHS_ROOT_FILE = "Stimulus_0.5sec_within1sec_randoffset_preStimulusBase_decFactor_8_outliers_10_passband_70.0-150.0_padLength_0.5s_stat_func_ttest_ind_equal_var_False"
    # Alternative epoch files:
    # EPOCHS_ROOT_FILE = "Stimulus_0.5sec_within1sec_randoffset_preStimulusBase_decFactor_8_outliers_10_passband_4.0-8.0_padLength_0.5s_stat_func_ttest_ind_equal_var_False"
    # EPOCHS_ROOT_FILE = "Stimulus_0.5sec_within1sec_randoffset_preStimulusBase_decFactor_8_outliers_10_passband_0.0-30.0_padLength_0.5s_stat_func_ttest_ind_equal_var_False"
else:
    EPOCHS_ROOT_FILE = "Response_0.5sec_within1sec_randoffset_preStimulusBase_decFactor_8_outliers_10_passband_70.0-150.0_padLength_0.5s_stat_func_ttest_ind"
    # EPOCHS_ROOT_FILE = "Response_0.5sec_within1sec_randoffset_preStimulusBase_decFactor_8_outliers_10_passband_4.0-8.0_padLength_0.5s_stat_func_ttest_ind"

# ROI dictionary
ROIS_DICT = {
    'lpfc': ["G_front_inf-Opercular", "G_front_inf-Orbital", "G_front_inf-Triangul", 
             "G_front_middle", "G_front_sup", "Lat_Fis-ant-Horizont", 
             "Lat_Fis-ant-Vertical", "S_circular_insula_ant", "S_circular_insula_sup", 
             "S_front_inf", "S_front_middle", "S_front_sup"],
    'occ': ["G_cuneus", "G_and_S_occipital_inf", "G_occipital_middle", 
            "G_occipital_sup", "G_oc-temp_lat-fusifor", "G_oc-temp_med-Lingual", 
            "Pole_occipital", "S_calcarine", "S_oc_middle_and_Lunatus", 
            "S_oc_sup_and_transversal", "S_occipital_ant"]
}

# ============================================================================
# TEST MODE CONFIGURATION (uncomment to enable)
# ============================================================================
# TEST_MODE = True
# if TEST_MODE:
#     print("WARNING: Running in TEST MODE with minimal parameters!")
#     SIGNAL_TIMES = [-0.2, 0.2]  # Shorter time window
#     FREQS = np.array([70, 80, 90])  # Just 3 frequencies
#     N_CYCLES = np.array([35, 40, 45])
#     N_PERM = 2  # Minimal permutations
#     P_THRESH = 0.5  # Higher threshold
#     ROIS_DICT = {'test_roi': ['G_front_inf-Opercular']}  # Single ROI

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_analysis():
    """Execute the sig TFR differences analysis with configured parameters."""
    
    # Create argument namespace
    args = SimpleNamespace(
        LAB_root=LAB_ROOT,
        subjects=SUBJECTS,
        signal_times=SIGNAL_TIMES,
        acc_trials_only=ACC_TRIALS_ONLY,
        error_trials_only=ERROR_TRIALS_ONLY,
        stat_func=STAT_FUNC,
        p_thresh=P_THRESH,
        ignore_adjacency=IGNORE_ADJACENCY,
        n_perm=N_PERM,
        n_jobs=N_JOBS,
        freqs=FREQS,
        n_cycles=N_CYCLES,
        return_itc=RETURN_ITC,
        time_bandwidth=TIME_BANDWIDTH,
        spec_method=SPEC_METHOD,
        average=AVERAGE,
        seed=SEED,
        tails=TAILS,
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
        task=TASK,
        conditions=CONDITIONS,
        epochs_root_file=EPOCHS_ROOT_FILE,
        rois_dict=ROIS_DICT
    )
    
    # Print configuration summary
    print("=" * 70)
    print("SIG TFR DIFFERENCES ANALYSIS")
    print("=" * 70)
    print(f"Subjects:          {SUBJECTS}")
    print(f"Conditions:        {list(CONDITIONS.keys())}")
    print(f"ROIs:              {list(ROIS_DICT.keys())}")
    print(f"Time window:       {SIGNAL_TIMES[0]} to {SIGNAL_TIMES[1]} s")
    print(f"Frequency range:   {FREQS[0]} to {FREQS[-1]} Hz ({len(FREQS)} frequencies)")
    print(f"Spectral method:   {SPEC_METHOD}")
    print(f"Permutations:      {N_PERM}")
    print(f"P-threshold:       {P_THRESH}")
    print(f"Stimulus locked:   {STIMULUS_LOCKED}")
    print(f"Epochs file:       {os.path.basename(EPOCHS_ROOT_FILE)}")
    print("=" * 70)
    
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