
#!/usr/bin/env python
"""
Submit script for james sun cluster decoding analysis.
This sets up input args for and calls james_sun_cluster_decoding_dcc.py
Should be wrapped in an sbatch script for cluster submission.
"""
import sys
import os
import numpy as np
from functools import partial
from scipy.stats import ttest_ind
from types import SimpleNamespace
import pickle
import json

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
from dcc_scripts.decoding.james_sun_cluster_decoding_dcc import main
from src.analysis.config import experiment_conditions

# ============================================================================
# ANALYSIS PARAMETERS
# ============================================================================
LAB_ROOT = None  # Will be determined automatically in main()

# Subject configuration
SUBJECTS = ['D0103']  # Update this for each run

# Time and task parameters
SIGNAL_TIMES = [-1.0, 1.5]
TASK = 'GlobalLocal'

# Trial selection
ACC_TRIALS_ONLY = True
ERROR_TRIALS_ONLY = False

# Frequency parameters
FREQS = np.arange(2, 200., 2.)  # Full spectrum
N_CYCLES = FREQS / 2

# Spectral analysis parameters
SPEC_METHOD = 'multitaper'
TIME_BANDWIDTH = 10
RETURN_ITC = False
AVERAGE = False

# Statistical parameters
# Note: We'll serialize the stat_func for passing to main
STAT_FUNC = partial(ttest_ind, equal_var=False, nan_policy='omit')
STAT_FUNC_NAME = 'ttest_ind'  # For reference
STAT_FUNC_PARAMS = {'equal_var': False, 'nan_policy': 'omit'}
P_THRESH = 0.05
N_PERM = 200
IGNORE_ADJACENCY = 1
TAILS = 2
SEED = None

# Parallel processing
N_JOBS = 1

# Decoding parameters
N_SPLITS = 5
N_REPEATS = 5
RANDOM_STATE = 42
EXPLAINED_VARIANCE = 0.8
BALANCE_METHOD = 'subsample'
NORMALIZE = 'all'
OBS_AXS = 0
CHANS_AXS = 1
FREQ_AXS = 2
TIME_AXS = 3
OVERSAMPLE = True
ALPHA = 1.0
CLEAR_MEMORY = True

# Condition selection
CONDITIONS = experiment_conditions.stimulus_big_letter_conditions
CONDITIONS_NAME = 'stimulus_big_letter_conditions'  # Store the name for reference

# Stimulus/Response locking
STIMULUS_LOCKED = True
RESPONSE_LOCKED = not STIMULUS_LOCKED

# Epochs file selection
if STIMULUS_LOCKED:
    EPOCHS_ROOT_FILE = "Stimulus_0.5sec_within1sec_randoffset_preStimulusBase_decFactor_8_outliers_10_passband_70.0-150.0_padLength_0.5s_stat_func_ttest_ind_equal_var_False"
else:
    EPOCHS_ROOT_FILE = "Response_0.5sec_within1sec_randoffset_preStimulusBase_decFactor_8_outliers_10_passband_70.0-150.0_padLength_0.5s_stat_func_ttest_ind"

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
        rois_dict=ROIS_DICT,
        explained_variance=EXPLAINED_VARIANCE,
        balance_method=BALANCE_METHOD,
        normalize=NORMALIZE,
        obs_axs=OBS_AXS,
        chans_axs=CHANS_AXS,
        freq_axs=FREQ_AXS,
        time_axs=TIME_AXS,
        oversample=OVERSAMPLE,
        alpha=ALPHA,
        clear_memory=CLEAR_MEMORY
    )
    
    # Print configuration summary
    print("=" * 70)
    print("JAMES SUN CLUSTER DECODING ANALYSIS")
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
    print(f"Explained variance: {EXPLAINED_VARIANCE}")
    print(f"Balance method:     {BALANCE_METHOD}")
    print(f"Normalize:          {NORMALIZE}")
    print(f"Obs axs:            {OBS_AXS}")
    print(f"Chans axs:          {CHANS_AXS}")
    print(f"Freq axs:           {FREQ_AXS}")
    print(f"Time axs:           {TIME_AXS}")
    print(f"Oversample:         {OVERSAMPLE}")
    print(f"Alpha:              {ALPHA}")
    print(f"Clear memory:       {CLEAR_MEMORY}")
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


