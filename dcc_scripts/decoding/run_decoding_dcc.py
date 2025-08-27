
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
from dcc_scripts.decoding.decoding_dcc import main
from src.analysis.config import experiment_conditions

# ============================================================================
# ANALYSIS PARAMETERS
# ============================================================================
LAB_ROOT = None  # Will be determined automatically in main()

# Subject configuration
SUBJECTS = ['D0057','D0059', 'D0063', 'D0065', 'D0069', 'D0071', 'D0077', 'D0090', 'D0094', 'D0100', 'D0102', 'D0103', 'D0107A', 'D0110', 'D0116', 'D0117', 'D0121']

# task
TASK = 'GlobalLocal'

# Trial selection
ACC_TRIALS_ONLY = True

# Statistical parameters
STAT_FUNC = partial(ttest_ind, equal_var=False, nan_policy='omit')
P_THRESH = 0.05
N_PERM = 1000

# Parallel processing
N_JOBS = -1

# Decoding parameters
N_SPLITS = 5
N_REPEATS = 1000
RANDOM_STATE = 42
EXPLAINED_VARIANCE = 0.8
BALANCE_METHOD = 'subsample'
NORMALIZE = 'true'
OBS_AXS = 0
CHANS_AXS = 1
TIME_AXS = -1

# Time-windowed decoding parameters
WINDOW_SIZE = 32  # Window size in samples (e.g., 64 samples = 250 ms at 256 Hz)
STEP_SIZE = 16    # Step size in samples (e.g., 16 samples = 62.5 ms at 256 Hz)
SAMPLING_RATE = 256 # Sampling rate of the data in Hz
FIRST_TIME_POINT = -1.0 # The time in seconds of the first sample in the epoch
TAILS = 1 # 1 for one-tailed (e.g., accuracy > chance), 2 for two-tailed

# # remove outlier timepoints or not
# MARK_OUTLIERS_AS_NAN = False

# Condition selection
CONDITIONS = experiment_conditions.stimulus__conditions

# Epochs file selection

EPOCHS_ROOT_FILE = "Stimulus_0.5sec_within1sec_randoffset_preStimulusBase_decFactor_8_outliers_10_passband_70.0-150.0_padLength_0.5s_stat_func_ttest_ind_equal_var_False"
# EPOCHS_ROOT_FILE = "Response_0.5sec_within1sec_randoffset_preStimulusBase_decFactor_8_outliers_10_passband_70.0-150.0_padLength_0.5s_stat_func_ttest_ind"

# ROI dictionary
ROIS_DICT = {
    'dlpfc': ["G_front_middle", "G_front_sup", "S_front_inf", "S_front_middle", "S_front_sup"],
    'acc': ["G_and_S_cingul-Ant", "G_and_S_cingul-Mid-Ant"],
    'parietal': ["G_parietal_sup", "S_intrapariet_and_P_trans", "G_pariet_inf-Angular", "G_pariet_inf-Supramar"],
    'lpfc': ["G_front_inf-Opercular", "G_front_inf-Orbital", "G_front_inf-Triangul", "G_front_middle", "G_front_sup", "Lat_Fis-ant-Horizont", "Lat_Fis-ant-Vertical", "S_circular_insula_ant", "S_circular_insula_sup", "S_front_inf", "S_front_middle", "S_front_sup"],
    'v1': ["G_oc-temp_med-Lingual", "S_calcarine", "G_cuneus"],
    'occ': ["G_cuneus", "G_and_S_occipital_inf", "G_occipital_middle", "G_occipital_sup", "G_oc-temp_lat-fusifor", "G_oc-temp_med-Lingual", "Pole_occipital", "S_calcarine", "S_oc_middle_and_Lunatus", "S_oc_sup_and_transversal", "S_occipital_ant"]
}



# # testing params (comment out)
# SUBJECTS = ['D0103']
# N_SPLITS = 2
# N_REPEATS = 2
# N_PERM = 10
# ROIS_DICT = {
#     'lpfc': ["G_front_inf-Opercular", "G_front_inf-Orbital", "G_front_inf-Triangul", "G_front_middle", "G_front_sup", "Lat_Fis-ant-Horizont", "Lat_Fis-ant-Vertical", "S_circular_insula_ant", "S_circular_insula_sup", "S_front_inf", "S_front_middle", "S_front_sup"]
# }

def run_analysis():
    """Execute the bandpass-filtered decoding analysis."""
    # Generate a timestamp string
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create argument namespace
    args = SimpleNamespace(
        timestamp=timestamp,
        LAB_root=LAB_ROOT,
        subjects=SUBJECTS,
        acc_trials_only=ACC_TRIALS_ONLY,
        stat_func=STAT_FUNC,
        p_thresh=P_THRESH,
        n_perm=N_PERM,
        n_jobs=N_JOBS,
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
        obs_axs=OBS_AXS,
        chans_axs=CHANS_AXS,
        time_axs=TIME_AXS,
        window_size=WINDOW_SIZE,
        step_size=STEP_SIZE,
        sampling_rate=SAMPLING_RATE,
        first_time_point=FIRST_TIME_POINT
    )
    
    # Print configuration summary
    print("=" * 70)
    print("BANDPASS FILTERED DECODING ANALYSIS")
    print("=" * 70)
    print(f"Subjects:          {SUBJECTS}")
    print(f"Conditions:        {list(CONDITIONS.keys())}")
    print(f"ROIs:              {list(ROIS_DICT.keys())}")
    print(f"Permutations:      {N_PERM}")
    print(f"P-threshold:       {P_THRESH}")
    print(f"Stimulus locked:   {STIMULUS_LOCKED}")
    print(f"Epochs file:       {os.path.basename(EPOCHS_ROOT_FILE)}")
    print(f"Explained variance: {EXPLAINED_VARIANCE}")
    print(f"Balance method:     {BALANCE_METHOD}")
    print(f"Obs axs:            {OBS_AXS}")
    print(f"Chans axs:          {CHANS_AXS}")
    print(f"Time axs:           {TIME_AXS}")
    print("-" * 70)
    print("Decoding Parameters:")
    print(f"  CV Splits/Repeats: {N_SPLITS}/{N_REPEATS}")
    print(f"  Shuffle Permutations: {N_PERM}")
    print(f"  Balance Method:    {BALANCE_METHOD}")
    print(f"  Explained Variance:{EXPLAINED_VARIANCE}")
    print(f"  Window/Step (samp):{WINDOW_SIZE}/{STEP_SIZE}")
    print(f"  Sampling Rate (Hz):{SAMPLING_RATE}")
    print("-" * 70)
    print("Statistical Parameters:")
    print(f"  Cluster Perms:     {N_PERM}")
    print(f"  P-value Threshold: {P_THRESH}")
    print(f"  Tails:             {TAILS}")
    print("=" * 70)
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


