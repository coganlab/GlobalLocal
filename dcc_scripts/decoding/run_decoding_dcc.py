
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
from scipy.stats import ttest_ind, ttest_rel
from types import SimpleNamespace
from datetime import datetime
from ieeg.calc.fast import mean_diff

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC

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
# subjects for err-corr
SUBJECTS = ['D0057', 'D0063', 'D0065', 'D0069', 'D0077', 'D0094', 'D0100', 'D0102', 'D0103', 'D0107A', 'D0116', 'D0117', 'D0121']
# SUBJECTS = ['D0065', 'D0069', 'D0077', 'D0102', 'D0103', 'D0121']
# SUBJECTS = ['D0057', 'D0059', 'D0063', 'D0069', 'D0071', 'D0077', 'D0090', 'D0094', 'D0100', 'D0102', 'D0103', 'D0107A', 'D0110', 'D0116', 'D0117', 'D0121']

# task
TASK = 'GlobalLocal'

# Trial selection
# switched to False for err-corr decoding
ACC_TRIALS_ONLY = False

# Parallel processing
N_JOBS = -1 


# DECODING PARAMETERS
# First, choose your classifier
MODEL_CHOICE = 'LDA'  # Options: 'LDA', 'SVC'

if MODEL_CHOICE == 'LDA':
    CLF_MODEL = LinearDiscriminantAnalysis()
    CLF_MODEL_STR = 'LDA'
elif MODEL_CHOICE == 'SVC':
    # You can configure your model here
    CLF_MODEL = SVC(C=1.0, kernel='linear', probability=False)
    CLF_MODEL_STR = 'SVC_C1_linear'
else:
    raise ValueError(f"Unknown MODEL_CHOICE: {MODEL_CHOICE}")

# Then, choose your decoding parameters
N_SPLITS = 5
N_REPEATS = 5
RANDOM_STATE = 42
EXPLAINED_VARIANCE = 0.90
BALANCE_METHOD = 'subsample'
NORMALIZE = 'true'
BOOTSTRAPS = 20
OBS_AXS = 0
CHANS_AXS = 1
TIME_AXS = -1

# Time-windowed decoding parameters
WINDOW_SIZE = 64  # Window size in samples (e.g., 64 samples = 250 ms at 256 Hz)
STEP_SIZE = 16    # Step size in samples (e.g., 16 samples = 62.5 ms at 256 Hz)
SAMPLING_RATE = 256 # Sampling rate of the data in Hz
FIRST_TIME_POINT = -1.0 # The time in seconds of the first sample in the epoch
TAILS = 1 # 1 for one-tailed (e.g., accuracy > chance), 2 for two-tailed
N_SHUFFLE_PERMS = 50 # how many times to shuffle labels and train decoder to make chance decoding results - this iterates over splits, so end up with N_SHUFFLE_PERMS * N_SPLITS for number of folds

# whether to do stats across fold, repeat, or bootstrap
UNIT_OF_ANALYSIS='repeat'

# whether to store individual folds (true) or sum them within repeats (false)
FOLDS_AS_SAMPLES = True if UNIT_OF_ANALYSIS == 'fold' else False

# percentile stats parameters
PERCENTILE=95
CLUSTER_PERCENTILE=95
N_CLUSTER_PERMS=200 # how many times to shuffle accuracies between chance and true to do cluster correction

# additional parameters for permutation cluster stats
STAT_FUNC_CHOICE = 'ttest_ind' # 'ttest_ind', 'ttest_rel' or 'mean_diff'

if STAT_FUNC_CHOICE == 'mean_diff':
    STAT_FUNC = mean_diff
    STAT_FUNC_STR = 'mean_diff'
elif STAT_FUNC_CHOICE == 'ttest_ind':
    STAT_FUNC = partial(ttest_ind, equal_var=False, nan_policy='omit')
    STAT_FUNC_STR = 'ttest_ind'
elif STAT_FUNC_CHOICE == 'ttest_rel':
    STAT_FUNC = partial(ttest_rel, nan_policy='omit')
    STAT_FUNC_STR = 'ttest_rel'
    
P_THRESH_FOR_TIME_PERM_CLUSTER_STATS = 0.025
P_CLUSTER = 0.025
PERMUTATION_TYPE = 'independent'
# CLUSTER_TAILS = 2

# plotting
SINGLE_COLUMN = True
SHOW_LEGEND = False
RUN_VISUALIZATION_DEBUG = True # Collapsed onto the first two PCs, this plots each trial and the SVM or LDA hyperplane.

# Condition selection
CONDITIONS = experiment_conditions.stimulus_switch_type_by_congruency_proportion_conditions

# Epochs file selection
EPOCHS_ROOT_FILE = "Stimulus_0.5sec_within-1.0-0.0sec_base_decFactor_8_outliers_10_drop_thresh_perc_5.0_70.0-150.0_Hz_padLength_0.5s_stat_func_ttest_ind_equal_var_False_nan_policy_omit"
# EPOCHS_ROOT_FILE = "Stimulus_0.5sec_within-1.0-0.0sec_base_decFactor_8_outliers_10_drop_and_nan_thresh_perc_5.0_70.0-150.0_Hz_padLength_0.5s_stat_func_ttest_ind_equal_var_False_nan_policy_omit"
# EPOCHS_ROOT_FILE = "Stimulus_0.5sec_within-1-0sec_randoffset_StimulusBase_decFactor_8_markOutliersAsNaN_False_passband_70.0-150.0_padLength_0.5s_stat_func_ttest_ind_equal_var_False"
# EPOCHS_ROOT_FILE = "Stimulus_0.5sec_within-1-0sec_randoffset_StimulusBase_decFactor_8_markOutliersAsNaN_False_passband_4.0-8.0_padLength_0.5s_stat_func_ttest_ind_equal_var_False"
# EPOCHS_ROOT_FILE = "Stimulus_0.5sec_within-1-0sec_randoffset_StimulusBase_decFactor_8_outlier_policy_interpolate_outliers_10_passband_70.0-150.0_padLength_0.5s_stat_func_ttest_ind_equal_var_False"
# EPOCHS_ROOT_FILE = "Stimulus_0.5sec_within1sec_randoffset_preStimulusBase_decFactor_8_outliers_10_passband_70.0-150.0_padLength_0.5s_stat_func_ttest_ind_equal_var_False"
# EPOCHS_ROOT_FILE = "Response_0.5sec_within1sec_randoffset_preStimulusBase_decFactor_8_outliers_10_passband_70.0-150.0_padLength_0.5s_stat_func_ttest_ind"

# ROI dictionary
# ROIS_DICT = {
#     'dlpfc': ["G_front_middle", "G_front_sup", "S_front_inf", "S_front_middle", "S_front_sup"],
#     'acc': ["G_and_S_cingul-Ant", "G_and_S_cingul-Mid-Ant"],
#     'parietal': ["G_parietal_sup", "S_intrapariet_and_P_trans", "G_pariet_inf-Angular", "G_pariet_inf-Supramar"],
#     'lpfc': ["G_front_inf-Opercular", "G_front_inf-Orbital", "G_front_inf-Triangul", "G_front_middle", "G_front_sup", "Lat_Fis-ant-Horizont", "Lat_Fis-ant-Vertical", "S_circular_insula_ant", "S_circular_insula_sup", "S_front_inf", "S_front_middle", "S_front_sup"],
#     'v1': ["G_oc-temp_med-Lingual", "S_calcarine", "G_cuneus"],
#     'occ': ["G_cuneus", "G_and_S_occipital_inf", "G_occipital_middle", "G_occipital_sup", "G_oc-temp_lat-fusifor", "G_oc-temp_med-Lingual", "Pole_occipital", "S_calcarine", "S_oc_middle_and_Lunatus", "S_oc_sup_and_transversal", "S_occipital_ant"]
# }

# adding parietal, dlpfc, acc for err-corr decoding
# ROIS_DICT = {
#     'lpfc': ["G_front_inf-Opercular", "G_front_inf-Orbital", "G_front_inf-Triangul", "G_front_middle", "G_front_sup", "Lat_Fis-ant-Horizont", "Lat_Fis-ant-Vertical", "S_circular_insula_ant", "S_circular_insula_sup", "S_front_inf", "S_front_middle", "S_front_sup"],
#     'occ': ["G_cuneus", "G_and_S_occipital_inf", "G_occipital_middle", "G_occipital_sup", "G_oc-temp_lat-fusifor", "G_oc-temp_med-Lingual", "Pole_occipital", "S_calcarine", "S_oc_middle_and_Lunatus", "S_oc_sup_and_transversal", "S_occipital_ant"],
#     'dlpfc': ["G_front_middle", "G_front_sup", "S_front_inf", "S_front_middle", "S_front_sup"],
#     'acc': ["G_and_S_cingul-Ant", "G_and_S_cingul-Mid-Ant"],
#     'parietal': ["G_parietal_sup", "S_intrapariet_and_P_trans", "G_pariet_inf-Angular", "G_pariet_inf-Supramar"],
# }

ROIS_DICT = {
    'lpfc': ["G_front_inf-Opercular", "G_front_inf-Orbital", "G_front_inf-Triangul", "G_front_middle", "G_front_sup", "Lat_Fis-ant-Horizont", "Lat_Fis-ant-Vertical", "S_circular_insula_ant", "S_circular_insula_sup", "S_front_inf", "S_front_middle", "S_front_sup"]
}

# which electrodes to use (all or sig)
ELECTRODES = 'all'

# # # # testing params (comment out)
# SUBJECTS = ['D0103']
# N_SPLITS = 2
# N_REPEATS = 2
# N_PERM = 2
# N_CLUSTER_PERMS= 2
# BOOTSTRAPS = 2
# N_JOBS = 1
# ROIS_DICT = {
#   'lpfc': ["G_front_inf-Opercular", "G_front_inf-Orbital", "G_front_inf-Triangul", "G_front_middle", "G_front_sup", "Lat_Fis-ant-Horizont", "Lat_Fis-ant-Vertical", "S_circular_insula_ant", "S_circular_insula_sup", "S_front_inf", "S_front_middle", "S_front_sup"]
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
        n_jobs=N_JOBS,
        tails=TAILS,
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
        task=TASK,
        conditions=CONDITIONS,
        epochs_root_file=EPOCHS_ROOT_FILE,
        rois_dict=ROIS_DICT,
        electrodes=ELECTRODES,
        clf_model=CLF_MODEL,           
        clf_model_str=CLF_MODEL_STR,  
        explained_variance=EXPLAINED_VARIANCE,
        balance_method=BALANCE_METHOD,
        bootstraps=BOOTSTRAPS,
        obs_axs=OBS_AXS,
        chans_axs=CHANS_AXS,
        time_axs=TIME_AXS,
        window_size=WINDOW_SIZE,
        step_size=STEP_SIZE,
        sampling_rate=SAMPLING_RATE,
        first_time_point=FIRST_TIME_POINT,
        folds_as_samples=FOLDS_AS_SAMPLES,
        unit_of_analysis=UNIT_OF_ANALYSIS,
        percentile=PERCENTILE,
        cluster_percentile=CLUSTER_PERCENTILE,
        n_cluster_perms=N_CLUSTER_PERMS,
        n_shuffle_perms=N_SHUFFLE_PERMS,
        p_thresh_for_time_perm_cluster_stats=P_THRESH_FOR_TIME_PERM_CLUSTER_STATS,
        p_cluster=P_CLUSTER,
        stat_func=STAT_FUNC,
        permutation_type=PERMUTATION_TYPE,
        stat_func_str=STAT_FUNC_STR,
        single_column=SINGLE_COLUMN,
        show_legend=SHOW_LEGEND,
        run_visualization_debug=RUN_VISUALIZATION_DEBUG
        # cluster_tails=CLUSTER_TAILS,
    )

    # Print configuration summary
    print("=" * 70)
    print("BANDPASS FILTERED DECODING ANALYSIS")
    print("=" * 70)
    print(f"Subjects:          {SUBJECTS}")
    print(f"Conditions:        {list(CONDITIONS.keys())}")
    print(f"ROIs:              {list(ROIS_DICT.keys())}")
    print(f"Epochs file:       {os.path.basename(EPOCHS_ROOT_FILE)}")
    print(f"Electrodes (all or sig):       {ELECTRODES}")
    print(f"Explained variance: {EXPLAINED_VARIANCE}")
    print(f"Balance method:     {BALANCE_METHOD}")
    print(f"Obs axs:            {OBS_AXS}")
    print(f"Chans axs:          {CHANS_AXS}")
    print(f"Time axs:           {TIME_AXS}")
    print("-" * 70)
    print("Decoding Parameters:")
    print(f"Classifier Model:   {CLF_MODEL_STR}")
    print(f"  CV Splits/Repeats: {N_SPLITS}/{N_REPEATS}")
    print(f"  Balance Method:    {BALANCE_METHOD}")
    print(f"  Explained Variance:{EXPLAINED_VARIANCE}")
    print(f"  Window/Step (samp):{WINDOW_SIZE}/{STEP_SIZE}")
    print(f"  Sampling Rate (Hz):{SAMPLING_RATE}")
    print("-" * 70)
    
    print("Perm Cluster Statistical Parameters:")
    print(f"  P-value Threshold: {P_THRESH_FOR_TIME_PERM_CLUSTER_STATS}")
    print(f"  P cluster: {P_CLUSTER}")
    print(f"  Stat Func: {STAT_FUNC_STR}")
    print(f"  Permutation Type: {PERMUTATION_TYPE}")
    # print(f"  Tails:             {CLUSTER_TAILS}")

    print(f" unit of analysis for stats (bootstrap, repeat, or fold): {UNIT_OF_ANALYSIS}")
    
    print("Percentile Statistical Parameters:")
    print(f"  Percentile: {PERCENTILE}")
    print(f"  Cluster Percentile: {CLUSTER_PERCENTILE}")
    print(f"  N Cluster Perms: {N_CLUSTER_PERMS}")
    print("=" * 70)
    print("=" * 70)
    
    print("plotting params")
    print(f" single column figure: {SINGLE_COLUMN}")
    print(f" show legend: {SHOW_LEGEND}")
    print(f" run visualization debugging of first two PCs and decision boundary: {RUN_VISUALIZATION_DEBUG}")
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


