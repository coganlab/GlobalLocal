
#!/usr/bin/env python
"""
Submit script for power traces analysis.
This sets up input args for and calls power_traces_dcc.py
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
from dcc_scripts.power.power_traces_dcc import main
from src.analysis.config import experiment_conditions

# ============================================================================
# ANALYSIS PARAMETERS
# ============================================================================
LAB_ROOT = None  # Will be determined automatically in main()

# Subject configuration
# remove D0110 because of low error trials
SUBJECTS = ['D0059', 'D0069', 'D0077', 'D0090', 'D0094', 'D0102', 'D0103', 'D0107A', 'D0121']
# SUBJECTS = ['D0057', 'D0059', 'D0063', 'D0069', 'D0071', 'D0077', 'D0090', 'D0094', 'D0100', 'D0102', 'D0103', 'D0107A', 'D0110', 'D0116', 'D0117', 'D0121']
# subjects for err/corr
# SUBJECTS = ['D0057', 'D0059', 'D0063', 'D0069', 'D0077', 'D0090', 'D0094', 'D0100', 'D0102', 'D0103', 'D0107A', 'D0116', 'D0117', 'D0121']

# task
TASK = 'GlobalLocal'

# Trial selection
# switch to False for err-corr 
ACC_TRIALS_ONLY = False

# Parallel processing
N_JOBS = -1 

# stats
STATISTICAL_METHOD = 'time_perm_cluster' # 'time_perm_cluster' or 'anova'
SAMPLING_RATE = 256 # Or whatever your decimated sampling rate is (e.g., 100 Hz)
WINDOW_SIZE = None # Sliding window size in samples. Set to None for time perm cluster stats. This is just for ANOVA.

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
    
P_THRESH_FOR_TIME_PERM_CLUSTER_STATS = 0.05
P_CLUSTER = 0.05
PERMUTATION_TYPE = 'independent'
N_PERM = 500
TAILS=1

# ============================================================================
# Condition selection
CONDITIONS = experiment_conditions.stimulus_iR_cS_err_conditions

# Epochs file selection
#EPOCHS_ROOT_FILE = "Stimulus_0.5sec_within-1.0-0.0sec_base_decFactor_8_outliers_10_drop_thresh_perc_5.0_70.0-150.0_Hz_padLength_0.5s_stat_func_ttest_ind_equal_var_False_nan_policy_omit"
# EPOCHS_ROOT_FILE = "Stimulus_0.5sec_within-1.0-0.0sec_base_decFactor_8_outliers_10_drop_and_nan_thresh_perc_5.0_70.0-150.0_Hz_padLength_0.5s_stat_func_ttest_ind_equal_var_False_nan_policy_omit"
EPOCHS_ROOT_FILE = "Stimulus_0.5sec_within-1-0sec_randoffset_StimulusBase_decFactor_8_markOutliersAsNaN_False_passband_70.0-150.0_padLength_0.5s_stat_func_ttest_ind_equal_var_False"
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
ELECTRODES = 'sig'

# plotting
YLIM = (-0.1,0.5)
SHOW_LEGEND = False

# # # # testing params (comment out)
# SUBJECTS = ['D0103']
# N_PERM = 2
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
        task=TASK,
        conditions=CONDITIONS,
        epochs_root_file=EPOCHS_ROOT_FILE,
        rois_dict=ROIS_DICT,
        electrodes=ELECTRODES,
        p_thresh_for_time_perm_cluster_stats=P_THRESH_FOR_TIME_PERM_CLUSTER_STATS,
        p_cluster=P_CLUSTER,
        stat_func=STAT_FUNC,
        permutation_type=PERMUTATION_TYPE,
        stat_func_str=STAT_FUNC_STR,
        statistical_method=STATISTICAL_METHOD,
        sampling_rate=SAMPLING_RATE,
        window_size=WINDOW_SIZE,
        n_perm=N_PERM,
        tails=TAILS,
        ylim=YLIM,
        show_legend=SHOW_LEGEND
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
    print("-" * 70)
    
    print("Perm Cluster Statistical Parameters:")
    print(f"  P-value Threshold: {P_THRESH_FOR_TIME_PERM_CLUSTER_STATS}")
    print(f"  P cluster: {P_CLUSTER}")
    print(f"  Stat Func: {STAT_FUNC_STR}")
    print(f"  Permutation Type: {PERMUTATION_TYPE}")
    print(f"  Number of Permutations: {N_PERM}")
    print(f"  Tails: {TAILS}")
    print("=" * 70)

    print('plotting parameters:')
    print(f'  y limits: {YLIM}')
    print(f'  show legend: {SHOW_LEGEND}')
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


