
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import sys
import os

# Get the absolute path to the directory containing the current script
# For GlobalLocal/src/analysis/preproc/make_epoched_data.py, this is GlobalLocal/src/analysis/preproc
# Get the absolute path to the directory containing the current script
try:
    # This will work if running as a .py script
    current_file_path = os.path.abspath(__file__)
    current_script_dir = os.path.dirname(current_file_path)
except NameError:
    # This will be executed if __file__ is not defined (e.g., in a Jupyter Notebook)
    current_script_dir = os.getcwd()

# Navigate up two levels to get to the 'GlobalLocal' directory
project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..'))

# Add the 'GlobalLocal' directory to sys.path if it's not already there
if project_root not in sys.path:
    sys.path.insert(0, project_root)  # insert at the beginning to prioritize it

from ieeg.navigate import channel_outlier_marker, trial_ieeg, crop_empty_data, \
    outliers_to_nan
from ieeg.io import raw_from_layout, get_data
from ieeg.timefreq.utils import crop_pad
from ieeg.timefreq import gamma
from ieeg.calc.scaling import rescale
import mne
import numpy as np
from ieeg.calc.stats import time_perm_cluster, window_averaged_shuffle
from ieeg.viz.mri import gen_labels
import logging

import matplotlib.pyplot as plt
from collections import OrderedDict, defaultdict
import json
# still need to test if the permutation test functions load in properly.
import pandas as pd
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

# from src.analysis.power.roi_analysis import blah_blah
from src.analysis.config import experiment_conditions
from src.analysis.config.plotting_parameters import plotting_parameters as plot_params
import src.analysis.utils.general_utils as utils # import utils functions one by one by name
from src.analysis.power.power_traces import make_multi_channel_evokeds_for_all_conditions_and_rois, \
                                            plot_power_traces_for_all_rois, create_subtracted_evokeds_dict, \
                                            time_perm_cluster_between_two_evokeds

# %% [markdown]
# get lab root for save dir

def main(args):
    # Determine LAB_root based on the operating system and environment
    if args.LAB_root is None:
        HOME = os.path.expanduser("~")
        USER = os.path.basename(HOME)
        
        if os.name == 'nt':  # Windows
            LAB_root = os.path.join(HOME, "Box", "CoganLab")
        elif sys.platform == 'darwin':  # macOS
            LAB_root = os.path.join(HOME, "Library", "CloudStorage", "Box-Box", "CoganLab")
        else:  # Linux (cluster)
            # Check if we're on the cluster by looking for /cwork directory
            if os.path.exists(f"/cwork/{USER}"):
                LAB_root = f"/cwork/{USER}"
            else:
                # Fallback for other Linux systems
                LAB_root = os.path.join(HOME, "CoganLab")
    else:
        LAB_root = args.LAB_root

    config_dir = os.path.join(project_root, 'src', 'analysis', 'config')
    subjects_electrodestoROIs_dict = utils.load_subjects_electrodes_to_ROIs_dict(save_dir=config_dir, filename='subjects_electrodestoROIs_dict.json')
    
    condition_names = list(args.conditions.keys()) # get the condition names as a list

    # filename is too long to save so let's just drop the epochs root file from the conditions save name for now.
    if args.conditions == experiment_conditions.stimulus_conditions:
        conditions_save_name = 'stimulus_conditions' + '_' + str(len(args.subjects)) + '_' + 'subjects'
    elif args.conditions == experiment_conditions.stimulus_experiment_conditions:
        conditions_save_name = 'stimulus_experiment_conditions' + '_' + str(len(args.subjects)) + '_' + 'subjects'
    elif args.conditions == experiment_conditions.stimulus_main_effect_conditions:
        conditions_save_name = 'stimulus_main_effect_conditions' + '_' + str(len(args.subjects)) + '_' + 'subjects'
    elif args.conditions == experiment_conditions.stimulus_lwpc_conditions:
        conditions_save_name = 'stimulus_lwpc_conditions' + '_' + str(len(args.subjects)) + '_' + 'subjects'
    elif args.conditions == experiment_conditions.stimulus_lwps_conditions:
        conditions_save_name = 'stimulus_lwps_conditions' + '_' + str(len(args.subjects)) + '_' + 'subjects'
    elif args.conditions == experiment_conditions.stimulus_big_letter_conditions:
        conditions_save_name = 'stimulus_big_letter_conditions' + '_' + str(len(args.subjects)) + '_' + 'subjects'
    elif args.conditions == experiment_conditions.stimulus_small_letter_conditions:
        conditions_save_name = 'stimulus_small_letter_conditions' + '_' + str(len(args.subjects)) + '_' + 'subjects'
    elif args.conditions == experiment_conditions.stimulus_task_conditions:
        conditions_save_name = 'stimulus_task_conditions' + '_' + str(len(args.subjects)) + '_' + 'subjects'
    elif args.conditions == experiment_conditions.stimulus_congruency_conditions:
        conditions_save_name = 'stimulus_congruency_conditions' + '_' + str(len(args.subjects)) + '_' + 'subjects'
    elif args.conditions == experiment_conditions.stimulus_switch_type_conditions:
        conditions_save_name = 'stimulus_switch_type_conditions' + '_' + str(len(args.subjects)) + '_' + 'subjects'
    elif args.conditions == experiment_conditions.stimulus_err_corr_conditions:
        conditions_save_name = 'stimulus_err_corr_conditions' + '_' + str(len(args.subjects)) + '_' + 'subjects'
    elif args.conditions == experiment_conditions.stimulus_congruency_by_switch_proportion_conditions:
        conditions_save_name = 'stimulus_congruency_by_switch_proportion_conditions' + '_' + str(len(args.subjects)) + '_' + 'subjects'
    elif args.conditions == experiment_conditions.stimulus_switch_type_by_congruency_proportion_conditions:
        conditions_save_name = 'stimulus_switch_type_by_congruency_proportion_conditions' + '_' + str(len(args.subjects)) + '_' + 'subjects'
    
    elif args.conditions == experiment_conditions.response_conditions:
        conditions_save_name = 'response_conditions' + '_' + str(len(args.subjects)) + '_' + 'subjects'
    elif args.conditions == experiment_conditions.response_experiment_conditions:
        conditions_save_name = 'response_experiment_conditions' + '_' + str(len(args.subjects)) + '_' + 'subjects'
    elif args.conditions == experiment_conditions.response_big_letter_conditions:
        conditions_save_name = 'response_big_letter_conditions' + '_' + str(len(args.subjects)) + '_' + 'subjects'
    elif args.conditions == experiment_conditions.response_small_letter_conditions:
        conditions_save_name = 'response_small_letter_conditions' + '_' + str(len(args.subjects)) + '_' + 'subjects'
    elif args.conditions == experiment_conditions.response_task_conditions:
        conditions_save_name = 'response_task_conditions' + '_' + str(len(args.subjects)) + '_' + 'subjects'
    elif args.conditions == experiment_conditions.response_congruency_conditions:
        conditions_save_name = 'response_congruency_conditions' + '_' + str(len(args.subjects)) + '_' + 'subjects'
    elif args.conditions == experiment_conditions.response_switch_type_conditions:
        conditions_save_name = 'response_switch_type_conditions' + '_' + str(len(args.subjects)) + '_' + 'subjects'
    elif args.conditions == experiment_conditions.response_err_corr_conditions:
        conditions_save_name = 'response_err_corr_conditions' + '_' + str(len(args.subjects)) + '_' + 'subjects'
    elif args.conditions == experiment_conditions.response_congruency_by_switch_proportion_conditions:
        conditions_save_name = 'response_congruency_by_switch_proportion_conditions' + '_' + str(len(args.subjects)) + '_' + 'subjects'
    elif args.conditions == experiment_conditions.response_switch_type_by_congruency_proportion_conditions:
        conditions_save_name = 'response_switch_type_by_congruency_proportion_conditions' + '_' + str(len(args.subjects)) + '_' + 'subjects'
    
    # Get data layout
    layout = get_data(args.task, root=LAB_root)
    save_dir = os.path.join(layout.root, 'derivatives', 'freqFilt', 'figs', f"{args.epochs_root_file}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Save directory created or already exists at: {save_dir}")
    
    sig_chans_per_subject = utils.get_sig_chans_per_subject(args.subjects, args.epochs_root_file, task=args.task, LAB_root=LAB_root)

    rois = list(args.rois_dict.keys())
    all_electrodes_per_subject_roi, sig_electrodes_per_subject_roi = utils.make_sig_electrodes_per_subject_and_roi_dict(args.rois_dict, subjects_electrodestoROIs_dict, sig_chans_per_subject)
      
    subjects_mne_objects = utils.create_subjects_mne_objects_dict(subjects=args.subjects, epochs_root_file=args.epochs_root_file, conditions=args.conditions, task=args.task, just_HG_ev1_rescaled=True, acc_trials_only=args.acc_trials_only)
    
    # determine which electrodes to use (all electrodes or just the task-significant ones)
    if args.electrodes == 'all':
        raw_electrodes = all_electrodes_per_subject_roi 
        elec_string_to_add_to_filename = 'all_elecs'
    elif args.electrodes == 'sig':
        raw_electrodes = sig_electrodes_per_subject_roi
        elec_string_to_add_to_filename = 'sig_elecs'

    else:
        raise ValueError("electrodes input must be set to all or sig")

    # filter electrodes to only the ones that exist in the epochs objects. This mismatch can arise due to dropping channels when making the epochs objects, because the subjects_electrodestoROIs_dict is made based on all the electrodes, with no dropping.
    electrodes = utils.filter_electrode_lists_against_subjects_mne_objects(rois, raw_electrodes, subjects_mne_objects)
    
    dropped_electrodes, _ = utils.find_difference_between_two_electrode_lists(raw_electrodes, electrodes)
    print("\n--- Summary of Dropped Electrodes ---")
    total_dropped = 0
    for roi, sub_dict in dropped_electrodes.items():
        if not sub_dict: continue # Skip ROIs with no dropped electrodes
        print(f"ROI: {roi}")
        for sub, elec_list in sub_dict.items():
            if elec_list:
                print(f"  - Subject {sub}: Dropped {len(elec_list)} electrode(s)")
                total_dropped += len(elec_list)
    print(f"Total electrodes dropped across all subjects/ROIs: {total_dropped}")
    print("-------------------------------------\n")
            
    condition_comparisons = {}

    # update these as needed!
    if args.conditions == experiment_conditions.stimulus_experiment_conditions:
        condition_comparisons['congruency'] = [['c25', 'c75'], ['i25', 'i75']]
        condition_comparisons['switchType'] = [['r25', 'r75'], ['s25', 's75']]
    elif args.conditions == experiment_conditions.stimulus_conditions:
        condition_comparisons['BigLetter'] = ['bigS', 'bigH']
        condition_comparisons['SmallLetter'] = ['smallS', 'smallH']
        condition_comparisons['Task'] = ['taskG', 'taskL']
    elif args.conditions == experiment_conditions.stimulus_big_letter_conditions:
        condition_comparisons['BigLetter'] = ['bigS', 'bigH']
    elif args.conditions == experiment_conditions.stimulus_small_letter_conditions:
        condition_comparisons['SmallLetter'] = ['smallS', 'smallH']
    elif args.conditions == experiment_conditions.stimulus_task_conditions:
        condition_comparisons['task'] = ['taskG', 'taskL']
    elif args.conditions == experiment_conditions.stimulus_congruency_conditions:
        condition_comparisons['congruency'] = [['Stimulus_c'], ['Stimulus_i']]
    elif args.conditions == experiment_conditions.stimulus_switch_type_conditions:
        condition_comparisons['switchType'] = [['Stimulus_r'], ['Stimulus_s']]
    elif args.conditions == experiment_conditions.stimulus_err_corr_conditions:
        condition_comparisons['responseType'] = [['Stimulus_err'], ['Stimulus_corr']]
        
    elif args.conditions == experiment_conditions.stimulus_lwpc_conditions:
        condition_comparisons['c25_vs_i25'] = ['c25', 'i25'] # lwpc
        condition_comparisons['c75_vs_i75'] = ['c75', 'i75']
        
        condition_comparisons['c25_vs_i75'] = ['c25', 'i75'] # control comparisons for lwpc, think more about how to interpret these
        condition_comparisons['c75_vs_i25'] = ['c75', 'i25']
        
        condition_comparisons['c25_vs_c75'] = ['c25', 'c75'] # these cross-block comparisons let me decode if there's pre-trial information about the congruency proportion
        condition_comparisons['i25_vs_i75'] = ['i25', 'i75']
        
    elif args.conditions == experiment_conditions.stimulus_lwps_conditions:
        condition_comparisons['s25_vs_r25'] = ['s25', 'r25'] # lwps
        condition_comparisons['s75_vs_r75'] = ['s75', 'r75']
        
        condition_comparisons['s25_vs_r75'] = ['s25', 'r75'] # control comparisons for lwps, think more about how to interpret these
        condition_comparisons['s75_vs_r25'] = ['s75', 'r25']

    elif args.conditions == experiment_conditions.stimulus_congruency_by_switch_proportion_conditions:
        condition_comparisons['c_in_25switchBlock_vs_i_in_25switchBlock'] = ['Stimulus_c_in_25switchBlock', 'Stimulus_i_in_25switchBlock']
        condition_comparisons['c_in_75switchBlock_vs_i_in_75switchBlock'] = ['Stimulus_c_in_75switchBlock', 'Stimulus_i_in_75switchBlock']
        condition_comparisons['c_in_25switchBlock_vs_i_in_75switchBlock'] = ['Stimulus_c_in_25switchBlock', 'Stimulus_i_in_75switchBlock']
        condition_comparisons['c_in_75switchBlock_vs_i_in_25switchBlock'] = ['Stimulus_c_in_75switchBlock', 'Stimulus_i_in_25switchBlock']
        condition_comparisons['c_in_25switchBlock_vs_c_in_75switchBlock'] = ['Stimulus_c_in_25switchBlock', 'Stimulus_c_in_75switchBlock']
        condition_comparisons['i_in_25switchBlock_vs_i_in_75switchBlock'] = ['Stimulus_i_in_25switchBlock', 'Stimulus_i_in_75switchBlock']
        
    elif args.conditions == experiment_conditions.stimulus_switch_type_by_congruency_proportion_conditions:
        condition_comparisons['s_in_25incongruentBlock_vs_r_in_25incongruentBlock'] = ['Stimulus_s_in_25incongruentBlock', 'Stimulus_r_in_25incongruentBlock']
        condition_comparisons['s_in_75incongruentBlock_vs_r_in_75incongruentBlock'] = ['Stimulus_s_in_75incongruentBlock', 'Stimulus_r_in_75incongruentBlock']
        condition_comparisons['s_in_25incongruentBlock_vs_r_in_75incongruentBlock'] = ['Stimulus_s_in_25incongruentBlock', 'Stimulus_r_in_75incongruentBlock']
        condition_comparisons['s_in_75incongruentBlock_vs_r_in_25incongruentBlock'] = ['Stimulus_s_in_75incongruentBlock', 'Stimulus_r_in_25incongruentBlock'] 
        condition_comparisons['s_in_25incongruentBlock_vs_s_in_75incongruentBlock'] = ['Stimulus_s_in_25incongruentBlock', 'Stimulus_s_in_75incongruentBlock'] 
        condition_comparisons['r_in_25incongruentBlock_vs_r_in_75incongruentBlock'] = ['Stimulus_r_in_25incongruentBlock', 'Stimulus_r_in_75incongruentBlock']

    sampling_rate = args.sampling_rate
    window_size = args.window_size

    evks_dict_elecs = make_multi_channel_evokeds_for_all_conditions_and_rois(
        subjects_mne_objects, args.subjects, rois, condition_names, 
        electrodes
    ) 
    
    if args.statistical_method == 'time_perm_cluster':
        print("\nRunning statistical tests comparing TWO conditions")
        if len(condition_names) != 2:
            raise ValueError("Time perm cluster stats requires exactly two conditions.")
        significant_clusters = {}
        p_values_dict = {}
        condition1_name = condition_names[0]
        condition2_name = condition_names[1]

        for roi in rois:
            print(f"-- Processing ROI: {roi} --")
            try:
                evoked_cond1_this_roi = evks_dict_elecs[condition1_name][roi]
                evoked_cond2_this_roi = evks_dict_elecs[condition2_name][roi]
                
                mask_roi, p_values = time_perm_cluster_between_two_evokeds(evoked_cond1_this_roi, evoked_cond2_this_roi, p_thresh=args.p_thresh_for_time_perm_cluster_stats, 
                                                                 p_cluster=args.p_cluster, n_perm=args.n_perm, tails=args.tails, 
                                                                 axis=0, stat_func=args.stat_func, ignore_adjacency=None, 
                                                                 permutation_type=args.permutation_type, vectorized=True, 
                                                                 n_jobs=args.n_jobs, seed=None, verbose=True)
                significant_clusters[roi] = mask_roi
                p_values_dict[roi] = p_values
            except KeyError:
                print(f"   Skipping ROI {roi}: Missing prepared evoked data for one or both conditions.")

    elif args.statistical_method == 'anova':
        raise NotImplementedError("ANOVA statistical method not yet implemented for power traces.")
    
    # now let's plot results
    evks_dict_elecs = make_multi_channel_evokeds_for_all_conditions_and_rois(
        subjects_mne_objects, args.subjects, rois, condition_names, 
        electrodes
    ) 

    plot_power_traces_for_all_rois(
        evks_dict_elecs, rois, condition_names, conditions_save_name, plot_params,
        args.window_size, args.sampling_rate, 
        significant_clusters=significant_clusters,
        save_dir=save_dir,
        error_type='sem', figsize=(12, 8), 
        x_label='Time from Stimulus Onset (s)', 
        y_label='Power (z)',
        axis_font_size=35, tick_font_size=24, title_font_size=40, save_name_suffix=elec_string_to_add_to_filename
    )
    
    # init subtracted evokeds dict
    subtracted_evks_dict_elecs = None
    
    # subtract conditions from each other for lwpc, lwps, and cross-construct comparisons
    if args.conditions == experiment_conditions.stimulus_lwpc_conditions:
        subtraction_pairs = [("Stimulus_i75", "Stimulus_c75"), ("Stimulus_i25", "Stimulus_c25")]
        subtraction_pairs_condition_names = ['-'.join(pair) for pair in subtraction_pairs]
        subtraction_pairs_conditions_save_name = 'stimulus_i75-c75_vs_i25-c25' + '_' + args.epochs_root_file + '_' + str(len(args.subjects)) + '_' + 'subjects'
        subtracted_evks_dict_elecs = create_subtracted_evokeds_dict(evks_dict_elecs, subtraction_pairs, rois)

    elif args.conditions == experiment_conditions.stimulus_lwps_conditions:
        subtraction_pairs = [("Stimulus_s75", "Stimulus_r75"), ("Stimulus_s25", "Stimulus_r25")]
        subtraction_pairs_condition_names = ['-'.join(pair) for pair in subtraction_pairs]
        subtraction_pairs_conditions_save_name = 'stimulus_s75-r75_vs_s25-r25' + '_' + args.epochs_root_file + '_' + str(len(args.subjects)) + '_' + 'subjects'
        subtracted_evks_dict_elecs = create_subtracted_evokeds_dict(evks_dict_elecs, subtraction_pairs, rois)

    elif args.conditions == experiment_conditions.stimulus_switch_type_by_congruency_proportion_conditions:
        raise NotImplementedError("this condition hasn't been made yet, also it would require reworking my create_subtracted_evokeds_dict to take in a tuple of lists of conditions for the subtraction")
    elif args.conditions == experiment_conditions.stimulus_congruency_by_switch_proportion_conditions:
        raise NotImplementedError("this condition hasn't been made yet, also it would require reworking my create_subtracted_evokeds_dict to take in a tuple of lists of conditions for the subtraction")

    if subtracted_evks_dict_elecs is not None:
        plot_power_traces_for_all_rois(
            subtracted_evks_dict_elecs, rois, subtraction_pairs_condition_names, subtraction_pairs_conditions_save_name, plot_params,
            save_dir=save_dir,
        window_size=args.window_size, sampling_rate=args.sampling_rate,
        error_type='sem', figsize=(12, 8), 
        x_label='Time from Stimulus Onset (s)', 
        y_label='Power (z)',
        axis_font_size=35, tick_font_size=24, title_font_size=40, save_name_suffix=elec_string_to_add_to_filename
    )

if __name__ == "__main__":
    # This block is only executed when someone runs this script directly
    # Since your run script calls main() directly, this block won't be executed
    # But we'll keep it minimal for compatibility
    
    # Check if being called with SimpleNamespace (from run script)
    import sys
    if len(sys.argv) == 1:
        # No command line arguments, must be imported and called from run script
        pass
    else:
        # Someone is trying to run this directly with command line args
        print("This script should be called via run_power_traces_dcc.py")
        print("Direct command-line execution is not supported with complex parameters.")
        sys.exit(1)
# %%
# TODO: hm i should make a function that can take in stimulus_experiment_conditions and create evokeds based on chosen comparisons (i.e., i75,c75,i25,c25)

# %% [markdown]
# 


