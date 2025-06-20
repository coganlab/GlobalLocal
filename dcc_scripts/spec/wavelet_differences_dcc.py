import os
import sys
import argparse
import mne
import numpy as np
import matplotlib.pyplot as plt 
from typing import List, Tuple, Optional
#print(sys.path)
#sys.path.append("C:/Users/jz421/Desktop/GlobalLocal/IEEG_Pipelines/")  # need to do this cuz otherwise ieeg isn't added to path...

# Get the absolute path to the directory containing the current script
# For GlobalLocal/src/analysis/preproc/make_epoched_data.py, this is GlobalLocal/src/analysis/preproc
try:
    # This will work if running as a .py script
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # This will be executed if __file__ is not defined (e.g., in a Jupyter Notebook)
    # os.getcwd() often gives the directory of the notebook,
    # or the directory from which the Jupyter server was started.
    current_script_dir = os.getcwd()

# Navigate up two levels to get to the 'GlobalLocal' directory
project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..'))

# Add the 'GlobalLocal' directory to sys.path if it's not already there
if project_root not in sys.path:
    sys.path.insert(0, project_root) # insert at the beginning to prioritize it

import mne.time_frequency
import mne
from ieeg.io import get_data, raw_from_layout
from ieeg.navigate import trial_ieeg, channel_outlier_marker, crop_empty_data, outliers_to_nan
from ieeg.calc.scaling import rescale
from ieeg.calc.fast import mean_diff
from ieeg.calc.stats import time_perm_cluster
import os
from ieeg.timefreq.utils import wavelet_scaleogram, crop_pad
from ieeg.viz.ensemble import chan_grid
from ieeg.viz.parula import parula_map
from scipy.stats import ttest_ind

from src.analysis.utils.general_utils import calculate_RTs, get_good_data
from src.analysis.spec.wavelet_functions import (
    load_wavelets,
    make_and_get_sig_wavelet_differences,
    load_and_get_sig_wavelet_differences,
    load_and_get_sig_wavelet_ratio_differences,
    plot_mask_pages
)

HOME = os.path.expanduser("~")
USER = os.path.basename(HOME)

# get box directory depending on OS
if os.name == 'nt': # windows
    LAB_root = os.path.join(HOME, "Box", "CoganLab")
else: # mac
    LAB_root = os.path.join("/cwork", USER)

layout = get_data('GlobalLocal', root=LAB_root)

# subjects = ['D0057', 'D0059', 'D0063', 'D0065', 'D0069', 'D0071', 'D0077', 'D0090', 'D0094', 'D0100', 'D0102', 'D0103', 'D0107A', 'D0110']
subjects = ['D0094']

# this is a toggle for which version to run - the one that makes the wavelets in this notebook directly, or the one that loads them
make_wavelets=False

save_dir = os.path.join(layout.root, 'derivatives', 'spec', 'wavelet', 'figs')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Mapping condition names to their corresponding output names and event lists
conditions_and_output_names_and_events = {
    'error': {
        'output_name': 'ErrorTrials_Stimulus_Locked',
        'events': ['Stimulus/Accuracy0.0']
    },
    'correct': {
        'output_name': 'CorrectTrials_Stimulus_Locked',
        'events': ['Stimulus/Accuracy1.0']
    }   
}

times = [-0.5, 1.5]
make_wavelets=False

if make_wavelets:
    # For making wavelets, extract events directly from the configuration dictionary.
    incongruent_events = conditions_and_output_names_and_events['incongruent']['events']
    congruent_events   = conditions_and_output_names_and_events['congruent']['events']
    switch_events      = conditions_and_output_names_and_events['switch']['events']
    repeat_events      = conditions_and_output_names_and_events['repeat']['events']

    sig_wavelet_differences_per_subject = {}

    for sub in subjects:

        sig_wavelet_differences_per_subject[sub] = {}
        # Preprocess and compute average reaction time (if desired)
        good = get_good_data(sub, layout)
        RTs, skipped = calculate_RTs(good)
        avg_RT = np.median(RTs)
        print(f"Subject {sub} average RT: {avg_RT}")

        # do inc-con, and also switch-repeat
        congruency_mask, congruency_pvals = make_and_get_sig_wavelet_differences(
            sub, layout, incongruent_events, congruent_events, times,
            stat_func=ttest_ind, p_thresh=0.05, ignore_adjacency=1, n_perm=100, n_jobs=1)
        switch_type_mask, switch_type_pvals = make_and_get_sig_wavelet_differences(
            sub, layout, switch_events, repeat_events, times,
            stat_func=ttest_ind, p_thresh=0.05, ignore_adjacency=1, n_perm=100, n_jobs=1)

        sig_wavelet_differences_per_subject[sub]['congruency'] = (congruency_mask, congruency_pvals)
        sig_wavelet_differences_per_subject[sub]['switch_type'] = (switch_type_mask, switch_type_pvals)

else:
    # For loading precomputed wavelets, use output names from the configuration.
    # Here, 'rescaled' toggles whether to load baseline-corrected (rescaled) TFRs.
    rescaled = False  # Set to True if you wish to use baseline-corrected (dB scaled) wavelets

    sig_wavelet_differences_per_subject = {}

    for sub in subjects:

        sig_wavelet_differences_per_subject[sub] = {}
        good = get_good_data(sub, layout)
        RTs, skipped = calculate_RTs(good)
        avg_RT = np.median(RTs)
        print(f"Subject {sub} average RT: {avg_RT}")

        # do inc-con, and also switch-repeat
        accuracy_mask, accuracy_pvals = load_and_get_sig_wavelet_differences(
            sub, layout,
            conditions_and_output_names_and_events['error']['output_name'],
            conditions_and_output_names_and_events['correct']['output_name'],
            rescaled,
            stat_func=mean_diff, p_thresh=0.05, ignore_adjacency=1, n_perm=100, n_jobs=1)
        ##switch_type_mask, switch_type_pvals = load_and_get_sig_wavelet_differences(
            ##sub, layout,
            ##conditions_and_output_names_and_events['switch']['output_name'],
            ##conditions_and_output_names_and_events['repeat']['output_name'],
            ##rescaled,
            ##stat_func=mean_diff, p_thresh=0.05, ignore_adjacency=1, n_perm=100, n_jobs=1)

        sig_wavelet_differences_per_subject[sub]['accuracy'] = (accuracy_mask, accuracy_pvals)
        #sig_wavelet_differences_per_subject[sub]['switch_type'] = (switch_type_mask, switch_type_pvals)

        # plot the sig cluster masks
        # Assume mask has shape (n_channels, ...) and you have channel names:
        # For example, you can get channel names from one of your TFR objects.
        # (Here we load one TFR to extract channel names.)
        error_spec = load_wavelets(sub, layout, conditions_and_output_names_and_events['error']['output_name'], rescaled)
        correct_spec = load_wavelets(sub, layout, conditions_and_output_names_and_events['correct']['output_name'], rescaled)
        #switch_spec = load_wavelets(sub, layout, conditions_and_output_names_and_events['switch']['output_name'], rescaled)
        #repeat_spec = load_wavelets(sub, layout, conditions_and_output_names_and_events['repeat']['output_name'], rescaled)

        ch_names = error_spec.ch_names  # list of channel names

        # Now plot the mask pages:
        accuracy_mask_pages = plot_mask_pages(accuracy_mask,
                            error_spec.ch_names,
                            times=error_spec.times,
                            freqs=error_spec.freqs,
                            channels_per_page=60,
                            grid_shape=(6, 10),
                            cmap=parula_map,
                            title_prefix=f"{sub} ",
                            log_freq=True,
                            show=False)
        
    ##
    #   switch_type_mask_pages = plot_mask_pages(switch_type_mask,
    #                      switch_spec.ch_names,
    #                       times=switch_spec.times,
    #                       freqs=switch_spec.freqs,
    #                        channels_per_page=60,
    #                       grid_shape=(6, 10),
    #                        cmap=parula_map,
    #                        title_prefix=f"{sub} ",
    #                        log_freq=True,
    #                        show=False)
        

        # Save each page as a separate figure file:
        for i, fig in enumerate(accuracy_mask_pages):
            if rescaled:
                fig_name = f"{sub}_err-corr_sig_wavelet_clusters_{conditions_and_output_names_and_events['error']['output_name']}-{conditions_and_output_names_and_events['correct']['output_name']}_rescaled_page_{i+1}.jpg"
            else:
                fig_name = f"{sub}_err-corr_sig_wavelet_clusters_{conditions_and_output_names_and_events['error']['output_name']}-{conditions_and_output_names_and_events['correct']['output_name']}_uncorrected_page_{i+1}.jpg"
            fig_pathname = os.path.join(save_dir, fig_name)
            fig.savefig(fig_pathname, bbox_inches='tight')
            print("Saved figure:", fig_name)

        #for i, fig in enumerate(switch_type_mask_pages):
        #   if rescaled:
        #        fig_name = f"{sub}_switch-repeat_sig_wavelet_clusters_{conditions_and_output_names_and_events['switch']['output_name']}-{conditions_and_output_names_and_events['repeat']['output_name']}_rescaled_page_{i+1}.jpg"
        #    else:
        #        fig_name = f"{sub}_switch-repeat_sig_wavelet_clusters_{conditions_and_output_names_and_events['switch']['output_name']}-{conditions_and_output_names_and_events['repeat']['output_name']}_uncorrected_page_{i+1}.jpg"
        #    fig_pathname = os.path.join(save_dir, fig_name)
        #    fig.savefig(fig_pathname, bbox_inches='tight')
        #   print("Saved figure:", fig_name)

        # get the mean differences themselves and plot them
        mean_diff_err_vs_corr = mean_diff(error_spec._data, correct_spec._data, axis=0)
        #mean_diff_switch_vs_repeat = mean_diff(switch_spec._data, repeat_spec._data, axis=0)

        # Now, plot the mean differences directly:
        accuracy_mean_diff_pages = plot_mask_pages(
            mean_diff_err_vs_corr,
            error_spec.ch_names,
            times=error_spec.times,
            freqs=error_spec.freqs,
            grid_shape=(6, 10),
            cmap=parula_map,  # play with color maps
            title_prefix=f"{sub} Mean Err_Corr Diff: ",
            log_freq=True,
            show=False
        )

        #switch_type_mean_diff_pages = plot_mask_pages(
        #    mean_diff_switch_vs_repeat,
        #    switch_spec.ch_names,
        #   times=switch_spec.times,
        #   freqs=switch_spec.freqs,
        #   grid_shape=(6, 10),
        #   cmap=parula_map,  # play with color maps
        #   title_prefix=f"{sub} Mean switch-repeat Diff: ",
        #   log_freq=True,
        #   show=False
        #)

        # Save each page as a separate figure file:
        for i, fig in enumerate(accuracy_mean_diff_pages):
            if rescaled:
                fig_name = f"{sub}_err-corr_mean_diff_{conditions_and_output_names_and_events['error']['output_name']}-{conditions_and_output_names_and_events['correct']['output_name']}_rescaled_page_{i+1}.jpg"
            else:
                fig_name = f"{sub}_err-corr_mean_diff_{conditions_and_output_names_and_events['error']['output_name']}-{conditions_and_output_names_and_events['correct']['output_name']}_uncorrected_page_{i+1}.jpg"
            fig_pathname = os.path.join(save_dir, fig_name)
            fig.savefig(fig_pathname, bbox_inches='tight')
            print("Saved figure:", fig_name)

        # Save each page as a separate figure file:
        #for i, fig in enumerate(switch_type_mean_diff_pages):
        #    if rescaled:
        #        fig_name = f"{sub}_switch-repeat_mean_diff_{conditions_and_output_names_and_events['switch']['output_name']}-{conditions_and_output_names_and_events['repeat']['output_name']}_rescaled_page_{i+1}.jpg"
        #    else:
        #        fig_name = f"{sub}_switch-repeat_mean_diff_{conditions_and_output_names_and_events['switch']['output_name']}-{conditions_and_output_names_and_events['repeat']['output_name']}_uncorrected_page_{i+1}.jpg"
        #    fig_pathname = os.path.join(save_dir, fig_name)
        #    fig.savefig(fig_pathname, bbox_inches='tight')
        #    print("Saved figure:", fig_name)