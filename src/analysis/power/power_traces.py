import sys
import os
print(sys.path)

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

import numpy as np
import mne
import matplotlib.pyplot as plt
from typing import Union, List, Sequence, Optional, Dict, Tuple
import logging
from ieeg.calc.stats import time_perm_cluster
from numpy.lib.stride_tricks import sliding_window_view
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from scipy import stats
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from joblib import Parallel, delayed
from src.analysis.decoding.decoding import (
    find_significant_clusters_of_series_vs_distribution_based_on_percentile,
)
from src.analysis.utils.general_utils import make_or_load_subjects_electrodes_to_ROIs_dict, \
                                            identify_bad_channels_by_trial_nan_rate, \
                                            impute_trial_nans_by_channel_mean, \
                                            create_subjects_mne_objects_dict, \
                                            filter_electrode_lists_against_subjects_mne_objects, \
                                            find_difference_between_two_electrode_lists, windower, \
                                            _subdir

from src.analysis.config.plotting_parameters import plotting_parameters                                                                                

DEFAULT_PLOT_STYLE = {
    # Toggles
    'show_title': True,
    'show_xlabel': True,
    'show_ylabel': True,
    'show_legend': True,
    
    # Labels
    'title': None,        # None = auto-generate from ROI name
    'x_label': 'Time (s)',
    'y_label': 'Power (z)',
    
    # Font sizes
    'title_font_size': 14,
    'axis_font_size': 12,
    'tick_font_size': 12,
    'legend_font_size': 10,
    
    # Tick customization
    'xticks': None,       # None = auto, or pass array
    'yticks': None,
    'xtick_labels': None, # Custom labels for xticks
    'ytick_labels': None,
    'xlim': None,
    'ylim': None,
    
    # Other
    'figsize': (12, 8),
    'text_color': '#002060',
    'sig_cluster_height': 0.3,
}

def combine_single_channel_evokeds(single_channel_evokeds, ch_type='seeg'):
    """
    Combine a list of single-channel evoked objects into one multi-channel evoked object.
    
    Parameters:
    -----------
    single_channel_evokeds : list of mne.Evoked
        List of single-channel evoked objects to combine
        
    Returns:
    --------
    combined_evoked : mne.Evoked
        Multi-channel evoked object
    """
    if not single_channel_evokeds:
        return None
    
    # Get the first evoked as a template
    template = single_channel_evokeds[0].copy()
    
    # Stack all the data from single channels
    all_data = []
    all_ch_names = []
    
    for evk in single_channel_evokeds:
        all_data.append(evk.data)
        all_ch_names.extend(evk.ch_names)
    
    # Create new data array with shape (n_channels, n_times)
    combined_data = np.vstack(all_data)
    
    # Create new info with all channels
    info = mne.create_info(
        ch_names=all_ch_names,
        sfreq=template.info['sfreq'],
        ch_types=ch_type  # probably all are sEEG
    )
    
    # Create the combined evoked object
    combined_evoked = mne.EvokedArray(
        data=combined_data,
        info=info,
        tmin=template.tmin,
        nave=template.nave,
        comment=template.comment
    )
    
    return combined_evoked

def get_subject_electrodes_for_roi(subject, roi, electrodes_per_subject_roi):
    """
    Get electrodes for a specific subject and ROI.
    
    Parameters:
    -----------
    subject : str
        Subject ID
    roi : str
        ROI name
    electrodes_per_subject_roi : dict
        Dictionary mapping ROIs to subjects and their electrodes. Example is sig_electrodes_per_subject_roi[roi][subject]
    
    Returns:
    --------
    list
        List of electrode names, empty if none found
    """
    return electrodes_per_subject_roi.get(roi, {}).get(subject, [])

def get_evoked_for_specific_subject_and_condition(subjects_mne_objects, subject, condition_name, 
                      mne_object_type='HG_ev1_power_rescaled'):
    """
    Get the trial-averaged evoked object for a specific subject and condition.
    
    Parameters:
    -----------
    subjects_mne_objects : dict
        Nested dictionary with MNE objects
    subject : str
        Subject ID
    condition_name : str
        Condition name
    mne_object_type : str
        Which MNE object to use
    
    Returns:
    --------
    mne.Evoked
        Evoked object for the subject and condition
    """
    return subjects_mne_objects[subject][condition_name][mne_object_type + "_avg"].copy()
    
def extract_single_electrode_evokeds(evoked, electrode_names):
    """
    Extract individual evoked objects for each electrode.
    
    Parameters:
    -----------
    evoked : mne.Evoked
        Multi-channel evoked object
    electrode_names : list
        List of electrode names to extract
    
    Returns:
    --------
    list
        List of single-electrode evoked objects
    """
    single_electrode_evokeds = []
    
    # First pick only the specified electrodes
    evoked_subset = evoked.copy().pick_channels(electrode_names)
    
    # Then create individual evoked objects for each electrode
    for ch_name in evoked_subset.ch_names:
        evoked_single = evoked_subset.copy().pick_channels([ch_name])
        single_electrode_evokeds.append(evoked_single)
    
    return single_electrode_evokeds

def create_list_of_single_channel_evokeds_across_subjects_for_roi_and_condition(subjects_mne_objects, subjects, roi, electrodes_per_subject_roi, 
                           condition_name, mne_object_type='HG_ev1_power_rescaled'):
    """
    Create lists of single-electrode evoked objects for each condition across all electrodes in an ROI.
    
    Parameters:
    -----------
    subjects_mne_objects : dict
        Nested dictionary with MNE objects
    subjects : list
        List of subject IDs
    roi : str
        ROI name (e.g., 'lpfc', 'occ')
    electrodes_per_subject_roi : dict
        Dictionary mapping ROIs to subjects and their electrodes
    condition_name : str
        Condition name to process
    mne_object_type : str
        Which MNE object to use (default: 'HG_ev1_power_rescaled')
    
    Returns:
    --------
    List
        A list of evokeds where each entry is a trial-averaged evoked object for a significant electrode, and a similar list for all electrodes.
    """

    all_evokeds_electrodes = []
    
    for sub in subjects:
        # Get the trial-averaged evoked for this subject and condition
        evoked = get_evoked_for_specific_subject_and_condition(subjects_mne_objects, sub, condition_name, mne_object_type)
        
        # Get electrode lists for this subject and ROI
        electrodes = get_subject_electrodes_for_roi(sub, roi, electrodes_per_subject_roi)
        
        if not electrodes:
            continue
            
        # Extract single-electrode evokeds for significant electrodes
        evoked_electrodes_for_this_subject = extract_single_electrode_evokeds(evoked, electrodes)
        all_evokeds_electrodes.extend(evoked_electrodes_for_this_subject)
            
    return all_evokeds_electrodes

def make_evoked_electrode_lists_for_rois(subjects_mne_objects, subjects, rois, 
                                       electrodes_per_subject_roi, 
                                       condition_name, mne_object_type='HG_ev1_power_rescaled'):
    """
    Create evoked electrode lists for all ROIs for a specific condition.
    
    Parameters:
    -----------
    subjects_mne_objects : dict
        Nested dictionary with MNE objects
    subjects : list
        List of subject IDs
    rois : list
        List of ROI names
    electrodes_per_subject_roi : dict
        Dictionary mapping ROIs to subjects and their all electrodes
    condition_name : str
        Condition name to process
    mne_object_type : str
        Which MNE object to use
    
    Returns:
    --------
    dict
        Dictionary with ROI names as keys and evokeds as values
    """
    out = {}
    for roi in rois:
        evokeds = create_list_of_single_channel_evokeds_across_subjects_for_roi_and_condition(
            subjects_mne_objects, subjects, roi, electrodes_per_subject_roi, 
            condition_name, mne_object_type
        )
        out[roi] = evokeds
    return out

def make_evoked_electrode_lists_for_all_conditions_and_rois(subjects_mne_objects, subjects, rois, 
                                                   condition_names, electrodes_per_subject_roi, 
                                                   mne_object_type='HG_ev1_power_rescaled'):
    """
    Create evoked electrode lists for all conditions and ROIs.
    
    Parameters:
    -----------
    subjects_mne_objects : dict
        Nested dictionary with MNE objects
    subjects : list
        List of subject IDs
    rois : list
        List of ROI names
    condition_names : list
        List of condition names
    electrodes_per_subject_roi : dict
        Dictionary mapping ROIs to subjects and their electrodes
    mne_object_type : str
        Which MNE object to use
    
    Returns:
    --------
    dict
        Nested dictionary: condition_name -> roi -> evokeds
    """
    out = {}
    for condition_name in condition_names:
        out[condition_name] = make_evoked_electrode_lists_for_rois(
            subjects_mne_objects, subjects, rois,
            electrodes_per_subject_roi, condition_name, mne_object_type
        )
    return out

def make_multi_channel_evokeds_for_all_conditions_and_rois(subjects_mne_objects, subjects, rois, 
                                                           condition_names, electrodes_per_subject_roi,
                                                           mne_object_type='HG_ev1_power_rescaled'):
    """
    Create multi-channel evoked objects for all conditions and ROIs by combining single-channel evokeds.
    
    Parameters:
    -----------
    subjects_mne_objects : dict
        Nested dictionary with MNE objects
    subjects : list
        List of subject IDs
    rois : list
        List of ROI names
    condition_names : list
        List of condition names
    electrodes_per_subject_roi : dict
        Dictionary mapping ROIs to subjects and their electrodes
    mne_object_type : str
        Which MNE object to use (default: 'HG_ev1_power_rescaled')
    
    Returns:
    --------
    dict
        Nested dictionary: condition_name -> roi -> multi-channel evoked object
    """
    # First get all single-channel evokeds
    evks_dict_single_elecs = make_evoked_electrode_lists_for_all_conditions_and_rois(
        subjects_mne_objects, subjects, rois, condition_names, 
        electrodes_per_subject_roi, mne_object_type
    )
    
    # Now combine them into multi-channel evokeds
    evks_dict_multi_elecs = {}
    
    for condition_name in condition_names:
        evks_dict_multi_elecs[condition_name] = {}
        
        for roi in rois:
            single_channel_evks = evks_dict_single_elecs[condition_name][roi]
            combined_evk = combine_single_channel_evokeds(single_channel_evks)
            evks_dict_multi_elecs[condition_name][roi] = combined_evk
            
    return evks_dict_multi_elecs
    
def create_roi_grand_average(subjects_mne_objects, subjects, roi, electrodes_per_subject_roi,
                           condition_names, mne_object_type='HG_ev1_power_rescaled'):
    """
    Create grand average evoked objects for each condition across all electrodes in an ROI.
    
    Parameters:
    -----------
    subjects_mne_objects : dict
        Nested dictionary with MNE objects
    subjects : list
        List of subject IDs
    roi : str
        ROI name (e.g., 'lpfc', 'occ')
    electrodes_per_subject_roi : dict
        Dictionary mapping ROIs to subjects and their electrodes
    condition_names : list
        List of condition names to process
    mne_object_type : str
        Which MNE object to use (default: 'HG_ev1_power_rescaled')
    
    Returns:
    --------
    dict
        Dictionary with condition names as keys and grand average evoked objects across all or sig electrodes in this ROI as values. AKA first trial average within each electrode, then average across electrodes. Also return SEM across electrodes.
    """
    grand_averages_electrodes = {}
    
    for condition_name in condition_names:
        all_evokeds_electrodes = create_list_of_single_channel_evokeds_across_subjects_for_roi_and_condition(subjects_mne_objects, subjects, roi, electrodes_per_subject_roi, condition_name, mne_object_type)
        grand_avg = mne.grand_average(all_evokeds_electrodes)
        grand_averages_electrodes[condition_name] = grand_avg

    return grand_averages_electrodes

def plot_power_trace_for_roi(evks_dict, roi, condition_names, conditions_save_name, 
                             plotting_parameters, significant_clusters=None, 
                             window_size=None, sampling_rate=None, save_dir=None, 
                             show_std=True, show_sem=False, show_ci=False, ci=0.95, 
                             plot_style=None, save_name_suffix=None):
    """
    Custom plot with standard deviation or standard error shading.
    
    Since MNE's plot_compare_evokeds only supports confidence intervals,
    this function manually creates plots with SD or SEM shading.
    
    Parameters:
    -----------
    evks_dict : dict
        Dictionary with condition names as keys and evoked dictionaries as values
    roi : str
        ROI name
    condition_names : list
        List of condition names to plot
    conditions_save_name : str
        Name to use for saving the plot
    plotting_parameters : dict
        Dictionary with plotting parameters for the traces.
    save_dir : str
        Directory to save the plot
    show_std : bool
        Whether to show standard deviation shading
    show_sem : bool
        Whether to show standard error of mean shading
    plot_style : dict
        Dictionary with plot style parameters for the figure settings.
    save_name_suffix : str
        Suffix to add to the save name
    significant_clusters : array-like of bool
        A boolean array indicating which time windows are part of a
        statistically significant cluster. Shape: (n_windows,).
    Returns:
    --------
    fig : matplotlib figure
    """
    # Resolve plot style with defaults
    s = {**DEFAULT_PLOT_STYLE, **(plot_style or {})}
    figsize = s['figsize']
    sig_cluster_height = s['sig_cluster_height']
    fig, ax = plt.subplots(figsize=figsize)
    
    for condition_name in condition_names:
        evoked = evks_dict[condition_name][roi]
        if evoked is None or evoked.data.shape[0] == 0:
            continue
            
        # Get plotting parameters
        param_key = None
        
        # First try exact match
        if condition_name in plotting_parameters:
            param_key = condition_name
        else:
            # Then try to find the best match by looking for the longest matching key
            best_match_length = 0
            for key in plotting_parameters.keys():
                # Check if the key is a substring of condition_name or vice versa
                if (key in condition_name or condition_name in key):
                    # Prefer longer matches to avoid "Stimulus_c" matching when "Stimulus_c25" exists
                    if len(key) > best_match_length:
                        best_match_length = len(key)
                        param_key = key
        
        if param_key and param_key in plotting_parameters:
            params = plotting_parameters[param_key]
            color = params.get('color', 'black')
            linestyle = params.get('line_style', '-')
            label = params.get('condition_parameter', condition_name)
        else:
            # Default parameters if no match found
            color = 'black'
            linestyle = '-'
            label = condition_name
        
        # Get data
        times = evoked.times
        data = evoked.data
        n_channels = data.shape[0]
        
        # Calculate mean across channels
        mean_data = np.mean(data, axis=0)
        
        # Plot mean
        ax.plot(times, mean_data, color=color, linestyle=linestyle,
                linewidth=2.5, label=label)
        
        # Add shading
        if show_std:
            std_data = np.std(data, axis=0)
            ax.fill_between(times, mean_data - std_data, mean_data + std_data,
                           alpha=0.3, color=color, linewidth=0)
        elif show_sem:
            sem_data = np.std(data, axis=0) / np.sqrt(n_channels)
            ax.fill_between(times, mean_data - sem_data, mean_data + sem_data,
                           alpha=0.3, color=color, linewidth=0)
        elif show_ci:
            ci_data = np.percentile(data, [100 * (1 - ci), 100 * ci], axis=0)
            ax.fill_between(times, ci_data[0], ci_data[1],
                           alpha=0.3, color=color, linewidth=0)

    # Find contiguous significant clusters
    def find_clusters(significant_clusters: Union[np.ndarray, List[bool], Sequence[bool]]):
        """Helper to find start and end indices of contiguous True blocks."""
        clusters = []
        in_cluster = False
        for idx, val in enumerate(list(significant_clusters)):
            if val and not in_cluster:
                # Start of a new cluster
                start_idx = idx
                in_cluster = True
            elif not val and in_cluster:
                # End of the cluster
                end_idx = idx - 1
                clusters.append((start_idx, end_idx))
                in_cluster = False
        # Handle the case where the last value is in a cluster
        if in_cluster:
            end_idx = len(list(significant_clusters)) - 1
            clusters.append((start_idx, end_idx))
        return clusters
    
    # logging.debug(f"--- For ROI: {roi} --- significant_clusters is: {significant_clusters}")

    if significant_clusters is not None:

        # logging.debug(f"    -> Not None. Trying to find and plot clusters for {roi}.")

        clusters = find_clusters(significant_clusters)

        # # Determine y position for the bars
        # max_y = np.max(mean_true_accuracy + se_true_accuracy)
        # min_y = np.min(mean_shuffle_accuracy - se_shuffle_accuracy)
        # sig_cluster_height = max_y + 0.02  # Adjust as needed
        # plt.ylim([min_y, sig_cluster_height + 0.05])  # Adjust ylim to accommodate the bars

        # Plot horizontal bars and asterisks for significant clusters
        for cluster in clusters:
            start_idx, end_idx = cluster
            
            if window_size is None:
                window_size = 0 # set to zero for point-wise analysis
                
            if window_size is None or window_size == 0:
                # Point-wise analysis: Bar spans the centers of the first/last points
                start_time = times[start_idx]
                end_time = times[end_idx]
            else:
                # Windowed analysis: Bar spans the outer edges of the first/last windows
                window_duration = window_size / sampling_rate
                start_time = times[start_idx] - (window_duration / 2)
                end_time = times[end_idx] + (window_duration / 2)
                
            plt.hlines(y=sig_cluster_height, xmin=start_time, xmax=end_time, color='black', linewidth=8)  
            # Place an asterisk at the center of the bar
            center_time = (start_time + end_time) / 2
            plt.text(center_time, sig_cluster_height + 0.01, '*', ha='center', va='bottom', fontsize=25)

    # Customize plot
    text_color = s['text_color']

    if s['show_xlabel']:
        ax.set_xlabel(s['x_label'], fontsize=s['axis_font_size'], color=text_color)
    if s['show_ylabel']:
        ax.set_ylabel(s['y_label'], fontsize=s['axis_font_size'], color=text_color)
    
    ax.axhline(y=0, color='black', linestyle=':', alpha=0.5)
    ax.axvline(x=0, color='black', linestyle=':', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', colors=text_color, labelsize=s['tick_font_size'])
    
    if s['xticks'] is not None:
        ax.set_xticks(s['xticks'])
    if s['yticks'] is not None:
        ax.set_yticks(s['yticks'])
    
    if s['show_title']:
        title = s['title'] if s['title'] else f'{roi.upper()}'
        ax.set_title(title, fontsize=s['title_font_size'], fontweight='bold', color=text_color)
    
    if s['ylim']:
        ax.set_ylim(s['ylim'])
    if s['xlim']:
        ax.set_xlim(s['xlim'])
        
    if s['show_legend']:
        ax.legend(loc='best', framealpha=0.95, fontsize=s.get('legend_font_size', 10))
    
    plt.tight_layout()
    
    # Save if directory provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        error_type = 'std' if show_std else 'sem' if show_sem else 'ci' if show_ci else 'no_error'
        base = f'{roi}_{conditions_save_name}_{save_name_suffix}_{error_type}_shading'
        for ext in ('.pdf', '.png'):
            filepath = os.path.join(save_dir, base + ext)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved plot to: {filepath}")
    
    plt.close()
    return fig

def plot_power_traces_for_all_rois(evks_dict_elecs, rois, condition_names, conditions_save_name,
                                   plotting_parameters, window_size=None, sampling_rate=None, 
                                   significant_clusters=None, save_dir=None, error_type='std', 
                                   plot_style=None, save_name_suffix=None):
    """
    Plot power traces for each ROI comparing the specified conditions
    
    Parameters:
    -----------
    evks_dict_elecs : dict
        Evoked objects for electrodes
    rois : list
        List of ROI names
    condition_names : list
        List of condition names
    conditions_save_name : str
        Name to use for saving the plot
    plotting_parameters : dict
        Plotting parameters dictionary (see config/plotting_parameters.py for details)
    save_dir : str
        Directory to save plots
    error_type : str
        Type of error to show: 'std', 'sem', 'ci', or 'none'
    plot_style : dict
        Dictionary with plot style parameters for the figure settings.
    save_name_suffix : str
        Suffix to add to the save name
    significant_clusters : array-like of bool
        A boolean array indicating which time windows are part of a
        statistically significant cluster. Shape: (n_windows,).
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    for roi in rois:
        clusters_for_this_roi = None
        if significant_clusters is not None:
            # Look up the specific mask for this ROI
            clusters_for_this_roi = significant_clusters.get(roi, None)

        # Plot all electrodes
        plot_power_trace_for_roi(
            evks_dict_elecs, roi, condition_names, conditions_save_name, 
            plotting_parameters, window_size=window_size, sampling_rate=sampling_rate,
            significant_clusters=clusters_for_this_roi, save_dir=_subdir(save_dir, roi),
            show_std=(error_type == 'std'), 
            show_sem=(error_type == 'sem'),
            show_ci=(error_type == 'ci'),
            plot_style=plot_style, save_name_suffix=save_name_suffix
        )
    
    if save_dir:
        print(f"\nAll plots saved to: {save_dir}")
    plt.close()

def subtract_evoked_conditions(evks_dict, cond1, cond2, roi):
    """
    Subtract two evoked conditions (condition 2 from condition 1) for a given ROI.
    
    Parameters
    ----------
    evks_dict : dict
        Dictionary of evoked objects for each condition
    cond1 : str
        Name of the first condition
    cond2 : str
        Name of the second condition
    roi : str
        ROI name
    
    Returns
    -------
    evoked : mne.Evoked
        Evoked object for the subtracted condition
    """
    evoked_cond1 = evks_dict[cond1][roi]
    evoked_cond2 = evks_dict[cond2][roi]
    diff_evoked = mne.combine_evoked([evoked_cond1, evoked_cond2], weights=[1,-1])
    return diff_evoked

def create_subtracted_evokeds_dict(evks_dict, subtraction_pairs, rois):
    """
    Create a dictionary of subtracted evokeds for each ROI and subtraction pair.
    
    Parameters
    ----------
    evks_dict : dict
        Dictionary of evoked objects for each condition
    subtraction_pairs : list of tuples
        List of tuples containing pairs of conditions to subtract (cond 1 - cond 2)
    rois : list
        List of ROI names
    
    Returns
    -------
    subtracted_evokeds_dict : dict
        Nested dictionary: subtraction_pair -> roi -> multi-channel evoked object
    """
    subtracted_evokeds_dict = {}

    for pair in subtraction_pairs:
        pair_name = '-'.join(pair)
        subtracted_evokeds_dict[pair_name] = {}
        for roi in rois:
            subtracted_evokeds_dict[pair_name][roi] = subtract_evoked_conditions(evks_dict, pair[0], pair[1], roi)

    return subtracted_evokeds_dict

def time_perm_cluster_between_two_evokeds(evoked_cond1, evoked_cond2, p_thresh=0.05, 
                                       p_cluster=0.05, n_perm=1000, tails=1, 
                                       axis=0, stat_func=None, ignore_adjacency=None, 
                                       permutation_type='independent', vectorized=True, 
                                       n_jobs=-1, seed=None, verbose=None):
    """
    Finds significant clusters across time between two evoked objects
    
    https://ieeg-pipelines.readthedocs.io/en/latest/references/ieeg.calc.stats.time_perm_cluster.html
    """
    data1 = evoked_cond1.data
    data2 = evoked_cond2.data

    clusters, p_obs = time_perm_cluster(data1, data2,
                                    p_thresh=p_thresh,
                                    p_cluster=p_cluster,
                                    n_perm=n_perm,
                                    tails=tails,
                                    axis=axis,        
                                    stat_func=stat_func,  
                                    ignore_adjacency=ignore_adjacency,
                                    permutation_type=permutation_type,
                                    vectorized=vectorized,
                                    n_jobs=n_jobs,
                                    seed=seed,
                                    verbose=verbose)
    
    return clusters, p_obs
    

def process_windowed_data_for_anova(subjects_mne_objects, condition_names, rois, subjects, 
                                    electrodes_per_subject_roi, window_size=64, 
                                    step_size=16, sampling_rate=256):
    """
    Process data with sliding windows for ANOVA analysis.
    
    Parameters:
    -----------
    window_size : int or None
        Size of window in samples. If None, uses full epoch.
    step_size : int
        Step size for sliding window in samples.
    Slide a window over each subject's trial-level epoch data and average within
    each window to produce per-window per-channel scalars suitable for ANOVA.

    For every (condition, ROI, subject) triple that has at least one electrode in
    the ROI, this function:
      1. Picks the subject's epochs for the given condition.
      2. Restricts channels to ``electrodes_per_subject_roi[roi][subject]``.
      3. Slides a window of ``window_size`` samples with stride ``step_size`` over
         the time axis using ``general_utils.windower``.
      4. Averages within each window (collapsing the time axis inside a window)
         to produce a (n_trials, n_windows, n_channels) array.

    Subjects with **no** electrodes for the ROI are silently skipped (the output
    list for that ROI simply omits them). The downstream
    ``create_windowed_anova_dataframe`` reproduces this same skipping logic so
    list indices line up with the right subject + electrodes — see
    ``create_windowed_anova_dataframe`` for the matching invariant.

    Parameters
    ----------
    subjects_mne_objects : dict
        Nested dict ``[subject][condition_name][mne_object_type]``; we read
        ``[sub][cond]['HG_ev1_power_rescaled']``.
    condition_names : list of str
        Condition keys (e.g. ``['Stimulus_i25s25', ...]``).
    rois : list of str
        ROI names (e.g. ``['lpfc', 'occ', ...]``).
    subjects : list of str
        Subject IDs to iterate. Must be the same ordered list passed to
        ``create_windowed_anova_dataframe`` later, since that function relies on
        the same subject-skipping order.
    electrodes_per_subject_roi : dict
        ``[roi][subject] -> list of electrode names``.
    window_size : int or None, default 64
        Size of window in samples. If None, ``windower`` falls back to a single
        full-epoch window.
    step_size : int, default 16
        Stride between successive windows in samples.
    sampling_rate : int, default 256
        Currently unused (kept for API symmetry with the dataframe builder).

    Returns
    -------
    windowed_data : dict
        ``{condition_name: {roi: list_of_arrays}}`` where each list entry is a
        (n_trials, n_windows, n_channels) ndarray, one per subject that had
        electrodes for that ROI, in the same order as ``subjects``.
    """
    windowed_data = {}
    
    for condition_name in condition_names:
        windowed_data[condition_name] = {}
        
        for roi in rois:
            roi_data_list = []
            
            for sub in subjects:
                electrodes = electrodes_per_subject_roi[roi].get(sub, [])
                if not electrodes:
                    continue
                
                # Get epochs for this condition
                epochs = subjects_mne_objects[sub][condition_name]['HG_ev1_power_rescaled'].copy()
                epochs = epochs.pick_channels(electrodes)
                
                # Get data: (n_trials, n_channels, n_times)
                data = epochs.get_data()
                
                # Apply windowing to each channel and trial
                # windower expects data with time axis last by default
                windowed_trials = []
                for trial in data:
                    trial_windowed = windower(trial, window_size=window_size, 
                                             axis=-1, step_size=step_size, insert_at=0)
                    # Shape: (n_windows, n_channels, window_size)
                    windowed_trials.append(trial_windowed)
                
                windowed_trials = np.array(windowed_trials)
                # Shape: (n_trials, n_windows, n_channels, window_size)
                
                # Average within each window
                windowed_avg = np.mean(windowed_trials, axis=-1)
                # Shape: (n_trials, n_windows, n_channels)
                
                roi_data_list.append(windowed_avg)
            
            windowed_data[condition_name][roi] = roi_data_list
    
    return windowed_data

def create_windowed_anova_dataframe(windowed_data, conditions, rois, subjects,
                                    electrodes_per_subject_roi, times,
                                    window_size=None, step_size=1, sampling_rate=256):
    """
    Create DataFrame for windowed ANOVA analysis.
    Flatten the nested ``windowed_data`` structure into a long DataFrame suitable
    for per-window OLS / ANOVA fits.

    Each row is one observation: ``(subject, electrode, ROI, trial, window) ->
    Activity``, with the factor columns from ``conditions[cond]`` attached
    (e.g. ``congruency``, ``incongruentProportion``, ``switchType``,
    ``switchProportion``). ``BIDS_events`` is included verbatim; downstream
    formula-builders should drop it.

    Window centers are precomputed once (in seconds) and attached to every row
    of the corresponding window so the resulting DataFrame can be plotted /
    grouped by ``WindowCenter`` directly.

    Parameters
    ----------
    windowed_data : dict
        Output of :func:`process_windowed_data_for_anova`.
    conditions : dict
        Mapping ``{condition_name: condition_parameters}`` from
        ``experiment_conditions``. The ``condition_parameters`` dict is merged
        into every row so factor columns become available for OLS.
    rois : list of str
    electrodes_per_subject_roi : dict
        ``[roi][subject] -> list of electrode names``.
    times : array-like
        The full per-sample time vector of the epoch (used to compute window
        centers).
    window_size : int or None
        Window length in samples; when None, a single window centered at the
        epoch midpoint is produced.
    step_size : int, default 1
    sampling_rate : int, default 256

    Returns
    -------
    df : pandas.DataFrame
        Columns: ``SubjectID``, ``Electrode``, ``ROI``, ``WindowCenter``,
        ``WindowIndex``, ``Trial``, ``Activity``, plus all keys from
        ``condition_parameters`` (factor columns + ``BIDS_events``).

    Notes
    -----
    Cross-subject channel-name collisions: ``combine_single_channel_evokeds``
    renames duplicates with running suffixes (``LFMM9-0``, ``LFMM9-1``), but the
    ``Electrode`` column here stores the **un-suffixed** original name. Always
    group by ``(SubjectID, Electrode)`` together if you need a unique key.
    """
    data_for_anova = []

    # Window centers
    if window_size is not None:
        n_windows = (len(times) - window_size) // step_size + 1
        window_centers = []
        for i in range(n_windows):
            start_idx = i * step_size
            center_idx = start_idx + window_size // 2
            if center_idx < len(times):
                window_centers.append(times[center_idx])
            else:
                window_centers.append(times[-1])
    else:
        window_centers = [np.mean(times)]

    for condition_name, condition_parameters in conditions.items():
        for roi in rois:
            roi_list = windowed_data.get(condition_name, {}).get(roi, [])
            sub_idx = 0
            for sub in subjects:
                electrodes = electrodes_per_subject_roi[roi].get(sub, [])
                if not electrodes:
                    continue  # process_windowed_data_for_anova also skipped
                if sub_idx >= len(roi_list):
                    break
                subject_data = roi_list[sub_idx]
                sub_idx += 1

                # Defensive: clamp electrode count to actual data shape.
                n_chans_data = subject_data.shape[2]
                n_chans = min(len(electrodes), n_chans_data)
                if n_chans_data != len(electrodes):
                    print(f"[create_windowed_anova_dataframe] WARNING: "
                          f"{sub}/{roi}: electrode list has {len(electrodes)} but "
                          f"data has {n_chans_data} channels; using first {n_chans}.")

                for trial_idx in range(subject_data.shape[0]):
                    for window_idx in range(subject_data.shape[1]):
                        for electrode_idx in range(n_chans):
                            electrode_name = electrodes[electrode_idx]
                            activity = subject_data[trial_idx, window_idx, electrode_idx]
                            data_dict = {
                                'SubjectID': sub,
                                'Electrode': electrode_name,
                                'ROI': roi,
                                'WindowCenter': window_centers[window_idx],
                                'WindowIndex': window_idx,
                                'Trial': trial_idx + 1,
                                'Activity': activity,
                            }
                            data_dict.update(condition_parameters)
                            data_for_anova.append(data_dict)

    return pd.DataFrame(data_for_anova)

def perform_windowed_anova(df, conditions, rois, save_dir, save_name, 
                           anova_type='within_electrode'):
    """
    Fit a Type II OLS ANOVA at every time window and persist the results.

    The OLS formula is built dynamically from the keys of any single condition
    in ``conditions`` (excluding ``BIDS_events``), as
    ``Activity ~ C(f1) * C(f2) * ... * C(fn)`` — i.e. all factors plus all
    interactions. This implicitly assumes that every condition shares the same
    factor-key set; mixing condition sets with different keys here will produce
    misleading formulas.

    Two analysis modes:

    - ``'within_electrode'``: per (subject, electrode, ROI), fit OLS on the
      trial-level rows for that electrode in that window. Stores only the
      effects with uncorrected p < 0.05.
    - ``'across_electrode'``: per ROI, average activity within each
      (subject, electrode, factors) cell first, then fit OLS treating each
      electrode-cell as one observation. Stores the full ANOVA table per ROI
      per window.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of :func:`create_windowed_anova_dataframe`.
    conditions : dict
        ``{condition_name: condition_parameters}``; only the first entry's keys
        are used to build the formula.
    rois : list of str
        Used only in the ``'across_electrode'`` branch.
    save_dir : str
        Directory to write the JSON output file.
    save_name : str
        Stem for the output filename
        (``{save_name}_windowed_anova_{anova_type}.json``).
    anova_type : {'within_electrode', 'across_electrode'}, default 'within_electrode'

    Returns
    -------
    results_by_window : dict
        - within_electrode: ``{window_center -> list of {SubjectID, Electrode,
          ROI, Effects (DataFrame of significant effects)}}``
        - across_electrode: ``{window_center -> {roi: anova_lm_table}}``

    Notes
    -----
    BUG (across_electrode branch): the inner loop overwrites
    ``results_by_window[window_center]`` for each ROI, so only the last ROI's
    table is preserved per window. To persist all ROIs, that branch should
    accumulate into ``{window_center: {roi: ...}}`` instead of reassigning.

    Output JSON serializes via ``str(result)``, which is lossy and only useful
    for human inspection — re-parsing the JSON back into DataFrames is not
    supported.
    """
    results_by_window = {}
    
    # Get unique window indices
    window_indices = df['WindowIndex'].unique()
    
    for window_idx in window_indices:
        df_window = df[df['WindowIndex'] == window_idx]
        window_center = df_window['WindowCenter'].iloc[0]
        
        if anova_type == 'within_electrode':
            # Perform within-electrode ANOVA for this window
            results = []
            
            for subject_id in df_window['SubjectID'].unique():
                for electrode in df_window['Electrode'].unique():
                    for roi in df_window['ROI'].unique():
                        df_filtered = df_window[
                            (df_window['SubjectID'] == subject_id) & 
                            (df_window['Electrode'] == electrode) & 
                            (df_window['ROI'] == roi)
                        ]
                        
                        if df_filtered.empty or len(df_filtered) < 2:
                            continue
                        
                        # Build formula
                        condition_keys = [k for k in conditions[next(iter(conditions))].keys() 
                                        if k != 'BIDS_events']
                        formula = 'Activity ~ ' + ' * '.join([f'C({k})' for k in condition_keys])
                        
                        try:
                            model = ols(formula, data=df_filtered).fit()
                            anova_results = anova_lm(model, typ=2)
                            
                            # Store significant effects
                            sig_effects = anova_results[anova_results['PR(>F)'] < 0.05]
                            if not sig_effects.empty:
                                results.append({
                                    'SubjectID': subject_id,
                                    'Electrode': electrode,
                                    'ROI': roi,
                                    'Effects': sig_effects
                                })
                        except:
                            continue
            
            results_by_window[window_center] = results
            
        elif anova_type == 'across_electrode':
            # Perform across-electrode ANOVA for this window.
            # Accumulate per-ROI results under this window so multiple ROIs
            # don't overwrite each other.
            if window_center not in results_by_window:
                results_by_window[window_center] = {}

            # Build the factor-key list once per window — exclude BIDS_events
            # because (a) it isn't an ANOVA factor and (b) its values are lists,
            # which are unhashable and would break the groupby below.
            condition_keys = [k for k in conditions[next(iter(conditions))].keys()
                              if k != 'BIDS_events']
            formula = 'Activity ~ ' + ' * '.join([f'C({k})' for k in condition_keys])

            for roi in rois:
                df_roi = df_window[df_window['ROI'] == roi]
                if df_roi.empty:
                    continue

                # Average across trials for each electrode-cell
                df_averaged = df_roi.groupby(
                    ['SubjectID', 'Electrode', 'ROI'] + condition_keys
                )['Activity'].mean().reset_index()

                model = ols(formula, data=df_averaged).fit()
                anova_results = anova_lm(model, typ=2)

                results_by_window[window_center][roi] = anova_results
    
    # Save results - TODO: this only works for across_electrode right now, need to handle within_electrode too. Check the way I used to store these results for plotting.
    rows = []
    
    for window_center, roi_dict in results_by_window.items():
        for roi, anova_table in roi_dict.items():
            df_out = anova_table.reset_index().rename(columns={'index': 'term'})
            df_out['window_center'] = window_center
            df_out['roi'] = roi
            rows.append(df_out)
            
    pd.concat(rows, ignore_index=True).to_csv(
        os.path.join(save_dir, f'{save_name}_windowed_anova_{anova_type}.csv'), 
        index=False
    )

    return results_by_window

def apply_fdr_correction_to_windowed_results(results_by_window, alpha=0.05):
    """
    Apply FDR correction across all windows and effects.
    """
    # Collect all p-values
    all_pvalues = []
    pvalue_info = []  # Track where each p-value comes from
    
    for window, results in results_by_window.items():
        if isinstance(results, list):  # within-electrode results
            for result in results:
                effects_df = result.get('Effects', pd.DataFrame())
                if not effects_df.empty:
                    for idx, row in effects_df.iterrows():
                        all_pvalues.append(row['PR(>F)'])
                        pvalue_info.append({
                            'window': window,
                            'subject': result['SubjectID'],
                            'electrode': result['Electrode'],
                            'effect': idx
                        })
        else:  # across-electrode results
            for roi, anova_table in results.items():
                for idx, row in anova_table.iterrows():
                    all_pvalues.append(row['PR(>F)'])
                    pvalue_info.append({
                        'window': window,
                        'roi': roi,
                        'effect': idx
                    })
    
    # Apply FDR correction
    if all_pvalues:
        rejected, corrected_pvalues, _, _ = multipletests(
            all_pvalues, alpha=alpha, method='fdr_bh'
        )
        
        # Create corrected results structure
        corrected_results = {}
        for i, info in enumerate(pvalue_info):
            window = info['window']
            if window not in corrected_results:
                corrected_results[window] = []
            
            if rejected[i]:  # Only keep significant after correction
                info['corrected_pvalue'] = corrected_pvalues[i]
                info['original_pvalue'] = all_pvalues[i]
                corrected_results[window].append(info)
    
    return corrected_results

def apply_plot_style(ax, roi, style=None):
    """Apply styling to an axis from a style dict, with defaults."""
    s = {**DEFAULT_PLOT_STYLE, **(style or {})}
    
    if s['show_title']:
        title = s['title'] if s['title'] else f"{roi.upper()}"
        ax.set_title(title, fontsize=s['title_font_size'], 
                     fontweight='bold', color=s['text_color'])
    
    if s['show_xlabel']:
        ax.set_xlabel(s['x_label'], fontsize=s['axis_font_size'], color=s['text_color'])
    else:
        ax.set_xlabel('')
    
    if s['show_ylabel']:
        ax.set_ylabel(s['y_label'], fontsize=s['axis_font_size'], color=s['text_color'])
    else:
        ax.set_ylabel('')
    
    # Ticks
    if s['xticks'] is not None:
        ax.set_xticks(s['xticks'])
    if s['yticks'] is not None:
        ax.set_yticks(s['yticks'])
    if s['xtick_labels'] is not None:
        ax.set_xticklabels(s['xtick_labels'])
    if s['ytick_labels'] is not None:
        ax.set_yticklabels(s['ytick_labels'])
    
    ax.tick_params(axis='both', colors=s['text_color'], labelsize=s['tick_font_size'])
    
    if s['xlim']:
        ax.set_xlim(s['xlim'])
    if s['ylim']:
        ax.set_ylim(s['ylim'])
    
    if s['show_legend']:
        ax.legend(loc='best', framealpha=0.95, fontsize=s['legend_font_size'])
    
    # Standard cleanup
    ax.axhline(y=0, color='black', linestyle=':', alpha=0.5)
    ax.axvline(x=0, color='black', linestyle=':', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
# =============================================================================
# Two-way interaction cluster correction across time
#
# We compute, per electrode per timepoint, the 2x2 interaction contrast
#   ((A1B1 - A1B2) - (A2B1 - A2B2))
# averaging across whatever other factors exist in the conditions_obj. We then
# use a one-sample sign-flip cluster permutation test across electrodes
# (mne.stats.permutation_cluster_1samp_test). This is mathematically equivalent
# to an F-test on the interaction term in a 2x2 ANOVA across electrodes
# (t**2 == F when df_num == 1), so the resulting cluster mask is equivalent to a
# F-statistic cluster correction.
# =============================================================================

def _get_subcell_condition_names(conditions_obj, factor1, factor2, level1, level2):
    """Find condition keys whose factor values match (level1, level2)."""
    return [
        k for k, v in conditions_obj.items()
        if v.get(factor1) == level1 and v.get(factor2) == level2
    ]

def _get_factor_levels(conditions_obj, factor):
    """Return the unique levels of 'factor' across the conditions, in stable order."""
    seen = []
    for v in conditions_obj.values():
        lvl = v.get(factor)
        if lvl is not None and lvl not in seen:
            seen.append(lvl)
    return seen

def compute_subcell_evoked_data(evks_dict, conditions_obj, factor1, factor2,
                                level1, level2, roi):
    """Average per-electrode evoked data across all condition keys that match (factor1=level1, factor2=level2).
    Other factors are collapsed by simmple average over the matching cells (equal weight per subcell).
    
    Returns
    -------
    arr : (n_electrodes, n_times) ndarray, or None if no matching condition has data
    """
    cond_keys = _get_subcell_condition_names(conditions_obj, factor1, factor2,
                                             level1, level2)
    arrays = []
    for k in cond_keys:
        evk = evks_dict.get(k, {}).get(roi)
        if evk is None or evk.data.shape[0] == 0:
            continue
        arrays.append(evk.data)
    if not arrays:
        return None
    return np.mean(np.stack(arrays, axis=0), axis=0)
    
# =============================================================================
# Plotting helpers
# =============================================================================

def _find_cluster_spans(mask):
    """Return list of (start_idx, end_idx) inclusive for contiguous True runs.""" # hmm pretty sure some version of this already exists somewhere
    spans = []
    in_run = False
    start = 0
    arr = np.asarray(mask).astype(bool)
    for i, val in enumerate(arr):
        if val and not in_run:
            start = i
            in_run = True
        elif not val and in_run:
            spans.append((start, i - 1))
            in_run = False
    if in_run:
        spans.append((start, len(arr) - 1))
    return spans

def _draw_cluster_bar(ax, times, mask, y, color='black', linewidth=6,
                      label=None, label_x=None, label_color=None,
                      label_fontsize=10):
    """Draw horizontal bar(s) wherever mask is True, at height y.""" # this code might be redundant, check plot_horizontal_bar from aaron or i'm sure i have other code that does this.
    spans = _find_cluster_spans(mask)
    for start_idx, end_idx in spans:
        ax.hlines(y=y, xmin=times[start_idx], xmax=times[end_idx],
                  color=color, linewidth=linewidth)
    if label is not None and label_x is not None:
        ax.text(label_x, y, label, ha='right', va='center',
                fontsize=label_fontsize,
                color=label_color if label_color is not None else color)

# --- Factor-pair → plotting_parameters key resolver ---
# The 4 levels of each 2-way factor pair map directly to a Stimulus_*** entry
# (e.g. (congruency='c', incongruentProportion='25%') -> 'Stimulus_c25').
# Each pair-resolver takes the factor *value* dict for a condition and returns
# the plotting_parameters key, or None if any factor is missing/unknown.

def _strip_pct(level):
    """Normalize '25%' / 25 / '25' -> '25' or '75', else None."""
    if level is None:
        return None
    s = str(level).rstrip('%')
    return s if s in {'25', '75'} else None

def _pp_key_incongruent_proportion(v):
    """(congruency, incongruentProportion) -> 'Stimulus_c25' / 'Stimulus_i75' / ...

    Now consistent: the stored incongruentProportion matches the suffix used
    in Stimulus_*** keys (both follow BIDS naming).
    """
    cong = v.get('congruency')
    prop = _strip_pct(v.get('incongruentProportion'))   # <-- no flip
    if cong in ('c', 'i') and prop in ('25', '75'):
        return f'Stimulus_{cong}{prop}'
    return None
def _pp_key_switch_proportion(v):
    """(switchType, switchProportion) -> 'Stimulus_s25' / 'Stimulus_r75' / ...

    switchProportion is NOT flipped -- both the stored value and the BIDS
    naming use the switch proportion directly.
    """
    st = v.get('switchType')
    prop = _strip_pct(v.get('switchProportion'))             # <-- no flip
    if st in ('s', 'r') and prop in ('25', '75'):
        return f'Stimulus_{st}{prop}'
    return None

def _pp_style(key, fallback=('gray', '-')):
    """Return (color, line_style) from plotting_parameters[key], with fallback."""
    if key is None or key not in plotting_parameters:
        return fallback
    p = plotting_parameters[key]
    return p.get('color', fallback[0]), p.get('line_style', fallback[1])

def _generate_16_condition_colors(condition_names, conditions_obj,
                                  factors=('congruency', 'incongruentProportion',
                                           'switchType', 'switchProportion')):
    """Style map for the 16-condition plot.

    Strategy: read directly from plotting_parameters[condition_name] when an
    entry exists. Otherwise compose a fallback style by looking up the
    (congruency, incongruentProportion) entry for color and the
    (switchType, switchProportion) entry for linestyle.
    """
    style_map = {}
    for name in condition_names:
        # Preferred path: direct entry exists in plotting_parameters.
        if name in plotting_parameters:
            p = plotting_parameters[name]
            style_map[name] = {
                'color':     p.get('color', 'gray'),
                'linestyle': p.get('line_style', '-'),
                'linewidth': p.get('linewidth', 1.8),
                'alpha':     p.get('alpha', 1.0),
            }
            continue

        # Fallback: compose from the two 2-way pair entries.
        v = conditions_obj.get(name, {})
        cong_key   = _pp_key_incongruent_proportion(v)
        switch_key = _pp_key_switch_proportion(v)
        color, _   = _pp_style(cong_key)
        _, ls      = _pp_style(switch_key)
        style_map[name] = {
            'color': color, 'linestyle': ls,
            'linewidth': 1.8, 'alpha': 1.0,
        }
    return style_map

def plot_2way_interaction_for_roi(
    evks_dict, roi, conditions_obj, factor1, factor2,
    interaction_mask, conditions_save_name, plotting_parameters=None,
    plot_style=None, save_dir=None, save_name_suffix=None, error_type='sem',
    interaction_label=None, factor_labels=None,
):
    """Plot the 4 sub-cell traces for a 2x2 interaction with a cluster bar overlay.

    Parameters
    ----------
    interaction_mask : (n_times,) bool array, or None
    """
    s = {**DEFAULT_PLOT_STYLE, **(plot_style or {})}
    fig, ax = plt.subplots(figsize=s['figsize'])

    # Figure out a time vector from any populated condition
    times = None
    for cname, roi_dict in evks_dict.items():
        evk = roi_dict.get(roi)
        if evk is not None and evk.data.shape[0] > 0:
            times = evk.times
            break
    if times is None:
        plt.close(fig)
        return None

    levels1 = _get_factor_levels(conditions_obj, factor1)
    levels2 = _get_factor_levels(conditions_obj, factor2)
    if len(levels1) != 2 or len(levels2) != 2:
        plt.close(fig)
        raise ValueError(
            f"Two-way plot expects 2 levels per factor; got "
            f"{factor1}={levels1}, {factor2}={levels2}"
        )

    # Resolve PP key per cell based on which factor pair this 2-way is.
    # If we recognize the pair, use direct PP lookups; otherwise fall back to a generic pair.
    pair = frozenset((factor1, factor2))
    if pair == frozenset(('congruency', 'incongruentProportion')):
        resolve = _pp_key_incongruent_proportion
    elif pair == frozenset(('switchType', 'switchProportion')):
        resolve = _pp_key_switch_proportion
    else:
        resolve = None

    for l1 in levels1:
        for l2 in levels2:
            data = compute_subcell_evoked_data(
                evks_dict, conditions_obj, factor1, factor2, l1, l2, roi
            )
            if data is None:
                continue
            mean_data = np.mean(data, axis=0)
            if resolve is not None:
                key = resolve({factor1: l1, factor2: l2})
                color, ls = _pp_style(key)
            else:
                # Cross-factor 2-way (e.g. congruency x switchProportion) -- no PP entry.
                color = ('#1f77b4' if l1 == levels1[0] else '#d62728')
                ls = '-' if l2 == levels2[0] else '--'
            label = f"{factor1}={l1}, {factor2}={l2}"
            ax.plot(times, mean_data, color=color, linestyle=ls,
                    linewidth=2.5, label=label)
            # ... (SEM shading block unchanged)
            n = data.shape[0]
            if error_type == 'sem' and n > 1:
                err = np.std(data, axis=0) / np.sqrt(n)
            elif error_type == 'std':
                err = np.std(data, axis=0)
            else:
                err = None
            if err is not None:
                ax.fill_between(times, mean_data - err, mean_data + err,
                                color=color, alpha=0.15 * alpha, linewidth=0)

    # Overlay cluster bar at top of plot
    if interaction_mask is not None and np.any(interaction_mask):
        ylim = ax.get_ylim() if s.get('ylim') is None else s['ylim']
        bar_y = s.get('sig_cluster_height', ylim[1] - (ylim[1] - ylim[0]) * 0.05)
        _draw_cluster_bar(ax, times, interaction_mask, y=bar_y,
                          color='black', linewidth=8)
        center_idx = np.where(interaction_mask)[0]
        if len(center_idx):
            ax.text(times[int(np.median(center_idx))], bar_y + 0.01, '*',
                    ha='center', va='bottom', fontsize=20)

    apply_plot_style(ax, roi, plot_style)
    # Always show the 4-trace legend on per-interaction plots — the 4 cells are
    # otherwise indistinguishable. This overrides plot_style['show_legend']
    # because for these plots the legend is essential, not optional.
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc='lower right', framealpha=0.9,
                  fontsize=s.get('legend_font_size', 10))
    if interaction_label and s.get('show_title', True):
        ax.set_title(f"{roi.upper()} — {interaction_label}",
                     fontsize=s['title_font_size'], fontweight='bold',
                     color=s['text_color'])
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        suf = save_name_suffix or ''
        base = (f'{roi}_{conditions_save_name}_2way_{factor1}_x_{factor2}_'
                f'{suf}_{error_type}_shading')
        for ext in ('.pdf', '.png'):
            filepath = os.path.join(save_dir, base + ext)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved plot to: {filepath}")
    plt.close(fig)
    return fig

def plot_16_conditions_with_interaction_clusters_for_roi(
    evks_dict, roi, conditions_obj, condition_names, conditions_save_name,
    interaction_results,  # dict[interaction_name] -> result dict
    anova_interactions,
    plot_style=None, save_dir=None, save_name_suffix=None, error_type='sem',
):
    """Plot all 16 condition power traces with 4 stacked horizontal cluster bars
    (one per 2-way interaction) overlaid at the top of the panel.
    """
    s = {**DEFAULT_PLOT_STYLE, **(plot_style or {})}
    fig, ax = plt.subplots(figsize=s['figsize'])

    times = None
    colors = _generate_16_condition_colors(condition_names, conditions_obj)

    for cname in condition_names:
        evk = evks_dict.get(cname, {}).get(roi)
        if evk is None or evk.data.shape[0] == 0:
            continue
        if times is None:
            times = evk.times
        data = evk.data
        mean_data = np.mean(data, axis=0)
        n = data.shape[0]
        color = colors.get(cname, 'black')
        ax.plot(times, mean_data, color=color, linewidth=1.5,
                label=cname.replace('Stimulus_', ''))
        if error_type == 'sem' and n > 1:
            err = np.std(data, axis=0) / np.sqrt(n)
            ax.fill_between(times, mean_data - err, mean_data + err,
                            color=color, alpha=0.10, linewidth=0)

    if times is None:
        plt.close(fig)
        return None

    # Stack 4 cluster bars at the top of the panel.
    ylim = s.get('ylim') if s.get('ylim') is not None else ax.get_ylim()
    y_top = ylim[1]
    y_bottom = ylim[0]
    span = y_top - y_bottom
    # Reserve top 16% of axis for 4 cluster bars + labels
    bar_band_top = y_top - span * 0.02
    bar_band_bottom = y_top - span * 0.16
    n_bars = max(len(anova_interactions), 1)
    bar_ys = np.linspace(bar_band_top, bar_band_bottom, n_bars)
    # Distinct, perceptually-separated colors for the 4 cluster bars
    # (ColorBrewer Set1, dropping yellow/orange which collide with the
    # 16-condition trace palette).
    bar_palette = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3',
                   '#ff7f00', '#a65628']
    bar_colors = [bar_palette[i % len(bar_palette)] for i in range(n_bars)]

    res_for_roi = interaction_results.get(roi, {})
    bar_legend_handles = []
    bar_legend_labels = []
    for i, inter in enumerate(anova_interactions):
        name = inter['name']
        info = res_for_roi.get(name)
        if info is None:
            continue
        mask = info['mask']
        bar_color = bar_colors[i % len(bar_colors)]
        # No per-bar text label — the legend below handles labeling instead.
        _draw_cluster_bar(ax, times, mask, y=bar_ys[i],
                          color=bar_color, linewidth=7,
                          label=None, label_x=None)
        bar_legend_handles.append(
            Line2D([0], [0], color=bar_color, linewidth=7)
        )
        bar_legend_labels.append(info.get('label', name))

    apply_plot_style(ax, roi, plot_style)
    # Force ylim to leave room for bars if not user-set
    if s.get('ylim') is None:
        ax.set_ylim(y_bottom, y_top)

    # Two legends: the 16-condition trace legend (gated on plot_style.show_legend
    # because it's huge and often obscures the data) and the cluster-bar legend
    # (always shown — the bars are uninterpretable without it).
    trace_legend = None
    if s.get('show_legend', True):
        trace_legend = ax.legend(loc='lower left', framealpha=0.9, ncol=2,
                                 fontsize=max(6, s.get('legend_font_size', 10) - 2))
    if bar_legend_handles:
        if trace_legend is not None:
            ax.add_artist(trace_legend)  # keep both legends visible
        ax.legend(bar_legend_handles, bar_legend_labels,
                  loc='lower right', framealpha=0.9,
                  title='2-way interaction clusters',
                  fontsize=s.get('legend_font_size', 10))

    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        suf = save_name_suffix or ''
        base = (f'{roi}_{conditions_save_name}_16cond_with_interaction_clusters_'
                f'{suf}_{error_type}_shading')
        for ext in ('.pdf', '.png'):
            filepath = os.path.join(save_dir, base + ext)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved plot to: {filepath}")
    plt.close(fig)
    return fig


def plot_anova_interaction_results(
    evks_dict, conditions_obj, condition_names, conditions_save_name,
    rois, anova_interactions, interaction_results,
    plot_style=None, save_dir=None, save_name_suffix=None, error_type='sem',
):
    """Convenience wrapper: for each ROI, draw the 16-condition mega-plot AND
    one 4-trace plot per 2-way interaction.
    """
    for roi in rois:   
        plot_16_conditions_with_interaction_clusters_for_roi(
            evks_dict, roi, conditions_obj, condition_names, conditions_save_name,
            interaction_results, anova_interactions,
            plot_style=plot_style, save_dir=_subdir(save_dir, roi),
            save_name_suffix=save_name_suffix, error_type=error_type,
        )
        for inter in anova_interactions:
            f1, f2 = inter['factors']
            info = interaction_results.get(roi, {}).get(inter['name'])
            mask = info['mask'] if info is not None else None
            plot_2way_interaction_for_roi(
                evks_dict, roi, conditions_obj, f1, f2, mask,
                conditions_save_name,
                plot_style=plot_style, save_dir=_subdir(save_dir, roi),
                save_name_suffix=save_name_suffix, error_type=error_type,
                interaction_label=inter.get('label'),
            )
            
# =============================================================================
# Full-ANOVA cluster correction (uses windowed OLS, all 16 effects in one fit)
#
# This is the trial-imbalance-robust version: per window, fit
#   Activity ~ C * CP * S * SP
# extract F per effect, build a permutation null by shuffling factor labels
# within electrode, then cluster-correct each effect's F-trace using
# find_significant_clusters_of_series_vs_distribution_based_on_percentile.
# =============================================================================

def _fit_anova_one_window(df_window, formula, factor_columns):
    """Fit OLS at a single window, return dict[effect_name] -> F-stat.

    `df_window` already aggregated to one row per (electrode × cell).
    Effect names match those returned by anova_lm (e.g., "C(congruency)",
    "C(congruency):C(incongruentProportion)", ...).
    """
    try:
        model = ols(formula, data=df_window).fit()
        table = anova_lm(model, typ=2)
    except Exception:
        return None
    return {idx: row['F'] for idx, row in table.iterrows() if idx != 'Residual'}


def _shuffle_labels_within_electrode(df_one_window, factor_columns, rng):
    """Return a copy of df_one_window with factor columns permuted within each
    electrode (so each electrode keeps its 16 cell means but their factor
    assignment is randomized as a block)."""
    df = df_one_window.copy()
    for (sub, elec), idxs in df.groupby(['SubjectID', 'Electrode']).groups.items():
        idxs = np.asarray(idxs)
        perm = rng.permutation(len(idxs))
        for col in factor_columns:
            df.loc[idxs, col] = df.loc[idxs[perm], col].values
    return df


def run_windowed_anova_cluster_correction(
    windowed_data, conditions_obj, anova_factors, rois, subjects,
    electrodes_per_subject_roi, times, window_size, step_size, sampling_rate,
    n_perm=1000, percentile=95, cluster_percentile=95,
    seed=42, n_jobs=-1, verbose=True,
):
    """Windowed full-ANOVA + cluster correction.

    Parameters
    ----------
    windowed_data : dict
        Output of process_windowed_data_for_anova (condition_name -> roi -> list of
        (n_trials, n_windows, n_channels) arrays per subject).
    conditions_obj : dict
        The conditions_obj from the registry (condition_name -> dict of factor values).
    anova_factors : list of str
        Factor column names (e.g. ['congruency', 'incongruentProportion',
        'switchType', 'switchProportion']).
    n_perm : int
    percentile : float
        Pointwise null percentile for cluster formation (e.g., 95 → uncorrected p=0.05).
    cluster_percentile : float
        Cluster-mass percentile for cluster correction (e.g., 95 → corrected p=0.05).

    Returns
    -------
    results : dict[roi][effect_name] -> {
        'observed_F': (n_windows,) array,
        'null_F': (n_perm, n_windows) array,
        'sig_clusters_windows': list of (start_window_idx, end_window_idx),
        'window_mask': (n_windows,) bool array,
        'sample_mask': (n_times,) bool array,
    }
    window_centers : (n_windows,) array of timepoints
    """
    rng_master = np.random.RandomState(seed)

    # Build long dataframe (re-uses your existing function)
    df = create_windowed_anova_dataframe(
        windowed_data, conditions_obj, rois, subjects,
        electrodes_per_subject_roi,
        times=times, window_size=window_size, step_size=step_size,
        sampling_rate=sampling_rate,
    )

    # OLS formula over the requested factors
    formula = 'Activity ~ ' + ' * '.join([f'C({f})' for f in anova_factors])
    if verbose:
        print(f"[anova-cluster] Formula: {formula}")

    # Pre-compute window centers (n_windows,) and the corresponding sample mapping
    window_indices = sorted(df['WindowIndex'].unique())
    n_windows = len(window_indices)
    window_centers = np.array(
        [df[df['WindowIndex'] == w]['WindowCenter'].iloc[0] for w in window_indices]
    )

    # Map each window to a (start_sample, end_sample) range, used to expand
    # window-level clusters back to the full sampling-rate mask for plotting.
    n_times = len(times)
    win_to_samples = []
    for w in window_indices:
        start_sample = int(w * step_size)
        end_sample = min(start_sample + window_size - 1, n_times - 1)
        win_to_samples.append((start_sample, end_sample))

    results = {}

    for roi in rois:
        if verbose:
            print(f"[anova-cluster] === ROI: {roi} ===")
        df_roi = df[df['ROI'] == roi]
        if df_roi.empty:
            continue

        # Aggregate to electrode × window × cell (across-electrode ANOVA)
        group_cols = ['SubjectID', 'Electrode', 'WindowIndex', 'WindowCenter'] + anova_factors
        df_agg = df_roi.groupby(group_cols, as_index=False)['Activity'].mean()

        # === Observed F per effect per window ===
        observed_per_window = {}  # window_idx -> dict(effect -> F)
        for w in window_indices:
            df_w = df_agg[df_agg['WindowIndex'] == w]
            f_dict = _fit_anova_one_window(df_w, formula, anova_factors)
            observed_per_window[w] = f_dict

        # All effects encountered (use first non-None window)
        effect_names = None
        for w in window_indices:
            if observed_per_window[w] is not None:
                effect_names = list(observed_per_window[w].keys())
                break
        if effect_names is None:
            print(f"[anova-cluster] No usable windows for ROI {roi}; skipping.")
            continue

        observed_F = np.full((len(effect_names), n_windows), np.nan)
        for wi, w in enumerate(window_indices):
            f_dict = observed_per_window[w]
            if f_dict is None:
                continue
            for ei, eff in enumerate(effect_names):
                observed_F[ei, wi] = f_dict.get(eff, np.nan)

        # === Permutation null ===
        # We shuffle factor labels per electrode once per perm (same shuffle for all windows),
        # since each electrode contributes independent rows per window.
        # Per-perm work: re-fit OLS at every window.
        seeds = rng_master.randint(0, 2**31 - 1, size=n_perm)

        def _one_perm(perm_seed):
            rng = np.random.RandomState(perm_seed)
            # Build a single shuffle map per (subject, electrode) that we apply at every window
            shuffle_map = {}
            for (sub, elec), grp in df_agg[df_agg['WindowIndex'] == window_indices[0]] \
                    .groupby(['SubjectID', 'Electrode']):
                n_cells = len(grp)
                shuffle_map[(sub, elec)] = rng.permutation(n_cells)

            # Apply the same shuffle to every window
            null_F_perm = np.full((len(effect_names), n_windows), np.nan)
            for wi, w in enumerate(window_indices):
                df_w = df_agg[df_agg['WindowIndex'] == w].copy().reset_index(drop=True)
                # For each (sub, elec), permute the factor columns in place using shuffle_map
                for (sub, elec), idxs in df_w.groupby(['SubjectID', 'Electrode']).groups.items():
                    perm = shuffle_map.get((sub, elec))
                    if perm is None or len(perm) != len(idxs):
                        continue
                    idxs = np.asarray(idxs)
                    for col in anova_factors:
                        df_w.loc[idxs, col] = df_w.loc[idxs[perm], col].values
                f_dict = _fit_anova_one_window(df_w, formula, anova_factors)
                if f_dict is None:
                    continue
                for ei, eff in enumerate(effect_names):
                    null_F_perm[ei, wi] = f_dict.get(eff, np.nan)
            return null_F_perm

        if verbose:
            print(f"[anova-cluster]   running {n_perm} permutations across {n_windows} windows "
                  f"({len(effect_names)} effects)")
        null_F_list = Parallel(n_jobs=n_jobs, verbose=5 if verbose else 0)(
            delayed(_one_perm)(s) for s in seeds
        )
        null_F = np.stack(null_F_list, axis=0)  # (n_perm, n_effects, n_windows)

        # === Cluster correction per effect ===
        results[roi] = {}
        for ei, eff in enumerate(effect_names):
            obs = observed_F[ei]                  # (n_windows,)
            null = null_F[:, ei, :]               # (n_perm, n_windows)

            # Replace NaNs (failed fits) with zeros so they never form clusters
            obs_clean = np.nan_to_num(obs, nan=0.0)
            null_clean = np.nan_to_num(null, nan=0.0)

            sig_clusters_windows = find_significant_clusters_of_series_vs_distribution_based_on_percentile(
                series=obs_clean,
                distribution=null_clean,
                time_points=window_centers,
                percentile=percentile,
                cluster_percentile=cluster_percentile,
                n_cluster_perms=n_perm,
                random_state=seed,
            )

            window_mask = np.zeros(n_windows, dtype=bool)
            for s, e in sig_clusters_windows:
                window_mask[s:e + 1] = True

            sample_mask = np.zeros(n_times, dtype=bool)
            for s, e in sig_clusters_windows:
                first = win_to_samples[s][0]
                last = win_to_samples[e][1]
                sample_mask[first:last + 1] = True

            results[roi][eff] = {
                'observed_F': obs,
                'null_F': null,
                'sig_clusters_windows': sig_clusters_windows,
                'window_mask': window_mask,
                'sample_mask': sample_mask,
            }
            if verbose and sig_clusters_windows:
                print(f"[anova-cluster]   {eff}: {len(sig_clusters_windows)} cluster(s), "
                      f"window indices {sig_clusters_windows}")

    return results, window_centers


def anova_results_to_interaction_results_for_plotting(
    anova_cluster_results, anova_interactions
):
    """Adapt the full-ANOVA result dict into the shape my mega-plot expects.

    The mega-plot's interaction_results structure is:
        dict[roi][interaction_name] -> {'mask', 'label', 'factors', ...}
    where `mask` is the *sample-level* boolean array.
    """
    out = {}
    for roi, by_effect in anova_cluster_results.items():
        out[roi] = {}
        for inter in anova_interactions:
            f1, f2 = inter['factors']
            # statsmodels names the interaction term as 'C(f1):C(f2)' regardless
            # of which factor is listed first in the formula
            candidate_names = [f'C({f1}):C({f2})', f'C({f2}):C({f1})']
            info = None
            for nm in candidate_names:
                if nm in by_effect:
                    info = by_effect[nm]
                    break
            if info is None:
                continue
            out[roi][inter['name']] = {
                'mask': info['sample_mask'],
                't_obs': info['observed_F'],   # F-trace, but field name kept for compat
                'cluster_p_values': np.array([]),  # not exposed by the percentile method
                'factors': (f1, f2),
                'levels': (_get_factor_levels({}, f1) or [], _get_factor_levels({}, f2) or []),
                'label': inter.get('label', inter['name']),
            }
    return out

