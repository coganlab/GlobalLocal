import numpy as np
import mne
import matplotlib.pyplot as plt
import os
from typing import Union, List, Sequence
import logging
from ieeg.calc.stats import time_perm_cluster

#to save print statements while on cluster
# PROJECT_DIR = '/hpc/group/coganlab/etb28/GlobalLocal/src/analysis/power' 

# LOG_DIR = os.path.join(PROJECT_DIR, 'logs')
# os.makedirs(LOG_DIR, exist_ok=True) 

# log_file_path = os.path.join(LOG_DIR, 'power_traces_debug.log')
# logging.basicConfig(filename='power_traces_debug.log', 
#                     level=logging.DEBUG, 
#                     format='%(asctime)s - %(message)s',
#                     filemode='w')

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

def plot_power_trace_for_roi(evks_dict, roi, condition_names, conditions_save_name, plotting_parameters, significant_clusters=None, window_size=None, sampling_rate=None, 
                            save_dir=None, show_std=True, show_sem=False, show_ci=False, ci=0.95, figsize=(12, 8), x_label='Time (s)', ylim=None, y_label='Power (z)', axis_font_size=12, tick_font_size=12, title_font_size=14, save_name_suffix=None):
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
        Dictionary with plotting parameters
    save_dir : str
        Directory to save the plot
    show_std : bool
        Whether to show standard deviation shading
    show_sem : bool
        Whether to show standard error of mean shading
    figsize : tuple
        Figure size 
    ylim : tuple
        Y-axis limits
    save_name_suffix : str
        Suffix to add to the save name
    significant_clusters : array-like of bool
        A boolean array indicating which time windows are part of a
        statistically significant cluster. Shape: (n_windows,).
    Returns:
    --------
    fig : matplotlib figure
    """
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


   # Get the number of timepoints
    time_axis_length = times

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

         # Compute window duration
        window_duration = window_size / sampling_rate

        clusters = find_clusters(significant_clusters)

        # # Determine y position for the bars
        # max_y = np.max(mean_true_accuracy + se_true_accuracy)
        # min_y = np.min(mean_shuffle_accuracy - se_shuffle_accuracy)
        # y_bar = max_y + 0.02  # Adjust as needed
        # plt.ylim([min_y, y_bar + 0.05])  # Adjust ylim to accommodate the bars

        # Set y_bar to a fixed value within the y-axis limits
        y_bar = 0.95  # Fixed value near the top of the y-axis

        # Plot horizontal bars and asterisks for significant clusters
        for cluster in clusters:
            start_idx, end_idx = cluster
            
            if window_size is None or window_size == 0:
                # Point-wise analysis: Bar spans the centers of the first/last points
                start_time = times[start_idx]
                end_time = times[end_idx]
            else:
                # Windowed analysis: Bar spans the outer edges of the first/last windows
                window_duration = window_size / sampling_rate
                start_time = times[start_idx] - (window_duration / 2)
                end_time = times[end_idx] + (window_duration / 2)
                
            plt.hlines(y=y_bar, xmin=start_time, xmax=end_time, color='black', linewidth=2)  
            # Place an asterisk at the center of the bar
            center_time = (start_time + end_time) / 2
            plt.text(center_time, y_bar + 0.01, '*', ha='center', va='bottom', fontsize=14)

    # Customize plot
    text_color = "#002060"

    ax.set_xlabel(x_label, fontsize=axis_font_size, color=text_color)
    ax.set_ylabel(y_label, fontsize=axis_font_size, color=text_color)
    ax.axhline(y=0, color='black', linestyle=':', alpha=0.5)
    ax.axvline(x=0, color='black', linestyle=':', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', colors=text_color, labelsize=tick_font_size)
    
    # Set title
    title = f'{roi.upper()}'

    if show_std:
        title += ' (±1 SD)'
    elif show_sem:
        title += ' (±1 SEM)'
    elif show_ci:
        title += f' ({ci*100}% CI)'

    ax.set_title(title, fontsize=title_font_size, fontweight='bold', color=text_color)
    
    if ylim:
        ax.set_ylim(ylim)
    
    ax.legend(loc='best', framealpha=0.95)
    #ax.grid(False, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # Save if directory provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        error_type = 'std' if show_std else 'sem' if show_sem else 'ci' if show_ci else 'no_error'
        filename = f'{roi}_{conditions_save_name}_{save_name_suffix}_{error_type}_shading.png'
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {filepath}")
    
    plt.close()
    return fig

def plot_power_traces_for_all_rois(evks_dict_elecs, rois, 
                                  condition_names, conditions_save_name, plotting_parameters, window_size=None, sampling_rate=None, significant_clusters=None, save_dir=None,
                                  error_type='std', figsize=(12, 8), x_label='Time (s)', y_label='Power (z)',
                                  axis_font_size=12, tick_font_size=12, title_font_size=14, save_name_suffix=None):
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
        Plotting parameters dictionary
    save_dir : str
        Directory to save plots
    error_type : str
        Type of error to show: 'std', 'sem', 'ci', or 'none'
    x_label : str
        X-axis label
    y_label : str
        Y-axis label
    axis_font_size : int
        Font size for axis labels and title
    tick_font_size : int
        Font size for tick labels
    title_font_size : int
        Font size for title
    figsize : tuple
        Figure size for each plot
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
        if error_type == 'std':
            # Use custom function for standard deviation
            plot_power_trace_for_roi(
                evks_dict_elecs, roi, condition_names, conditions_save_name, plotting_parameters, window_size=window_size, sampling_rate=sampling_rate, 
                significant_clusters=clusters_for_this_roi,
                save_dir=save_dir,
                show_std=True, show_sem=False, axis_font_size=axis_font_size, tick_font_size=tick_font_size, 
                x_label=x_label, y_label=y_label,
                title_font_size=title_font_size, figsize=figsize, save_name_suffix=save_name_suffix
            )
        elif error_type == 'sem':
            # Use custom function for standard error
            plot_power_trace_for_roi(
                evks_dict_elecs, roi, condition_names, conditions_save_name, plotting_parameters, window_size=window_size, sampling_rate=sampling_rate, 
                significant_clusters=clusters_for_this_roi,
                save_dir=save_dir,
                show_std=False, show_sem=True, axis_font_size=axis_font_size, tick_font_size=tick_font_size, 
                x_label=x_label, y_label=y_label,
                title_font_size=title_font_size, figsize=figsize, save_name_suffix=save_name_suffix
            )
        elif error_type == 'ci':
            # Use MNE function with 95% CI
            plot_power_trace_for_roi(
                evks_dict_elecs, roi, condition_names, conditions_save_name, plotting_parameters, window_size=window_size, sampling_rate=sampling_rate, 
                significant_clusters=clusters_for_this_roi,
                save_dir=save_dir,
                show_std=False, show_sem=False, show_ci=True, ci=0.95, axis_font_size=axis_font_size, tick_font_size=tick_font_size, 
                x_label=x_label, y_label=y_label,
                title_font_size=title_font_size, figsize=figsize, save_name_suffix=save_name_suffix
            )
        else:
            # No error bars
            plot_power_trace_for_roi(
                evks_dict_elecs, roi, condition_names, conditions_save_name, plotting_parameters, window_size=window_size, sampling_rate=sampling_rate, significant_clusters=clusters_for_this_roi,
                save_dir=save_dir,
                show_std=False, show_sem=False, show_ci=False, ci=None, axis_font_size=axis_font_size, tick_font_size=tick_font_size, 
                x_label=x_label, y_label=y_label,
                title_font_size=title_font_size, figsize=figsize, save_name_suffix=save_name_suffix
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
    
    