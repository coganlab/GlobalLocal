"""Building evoked objects for power-trace analysis.

Per-electrode and multi-channel evokeds, ROI grand averages, condition
subtraction, and the cluster-stats comparison between two evokeds. This is the
"data assembly" layer that the plotting and ANOVA modules consume.
"""

import numpy as np
import mne
from ieeg.calc.stats import time_perm_cluster


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
    evoked : mne.Evoked or None
        Evoked object for the subtracted condition, or ``None`` when either
        condition has no data for the requested ROI.
    """
    evoked_cond1 = evks_dict.get(cond1, {}).get(roi)
    evoked_cond2 = evks_dict.get(cond2, {}).get(roi)

    # ``combine_single_channel_evokeds`` deliberately returns None when an
    # ROI has no electrodes.  Preserve that missing-data marker instead of
    # passing it to MNE, which expects every item to expose ``nave``.
    if evoked_cond1 is None or evoked_cond2 is None:
        return None

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
