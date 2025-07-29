import sys
import os
print(sys.path)
sys.path.append("C:/Users/jz421/Desktop/GlobalLocal/IEEG_Pipelines/") #need to do this cuz otherwise ieeg isn't added to path...

# Get the absolute path to the directory containing the current script
# For GlobalLocal/src/analysis/preproc/make_epoched_data.py, this is GlobalLocal/src/analysis/preproc
current_script_dir = os.path.dirname(os.path.abspath(__file__))

# Navigate up three levels to get to the 'GlobalLocal' directory
project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..', '..'))

# Add the 'GlobalLocal' directory to sys.path if it's not already there
if project_root not in sys.path:
    sys.path.insert(0, project_root) # insert at the beginning to prioritize it

import joblib
from src.analysis.spec.wavelet_functions import get_uncorrected_wavelets, get_uncorrected_multitaper, get_sig_tfr_differences
from src.analysis.utils.general_utils import get_good_data

def make_subject_tfr_object(sub, layout, good, condition_name, condition_dict, spec_method, signal_times, freqs, n_cycles, time_bandwidth, return_itc, n_jobs, average, epochs_root_file, acc_trials_only=False, error_trials_only=False):
    '''
    Calculates, saves, and returns a TFR object for a single subject and condition.

    This function computes the time-frequency representation from epochs data
    using either multitaper or wavelet analysis, optionally filtering for
    accurate or error trials. The resulting TFR object is saved to disk.

    Parameters
    ----------
    sub : str
        The subject identifier (e.g., '01').
    layout : bids.layout.BIDSLayout
        A BIDSLayout object pointing to the dataset root.
    good : mne.Epochs
        The cleaned epochs data for the subject. Used for referencing
        bad channels and file paths.
    condition_name : str
        A descriptive name for the experimental condition (e.g., 'CorrectGo').
    condition_dict : dict
        A dictionary for the condition, which must contain the key "BIDS_events"
        with a list of BIDS event names.
    spec_method : str
        The spectral estimation method to use ('multitaper' or 'wavelets').
    signal_times : tuple of float
        The time window for TFR calculation in seconds, e.g., (-0.5, 1.5).
    freqs : np.ndarray
        An array of frequencies for the analysis.
    n_cycles : int or np.ndarray
        The number of cycles for the wavelet transform.
    time_bandwidth : float
        The time-bandwidth product for multitaper analysis (if used).
    return_itc : bool
        If True, calculates inter-trial coherence.
    n_jobs : int
        The number of parallel jobs to run for the calculation.
    average : bool
        Whether to average the TFR across trials.
    epochs_root_file : str
        The base name for the epochs root file (without the suffix).
    acc_trials_only : bool, optional
        If True, appends "/Accuracy1.0" to BIDS event names to select
        only accurate trials. Defaults to False.
    error_trials_only : bool, optional
        If True, appends "/Accuracy0.0" to BIDS event names to select
        only error trials. Defaults to False.

    Returns
    -------
    spec : mne.time_frequency.TFR
        The computed TFR object for the subject and condition.
    '''
    sub_spec_dir = os.path.join(layout.root, 'derivatives', 'spec', spec_method, sub)
    if not os.path.exists(sub_spec_dir):
        os.makedirs(sub_spec_dir)

    # make signal wavelets
    BIDS_events = condition_dict["BIDS_events"] # extract the list of BIDS events

    if acc_trials_only:
        BIDS_events = [BIDS_event + "/Accuracy1.0" for BIDS_event in BIDS_events]
    elif error_trials_only:
        BIDS_events = [BIDS_event + "/Accuracy0.0" for BIDS_event in BIDS_events]
    if spec_method == 'multitaper':
        spec = get_uncorrected_multitaper(sub=sub, layout=layout, events=BIDS_events, times=signal_times, freqs=freqs, n_cycles=n_cycles, time_bandwidth=time_bandwidth, return_itc=return_itc, n_jobs=n_jobs, average=average)
    elif spec_method == 'wavelet':
        spec = get_uncorrected_wavelets(sub=sub, layout=layout, events=BIDS_events, times=signal_times, n_jobs=n_jobs)
        if average:
            spec = spec.average(
                lambda x: np.nanmean(x, axis=0), copy=True)
    fnames = [os.path.relpath(f, layout.root) for f in good.filenames]
    spec.info['subject_info']['files'] = tuple(fnames)
    spec.info['bads'] = good.info['bads']

    # save the spec object
    save_path = os.path.join(sub_spec_dir, condition_name + '_' + spec_method + '_' + epochs_root_file)
    joblib.dump(spec, save_path)

    return spec

def make_subjects_tfr_objects(subjects, layout, conditions, spec_method, signal_times, freqs, n_cycles, time_bandwidth, return_itc, n_jobs, average, epochs_root_file="_epo.fif", acc_trials_only=False, error_trials_only=False, conditions_save_name="tfr_data"):
    '''
    Calculates and saves time-frequency representations (TFRs) for subjects.

    This function iterates through subjects and conditions to compute TFRs.
    It first checks if the TFR for a given subject and condition has already
    been computed and saved. If so, it loads the file; otherwise, it computes,
    saves, and then loads the TFR data. Finally, it saves a dictionary
    containing all TFR objects into a single file.

    Note: This function assumes that the underlying data for TFR calculation
    is not averaged across trials yet, allowing for trial-level operations
    (e.g., train-test splits) at a later stage.

    Parameters
    ----------
    subjects : list of str
        A list of subject identifiers (e.g., ['01', '02']).
    layout : bids.layout.BIDSLayout
        A BIDSLayout object pointing to the dataset root.
    conditions : dict
        A dictionary where keys are condition names (str) and
        values are dicts containing a list of BIDS event names under the key "BIDS_events".
        Example: {'ConditionA': {'BIDS_events': ['event/A1', 'event/A2']}}
    spec_method : str
        The spectral estimation method to use. Must be either
        'multitaper' or 'wavelets'.
    signal_times : tuple of float
        A tuple defining the time window for TFR
        calculation in seconds, e.g., (-0.5, 1.5).
    freqs : np.ndarray
        An array of frequencies for the analysis.
    n_cycles : int or np.ndarray
        The number of cycles for the wavelet transform.
    time_bandwidth : float
        The time-bandwidth product for multitaper analysis.
    return_itc : bool
        If True, calculates inter-trial coherence.
    n_jobs : int
        The number of parallel jobs to run.
    average : bool
        Whether to average the multitaper spectrogram across trials.
    epochs_root_file : str
        The base name for the epochs root file (without the suffix).
    acc_trials_only : bool, optional
        If True, appends "/Accuracy1.0" to BIDS
        event names to select only accurate trials. Defaults to False.
    error_trials_only : bool, optional
        If True, appends "/Accuracy0.0" to BIDS
        event names to select only error trials. Defaults to False.
    conditions_save_name (str, optional): The base name for the output file
        that aggregates all subjects' TFRs. Defaults to "tfr_data".

    Returns
    -------
    subjects_tfr_objects : dict
        A nested dictionary where outer keys are subjects (str) and
        inner keys are condition names, with values containing the corresponding TFR object for that subject and condition.
    '''
    subjects_tfr_objects_save_dir = os.path.join(layout.root, 'derivatives', 'spec', spec_method, 'subjects_tfr_objects')
    if not os.path.exists(subjects_tfr_objects_save_dir):
        os.makedirs(subjects_tfr_objects_save_dir)

    subjects_tfr_objects = {}

    for sub in subjects:
        subjects_tfr_objects[sub] = {}
        good = get_good_data(sub, layout)
        sub_spec_dir = os.path.join(layout.root, 'derivatives', 'spec', spec_method, sub)

        for condition_name, condition_dict in conditions.items():
            # Define the path where the individual TFR object is or will be saved
            spec_save_path = os.path.join(sub_spec_dir, condition_name + '_' + spec_method + '_' + epochs_root_file)

            # Check if the TFR object already exists
            if os.path.exists(spec_save_path):
                print(f"Loading existing TFR object for sub-{sub}, condition: {condition_name}")
                spec = joblib.load(spec_save_path)
            else:
                print(f"Creating TFR object for sub-{sub}, condition: {condition_name}")
                spec = make_subject_tfr_object(sub=sub, layout=layout, good=good, condition_name=condition_name, condition_dict=condition_dict, spec_method=spec_method, signal_times=signal_times, freqs=freqs, n_cycles=n_cycles, time_bandwidth=time_bandwidth, return_itc=return_itc, n_jobs=n_jobs, average=average, epochs_root_file=epochs_root_file, acc_trials_only=acc_trials_only, error_trials_only=error_trials_only)

            subjects_tfr_objects[sub][condition_name] = spec

    # Save the aggregated dictionary of all subjects' TFR objects
    save_path = os.path.join(subjects_tfr_objects_save_dir, conditions_save_name + '_' + spec_method)
    joblib.dump(subjects_tfr_objects, save_path)

    return subjects_tfr_objects

def load_or_make_subjects_tfr_objects(
    layout, 
    spec_method, 
    conditions_save_name, 
    subjects=None, 
    conditions=None, 
    signal_times=None, 
    freqs=None, 
    n_cycles=None, 
    time_bandwidth=None, 
    return_itc=None, 
    n_jobs=1, 
    average=False, 
    epochs_root_file="_epo.fif", 
    acc_trials_only=False, 
    error_trials_only=False
):
    """
    Loads pre-computed TFR objects, or makes them if they don't exist.

    This function checks for a file containing aggregated TFR objects. If the
    file exists, it is loaded. If not, this function calls
    `make_subjects_tfr_objects` to generate, save, and return the data.

    Parameters
    ----------
    layout : bids.layout.BIDSLayout
        The BIDS layout object for the dataset.
    spec_method : str
        The spectral estimation method used ('multitaper' or 'wavelets').
    conditions_save_name : str
        The base name of the file where the TFR objects are stored.
    
    ** The following parameters are required only if the TFR object file does not exist. **

    subjects : list of str, optional
        A list of subject identifiers (e.g., ['01', '02']).
    conditions : dict, optional
        A dictionary defining experimental conditions and their BIDS events.
    signal_times : tuple of float, optional
        Time window for TFR calculation, e.g., (-0.5, 1.5).
    freqs : np.ndarray, optional
        An array of frequencies for the analysis.
    n_cycles : int or np.ndarray, optional
        The number of cycles for the wavelet transform.
    time_bandwidth : float, optional
        The time-bandwidth product for multitaper analysis.
    return_itc : bool, optional
        If True, calculates inter-trial coherence.
    n_jobs : int, optional
        The number of parallel jobs to run. Defaults to 1.
    average : bool, optional
        Whether to average the TFR across trials. Defaults to False.
    epochs_root_file : str
        The base name for the epochs root file (without the suffix).
    acc_trials_only : bool, optional
        If True, select only accurate trials. Defaults to False.
    error_trials_only : bool, optional
        If True, select only error trials. Defaults to False.

    Returns
    -------
    dict
        A dictionary of TFR objects, structured as {subject: {condition: tfr_object}}.
    """
    subjects_tfr_objects_save_dir = os.path.join(layout.root, 'derivatives', 'spec', spec_method, 'subjects_tfr_objects')
    file_path = os.path.join(subjects_tfr_objects_save_dir, conditions_save_name + '_' + spec_method) # ugh it's not letting me rename the file to include an underscore so might have to remake it

    # 1. Check if the subjects_tfr_objects for this condition exists.
    if os.path.exists(file_path):
        # 2. If so, load it.
        print(f"Found existing TFR data. Loading from: {file_path}")
        subjects_tfr_objects = joblib.load(file_path)
    else:
        # 3. If not, make it.
        print(f"No existing TFR data found. Creating new file at: {file_path}")
        # Check if necessary arguments for creation are provided
        if any(arg is None for arg in [subjects, conditions, signal_times, freqs, n_cycles, time_bandwidth, return_itc]):
            raise ValueError("To create TFR objects, you must provide values for subjects, conditions, and all TFR parameters.")
        
        subjects_tfr_objects = make_subjects_tfr_objects(
            subjects=subjects,
            layout=layout,
            conditions=conditions,
            spec_method=spec_method,
            signal_times=signal_times,
            freqs=freqs,
            n_cycles=n_cycles,
            time_bandwidth=time_bandwidth,
            return_itc=return_itc,
            n_jobs=n_jobs,
            average=average,
            epochs_root_file=epochs_root_file,
            acc_trials_only=acc_trials_only,
            error_trials_only=error_trials_only,
            conditions_save_name=conditions_save_name
        )

    return subjects_tfr_objects

def get_sig_tfr_differences_per_subject(
    subjects_tfr_objects: dict,
    condition_names: list[str],
    stat_func: callable,
    p_thresh: float = 0.05,
    p_cluster: float = None,
    n_perm: int = 1000,
    tails: int = 1,
    axis: int = 0,
    ignore_adjacency: int | tuple[int, ...] = 1,
    n_jobs: int = 1,
    seed: int = None):
    """
    Performs TFR statistical analysis for each subject individually.

    Parameters
    ----------
    subjects_tfr_objects : dict
        A dictionary structured as {subject_id: {condition_name: tfr_object}}.
    condition_names : list[str]
        A list of two condition names to be compared.
    stat_func: callable, optional
        The statistical function to use for significance testing. You should probably use partial(ttest_ind, equal_var=False).
    p_thresh : float
        The p-value threshold to use for determining significant time points.
    p_cluster : float, optional
        The p-value threshold to use for determining significant clusters.
    n_perm : int, optional
        The number of permutations to perform.
    tails : int, optional
        The number of tails to use. 1 for one-tailed, 2 for two-tailed.
    axis : int, optional
        The axis to perform the permutation test across. Also known as the
        observations axis
    ignore_adjacency : int or tuple of ints, optional
        The axis or axes to ignore when finding clusters. For example, if
        sig1.shape = (trials, channels, time), and you want to find clusters
        across time, but not channels, you would set ignore_adjacency = 1.
    n_jobs : int, optional
        The number of jobs to run in parallel. -1 for all processors. Default
        is -1.
    seed : int, optional
        The random seed to use for the permutation test. Default is None.

    Returns
    -------
    sub_masks : dict
        Dictionary mapping subject IDs to binary mask arrays indicating 
        significant differences between conditions.
    sub_pvals : dict
        Dictionary mapping subject IDs to p-values for each cluster found
        in the statistical comparison.
    """
    if len(condition_names) != 2:
        raise ValueError("This function requires exactly two conditions for comparison.")

    sub_masks = {}
    sub_pvals = {}
    cond1, cond2 = condition_names[0], condition_names[1]

    for sub, tfrs in subjects_tfr_objects.items():
        print(f"Processing statistics for subject: {sub}")
        
        mask, pvals = get_sig_tfr_differences(
            tfr_data_cond1=tfrs[cond1],
            tfr_data_cond2=tfrs[cond2],
            stat_func=stat_func,
            p_thresh=p_thresh,
            p_cluster=p_cluster,
            n_perm=n_perm,
            tails=tails,
            axis=axis,
            ignore_adjacency=ignore_adjacency,
            n_jobs=n_jobs,
            seed=seed
        )
        
        sub_masks[sub] = mask
        sub_pvals[sub] = pvals

    return sub_masks, sub_pvals

def get_sig_tfr_differences_per_roi(
    subjects_tfr_objects: dict,
    electrodes_per_subject_roi: dict,
    condition_names: list[str],
    stat_func: callable,
    p_thresh: float = 0.05,
    p_cluster: float = None,
    n_perm: int = 1000,
    tails: int = 1,
    axis: int = 0,
    ignore_adjacency: int | tuple[int, ...] = 1,
    n_jobs: int = 1,
    seed: int = None):
    """
    Performs TFR statistical analysis for each ROI by combining subjects.

    Parameters
    ----------
    subjects_tfr_objects : dict
        Dictionary of TFR data: {subject_id: {condition_name: tfr_object}}.
    electrodes_per_roi : dict
        Dictionary mapping ROIs to rois and electrodes: {roi_name: {subject_id: [elecs]}}.
    condition_names : list[str]
        A list of two condition names to compare.
    stat_func: callable, optional
        The statistical function to use for significance testing. You should probably use partial(ttest_ind, equal_var=False).
    p_thresh : float
        The p-value threshold to use for determining significant time points.
    p_cluster : float, optional
        The p-value threshold to use for determining significant clusters.
    n_perm : int, optional
        The number of permutations to perform.
    tails : int, optional
        The number of tails to use. 1 for one-tailed, 2 for two-tailed.
    axis : int, optional
        The axis to perform the permutation test across. Also known as the
        observations axis
    ignore_adjacency : int or tuple of ints, optional
        The axis or axes to ignore when finding clusters. For example, if
        sig1.shape = (trials, channels, time), and you want to find clusters
        across time, but not channels, you would set ignore_adjacency = 1.
    n_jobs : int, optional
        The number of jobs to run in parallel. -1 for all processors. Default
        is -1.
    seed : int, optional
        The random seed to use for the permutation test. Default is None.

    Returns
    -------
    roi_masks : dict
        Dictionary mapping ROI names to concatenated binary mask arrays.
        For each ROI, masks from all subjects with electrodes in that ROI
        are concatenated along axis 0 (subjects dimension), resulting in
        a combined mask array with shape (n_subjects_in_roi, *original_dims).
    roi_pvals : dict
        Dictionary mapping ROI names to concatenated p-value arrays.
        For each ROI, p-values from all subjects with electrodes in that ROI
        are concatenated along axis 0, matching the structure of roi_masks.
        Empty arrays are returned for ROIs with no subject data.
    """
    if len(condition_names) != 2:
        raise ValueError("This function requires exactly two conditions for comparison.")

    roi_masks = {}
    roi_pvals = {}
    cond1, cond2 = condition_names[0], condition_names[1]

    for roi, subjects_in_roi in electrodes_per_subject_roi.items():
        print(f"Processing statistics for ROI: {roi}")
        
        subject_masks_for_roi = []
        subject_pvals_for_roi = []

        for sub, tfrs in subjects_tfr_objects.items():
            elecs = subjects_in_roi.get(sub, [])
            if not elecs:
                continue

            mask, pvals = get_sig_tfr_differences(
                tfr_data_cond1=tfrs[cond1],
                tfr_data_cond2=tfrs[cond2],
                stat_func=stat_func,
                elecs_to_pick=elecs,
                p_thresh=p_thresh,
                p_cluster=p_cluster,
                n_perm=n_perm,
                tails=tails,
                axis=axis,
                ignore_adjacency=ignore_adjacency,
                n_jobs=n_jobs,
                seed=seed
            )

            subject_masks_for_roi.append(mask)
            subject_pvals_for_roi.append(pvals)

        if subject_masks_for_roi:
            roi_masks[roi] = np.concatenate(subject_masks_for_roi, axis=0)
            roi_pvals[roi] = np.concatenate(subject_pvals_for_roi, axis=0)
        else:
            roi_masks[roi] = np.array([])
            roi_pvals[roi] = np.array([])

    return roi_masks, roi_pvals