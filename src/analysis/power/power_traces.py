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

def plot_power_trace_for_roi(evks_dict, roi, condition_names, conditions_save_name, plotting_parameters,
                            save_dir=None, show_std=True, show_sem=False, show_ci=False, ci=0.95, figsize=(12, 8), x_label='Time (s)', ylim=None, y_label='Power (z)', font_size=12, title_font_size=14, save_name_suffix=None):
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
        for key in plotting_parameters.keys():
            if condition_name in key or key in condition_name:
                param_key = key
                break
        
        if param_key and param_key in plotting_parameters:
            params = plotting_parameters[param_key]
            color = params.get('color', 'black')
            linestyle = params.get('line_style', '-')
            label = params.get('condition_parameter', condition_name)
        else:
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

    # Customize plot

    ax.set_xlabel(x_label, fontsize=font_size)
    ax.set_ylabel(y_label, fontsize=font_size)
    ax.axhline(y=0, color='black', linestyle=':', alpha=0.5)
    ax.axvline(x=0, color='black', linestyle=':', alpha=0.5)
    
    # Set title
    title = f'{roi.upper()}'

    if show_std:
        title += ' (±1 SD)'
    elif show_sem:
        title += ' (±1 SEM)'
    elif show_ci:
        title += f' ({ci*100}% CI)'

    ax.set_title(title, fontsize=title_font_size, fontweight='bold')
    
    if ylim:
        ax.set_ylim(ylim)
    
    ax.legend(loc='best', framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    
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
                                  condition_names, conditions_save_name, plotting_parameters, save_dir=None,
                                  error_type='std', figsize=(12, 8), x_label='Time (s)', y_label='Power (z)',
                                  font_size=12, title_font_size=14, save_name_suffix=None):
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
    font_size : int
        Font size for labels and title
    title_font_size : int
        Font size for title
    figsize : tuple
        Figure size for each plot
    save_name_suffix : str
        Suffix to add to the save name
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    for roi in rois:
        # Plot all electrodes
        if error_type == 'std':
            # Use custom function for standard deviation
            plot_power_trace_for_roi(
                evks_dict_elecs, roi, condition_names, conditions_save_name, plotting_parameters,
                save_dir=save_dir,
                show_std=True, show_sem=False, font_size=font_size, 
                x_label=x_label, y_label=y_label,
                title_font_size=title_font_size, figsize=figsize, save_name_suffix=save_name_suffix
            )
        elif error_type == 'sem':
            # Use custom function for standard error
            plot_power_trace_for_roi(
                evks_dict_elecs, roi, condition_names, conditions_save_name, plotting_parameters,
                save_dir=save_dir,
                show_std=False, show_sem=True, font_size=font_size, 
                x_label=x_label, y_label=y_label,
                title_font_size=title_font_size, figsize=figsize, save_name_suffix=save_name_suffix
            )
        elif error_type == 'ci':
            # Use MNE function with 95% CI
            plot_power_trace_for_roi(
                evks_dict_elecs, roi, condition_names, conditions_save_name, plotting_parameters,
                save_dir=save_dir,
                show_std=False, show_sem=False, show_ci=True, ci=0.95, font_size=font_size, 
                x_label=x_label, y_label=y_label,
                title_font_size=title_font_size, figsize=figsize, save_name_suffix=save_name_suffix
            )
        else:
            # No error bars
            plot_power_trace_for_roi(
                evks_dict_elecs, roi, condition_names, conditions_save_name, plotting_parameters,
                save_dir=save_dir,
                show_std=False, show_sem=False, show_ci=False, ci=None, font_size=font_size, 
                x_label=x_label, y_label=y_label,
                title_font_size=title_font_size, figsize=figsize, save_name_suffix=save_name_suffix
            )
    
    if save_dir:
        print(f"\nAll plots saved to: {save_dir}")
    plt.close()
