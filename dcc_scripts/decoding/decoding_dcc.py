import sys
import os

try:
    current_file_path = os.path.abspath(__file__)
    current_script_dir = os.path.dirname(current_file_path)
except NameError:
    current_script_dir = os.getcwd()

project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..'))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ieeg.navigate import channel_outlier_marker, trial_ieeg, crop_empty_data, outliers_to_nan
from ieeg.io import raw_from_layout, get_data
from ieeg.calc.stats import time_perm_cluster
from ieeg.decoding.models import PcaLdaClassification
from ieeg.calc.oversample import MinimumNaNSplit

import mne
import numpy as np
import pandas as pd
import json
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from src.analysis.config import experiment_conditions

from src.analysis.utils.general_utils import (
    make_or_load_subjects_electrodes_to_ROIs_dict,
    load_subjects_electrodes_to_ROIs_dict,
    identify_bad_channels_by_trial_nan_rate,
    impute_trial_nans_by_channel_mean,
    create_subjects_mne_objects_dict,
    filter_electrode_lists_against_subjects_mne_objects,
    find_difference_between_two_electrode_lists,
    print_summary_of_dropped_electrodes,
    get_conditions_save_name,
    get_default_LAB_root,
    build_condition_comparisons,
    get_sig_chans_per_subject,
    make_sig_electrodes_per_subject_and_roi_dict,
)

from src.analysis.utils.labeled_array_utils import (
    put_data_in_labeled_array_per_roi_subject,
    remove_nans_from_labeled_array,
    remove_nans_from_all_roi_labeled_arrays,
    concatenate_conditions_by_string,
    get_data_in_time_range,
    make_bootstrapped_roi_labeled_arrays_with_nan_trials_removed_for_each_channel,
)

from src.analysis.decoding.decoding import (
    get_and_plot_confusion_matrix_for_rois_jim,
    Decoder,
    windower,
    get_confusion_matrices_for_rois_time_window_decoding_jim,
    compute_accuracies,
    plot_true_vs_shuffle_accuracies,
    plot_accuracies_with_multiple_sig_clusters,
    plot_accuracies_nature_style,
    make_pooled_shuffle_distribution,
    find_significant_clusters_of_series_vs_distribution_based_on_percentile,
    compute_pooled_bootstrap_statistics,
    do_time_perm_cluster_comparing_two_true_bootstrap_accuracy_distributions,
    do_mne_paired_cluster_test,
    get_pooled_accuracy_distributions_for_comparison,
    get_time_averaged_confusion_matrix,
    cluster_perm_paired_ttest_by_duration,
    run_two_one_tailed_tests_with_time_perm_cluster,
    extract_pooled_cm_traces,
    plot_cm_traces_nature_style,
    run_context_comparison_analysis,
    plot_cross_block_overlay,
)

from src.analysis.decoding.process_bootstrap import process_bootstrap
from src.analysis.decoding.run_visualization_debug import run_visualization_debug
from src.analysis.decoding.run_debug_cm_traces import run_debug_cm_traces
from src.analysis.decoding.run_aggregate_and_plot_time_averaged_cms import run_aggregate_and_plot_time_averaged_cms

'''
when adding a new condition to decoding - 
1. update get_conditions_save_name. 
2. update process_bootstrap for the pooled_conditions. 
3. update run_visualization_debug for the new condition.
4. update run_context_comparisons.py to include the new condition and comparisons if you want to compare both true vs. true and true vs. shuffle.

'''
def main(args):
    # Determine LAB_root based on the operating system and environment
    if args.LAB_root is None:
        LAB_root = get_default_LAB_root()
    else:
        LAB_root = args.LAB_root
    print('LAB_root: ', LAB_root)
    
    config_dir = os.path.join(project_root, 'src', 'analysis', 'config')
    subjects_electrodestoROIs_dict = load_subjects_electrodes_to_ROIs_dict(save_dir=config_dir, filename='subjects_electrodestoROIs_dict.json')
    
    condition_names = list(args.conditions.keys()) # get the condition names as a list
    conditions_save_name = args.condition_label # apparently this isn't even used and can be deleted...2/26/26. But still make sure to update get_conditions_save_name as you add new conditions.
    
    save_dir = os.path.join(LAB_root, 'BIDS-1.1_GlobalLocal', 'BIDS', 'derivatives', 'decoding', 'figs', f"{args.epochs_root_file}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Save directory created or already exists at: {save_dir}")
    
    sig_chans_per_subject = get_sig_chans_per_subject(args.subjects, args.epochs_root_file, task=args.task, LAB_root=LAB_root)

    rois = list(args.rois_dict.keys())
    all_electrodes_per_subject_roi, sig_electrodes_per_subject_roi = make_sig_electrodes_per_subject_and_roi_dict(args.rois_dict, subjects_electrodestoROIs_dict, sig_chans_per_subject)
      
    subjects_mne_objects = create_subjects_mne_objects_dict(subjects=args.subjects, epochs_root_file=args.epochs_root_file, conditions=args.conditions, task=args.task, just_HG_ev1_rescaled=True, acc_trials_only=args.acc_trials_only)
    
    # determine which electrodes to use (all electrodes or just the task-significant ones)
    if args.electrodes == 'all':
        raw_electrodes = all_electrodes_per_subject_roi 
        elec_string_to_add_to_filename = 'all_elecs'
    elif args.electrodes == 'sig':
        raw_electrodes = sig_electrodes_per_subject_roi
        elec_string_to_add_to_filename = 'sig_elecs'

    else:
        raise ValueError("electrodes input must be set to all or sig")
    
    # ADD THIS BLOCK to create a string for the sampling method
    folds_info_str = 'folds_as_samples' if args.folds_as_samples else 'repeats_as_samples'

    # filter electrodes to only the ones that exist in the epochs objects. This mismatch can arise due to dropping channels when making the epochs objects, because the subjects_electrodestoROIs_dict is made based on all the electrodes, with no dropping.
    electrodes = filter_electrode_lists_against_subjects_mne_objects(rois, raw_electrodes, subjects_mne_objects)
    
    print_summary_of_dropped_electrodes(raw_electrodes, electrodes)
    
    condition_comparisons = build_condition_comparisons(args.conditions, experiment_conditions) # make sure to edit this function as you add new condition comparisons
 
    # get the confusion matrix using the downsampled version
    # add elec and subject info to filename 6/11/25
    other_string_to_add = (
        f"{elec_string_to_add_to_filename}_{str(len(args.subjects))}_subjects_{folds_info_str}_ev_{args.explained_variance}"
    )
            
    time_window_decoding_results = {}     
     
    print(f"\n{'='*20} STARTING PARALLEL BOOTSTRAPPING ({args.bootstraps} samples across {args.n_jobs} jobs) {'='*20}\n")

    if args.run_visualization_debug: # this needs to be pulled into its own function so that this can be a one liner.
        run_visualization_debug(args, rois, condition_names, electrodes, subjects_mne_objects, save_dir)
        
    # use joblib to run the bootstrap processing in parallel
    bootstrap_results_list = Parallel(n_jobs=args.n_jobs, verbose=10, backend='loky')(
        delayed(process_bootstrap)(
            bootstrap_idx,
            subjects_mne_objects,
            args,
            rois,
            condition_names,
            electrodes,
            condition_comparisons,
            save_dir
        ) for bootstrap_idx in range(args.bootstraps)
    )

    # reconstruct the main results dictionary from the list returned by the parallel jobs
    time_window_decoding_results = {i: result['time_window_results'] for i, result in enumerate(bootstrap_results_list) if result is not None}
    time_averaged_cms_list = [result['time_averaged_cms'] for result in bootstrap_results_list if result]

    ## Extract the 'cats_by_roi' dictionary from the first valid bootstrap result.
    ## This is necessary to get the correct labels for plotting the confusion matrices.
    cats_by_roi = {}
    first_valid_result = next((res for res in bootstrap_results_list if res), None)
    if first_valid_result:
        cats_by_roi = first_valid_result.get('cats_by_roi', {})

    # --- Step 1: Aggregate and Plot Time-Averaged CMs ---
    run_aggregate_and_plot_time_averaged_cms(
        time_averaged_cms_list, condition_comparisons, rois, cats_by_roi, args, save_dir
    )
    
    if not time_window_decoding_results:
        print("\n✗ Analysis failed: No bootstrap samples were successfully processed.")
        return
    
    print(f"\n{'-'*20} PARALLEL BOOTSTRAPPING COMPLETE {'='*20}\n")
    
    # after all bootstraps complete, run pooled statistics
    all_bootstrap_stats = compute_pooled_bootstrap_statistics(
        time_window_decoding_results,
        args.bootstraps,
        condition_comparisons,
        rois,
        percentile=args.percentile,
        cluster_percentile=args.cluster_percentile,
        n_cluster_perms=args.n_cluster_perms,
        random_state=args.random_state,
        unit_of_analysis=args.unit_of_analysis
    )
    
    sub_str = str(len(args.subjects))
    analysis_params_str = (
            f"{sub_str}_subs_{elec_string_to_add_to_filename}_{args.clf_model_str}_" 
            f"{args.bootstraps}boots_{args.n_splits}splits_{args.n_repeats}reps_"
            f"{args.unit_of_analysis}_unit_ev_{args.explained_variance}"
        )               
    master_results = {
        'stats': all_bootstrap_stats,
        'metadata': {
            'args': vars(args), # Save all arguments from the run
            'analysis_params_str': analysis_params_str
        },
        'comparison_clusters': {} # We will populate this in the loops below
    }
       
    # define color and linestyle for plotting true vs shuffle
    colors = {
        'true': '#0173B2',  # Blue
        'shuffle': '#949494'  # Gray
    }
    
    linestyles = {
        'true': '-',
        'shuffle': '--'
    }  
    
    # then plot using the pooled statistics
    for condition_comparison in condition_comparisons.keys():
        for roi in rois:
            if roi in all_bootstrap_stats[condition_comparison]:
                stats = all_bootstrap_stats[condition_comparison][roi] 
                time_window_centers = time_window_decoding_results[0][condition_comparison][roi]['time_window_centers']
                
                # extract the correct keys based on unit_of_analysis
                unit = stats['unit_of_analysis']
                
                plot_accuracies_nature_style(
                    time_points=time_window_centers,
                    accuracies_dict={
                        'true': stats[f'{unit}_true_accs'], # use the full distribution
                        'shuffle': stats[f'{unit}_shuffle_accs']
                    },
                    significant_clusters=stats['significant_clusters'],
                    window_size=args.window_size,
                    step_size=args.step_size,
                    sampling_rate=args.sampling_rate,
                    comparison_name=f'bootstrap_true_vs_shuffle_{condition_comparison}',
                    roi=roi,
                    save_dir=os.path.join(save_dir, f"{condition_comparison}", f"{roi}"),
                    timestamp=args.timestamp,
                    p_thresh=args.percentile,
                    colors=colors,
                    linestyles=linestyles,
                    single_column=args.single_column,
                    show_legend=args.show_legend,
                    ylim=(0.3, 0.8),
                    show_chance_level=False, # The pooled shuffle line is the new chance level 
                    filename_suffix=analysis_params_str  
                )    
                
    # pooled cm traces for debugging        
    run_debug_cm_traces(
        time_window_decoding_results=time_window_decoding_results,
        condition_comparisons=condition_comparisons,
        rois=rois,
        cats_by_roi=cats_by_roi,
        args=args,
        save_dir=save_dir,
        analysis_params_str=analysis_params_str,
    )
    
    # run context comparisons comparing true vs. true decoding and also true vs. shuffle decoding for lwpc, lwps, etc.         
    run_all_context_comparisons(
        args=args,
        time_window_decoding_results=time_window_decoding_results,
        all_bootstrap_stats=all_bootstrap_stats,
        master_results=master_results,
        rois=rois,
        save_dir=save_dir,
        analysis_params_str=analysis_params_str,
    )
                
    # --- Save all results to a single file ---
    results_filename = f"{args.timestamp}_MASTER_RESULTS_{analysis_params_str}.pkl"
    results_save_path = os.path.join(save_dir, results_filename)
    
    # Try to grab time_window_centers and add to metadata
    try:
        first_comp = list(time_window_decoding_results[0].keys())[0]
        first_roi = list(time_window_decoding_results[0][first_comp].keys())[0]
        twc = time_window_decoding_results[0][first_comp][first_roi]['time_window_centers']
        master_results['metadata']['time_window_centers'] = twc
    except Exception as e:
        print(f"Warning: Could not save time_window_centers to metadata. {e}")

    print(f"\n💾 Saving all statistical results to: {results_save_path}")
    with open(results_save_path, 'wb') as f:
        pickle.dump(master_results, f)

    print("\n✅ Analysis and saving complete.")
                  
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
        print("This script should be called via run_decoding.py")
        print("Direct command-line execution is not supported with complex parameters.")
        sys.exit(1)
