'''
This does visualization debugging (2D PCA) for decoding
'''

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
project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..', '..'))

# Add the 'GlobalLocal' directory to sys.path if it's not already there
if project_root not in sys.path:
    sys.path.insert(0, project_root)  # insert at the beginning to prioritize it

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from src.analysis.config import experiment_conditions
from src.analysis.config.condition_registry import get_comparisons, get_pooled_shuffle_settings, get_conditions_obj
from src.analysis.utils.labeled_array_utils import make_bootstrapped_roi_labeled_arrays_with_nan_trials_removed_for_each_channel
from src.analysis.decoding.decoding import concatenate_and_balance_data_for_decoding, plot_high_dim_decision_slice, plot_pca_3d_trajectory, plot_static_pca_projection, plot_pca_over_time

# TODO: go through and add save dir and test these plotting functions. Also, update to use condition_registry. Add a vis pairs line to each condition_registry entry i think.
def run_visualization_debug(args, rois, condition_label, electrodes, subjects_mne_objects, save_dir):
    conditions = get_conditions_obj(condition_label)
    condition_comparisons = get_comparisons(condition_label)
    condition_names = list(conditions.keys()) # get the condition names as a list

    print(f"\n{'='*20} 🔬 RUNNING 2D VISUALIZATION DEBUG (first two PCs and decision boundary) {'='*20}\n")

    # 2. Get the single data sample for visualization
    print("Generating LabeledArray data for visualization (n_bootstraps=1)...")
    roi_labeled_arrays_viz = make_bootstrapped_roi_labeled_arrays_with_nan_trials_removed_for_each_channel(
        rois=rois,
        subjects_data_objects=subjects_mne_objects,
        condition_names=condition_names, # This is already defined in main()
        subjects=args.subjects,
        electrodes_per_subject_roi=electrodes, # This is already defined in main()
        n_bootstraps=1,
        chans_axs=args.chans_axs,
        time_axs=args.time_axs,
        random_state=args.random_state,
        n_jobs=args.n_jobs
    )
    roi_labeled_arrays_viz = {roi: arrs[0] for roi, arrs in roi_labeled_arrays_viz.items() if arrs}

    # 3. Loop through ROIs and Pairs and plot
    for condition_comparison, strings_to_find in condition_comparisons.items():

        for roi in rois:
            if roi not in roi_labeled_arrays_viz:
                print(f"Skipping visualization for {roi}: No data found.")
                continue
            
            print(f"\n--- Plotting for ROI: {roi} and condition_comparison {condition_comparison} ---")

            # 4. Get balanced data and 'cats'
            data, labels, cats = concatenate_and_balance_data_for_decoding(
                roi_labeled_arrays_viz, roi, strings_to_find, args.obs_axs,
                balance_method='subsample', # Must use subsample for this viz
                random_state=args.random_state
            )
            if data.size == 0:
                print("No data after balancing. Skipping plot.")
                continue
                
            data_flat = data.reshape(data.shape[0], -1)

            # 5. Create and FIT the FULL pipeline
            # This uses the *exact* classifier and PCA settings from your args
            full_pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('pca', PCA(n_components=args.explained_variance)), 
                ('clf', args.clf_model) 
            ])
            
            full_pipeline.fit(data_flat, labels)
            print("Fit complete.")

            # 5. Call the plotting functions
            
            plot_static_pca_projection(
                roi_labeled_arrays_viz,
                roi,
                strings_to_find,
                cats,
                save_dir=os.path.join(save_dir, f"{condition_comparison}", f"{roi}"),
                obs_axs=args.obs_axs,
                random_state=args.random_state
            )
            
            plot_pca_over_time(
                roi_labeled_arrays_viz,
                roi,
                strings_to_find,
                cats,
                window_size=args.window_size,
                step_size=args.step_size,
                sampling_rate=args.sampling_rate,
                first_time_point=args.first_time_point,
                save_dir=os.path.join(save_dir, f"{condition_comparison}", f"{roi}"),
                obs_axs=args.obs_axs,
                random_state=args.random_state,
            )
            
            plot_pca_3d_trajectory(
                roi_labeled_arrays_viz, roi, strings_to_find, cats,
                save_dir=os.path.join(save_dir, f"{condition_comparison}", f"{roi}"),
                window_size=args.window_size,
                step_size=args.step_size,
                sampling_rate=args.sampling_rate,
                first_time_point=args.first_time_point,
                obs_axs=args.obs_axs,
                random_state=args.random_state,
                explained_variance=args.explained_variance,
                clf=args.clf_model
            )
                    
    print(f"\n{'='*20} ✅ VISUALIZATION DEBUG COMPLETE {'='*20}\n")

