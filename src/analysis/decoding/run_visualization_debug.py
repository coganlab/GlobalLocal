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
from src.analysis.utils.labeled_array_utils import make_bootstrapped_roi_labeled_arrays_with_nan_trials_removed_for_each_channel
from src.analysis.decoding.decoding import concatenate_and_balance_data_for_decoding, plot_high_dim_decision_slice

def run_visualization_debug(args, rois, condition_names, electrodes, subjects_mne_objects, save_dir):
    condition_comparison = None
    print(f"\n{'='*20} 🔬 RUNNING 2D VISUALIZATION DEBUG (first two PCs and decision boundary) {'='*20}\n")
    
    # 1. Define the visualization pairs for each condition set
    viz_pairs = []
    if args.conditions == experiment_conditions.stimulus_lwpc_conditions:
        print("Setting up LWPC visualization pairs...")
        viz_pairs = [(['c25'], ['i25']), (['c75'], ['i75'])]
        condition_comparison = 'LWPC_comparison'
    elif args.conditions == experiment_conditions.stimulus_lwps_conditions:
        print("Setting up LWPS visualization pairs...")
        viz_pairs = [(['s25'], ['r25']), (['s75'], ['r75'])]
        condition_comparison = 'LWPS_comparison'
    elif args.conditions == experiment_conditions.stimulus_congruency_by_switch_proportion_conditions:
        print("Setting up Congruency x Switch Prop. visualization pairs...")
        viz_pairs = [
            (['Stimulus_c_in_25switchBlock'], ['Stimulus_i_in_25switchBlock']),
            (['Stimulus_c_in_75switchBlock'], ['Stimulus_i_in_75switchBlock'])
        ]
        condition_comparison = 'congruency_by_switch_proportion_comparison'
    elif args.conditions == experiment_conditions.stimulus_switch_type_by_congruency_proportion_conditions:
        print("Setting up Switch Type x Congruency Prop. visualization pairs...")
        viz_pairs = [
            (['Stimulus_s_in_25incongruentBlock'], ['Stimulus_r_in_25incongruentBlock']),
            (['Stimulus_s_in_75incongruentBlock'], ['Stimulus_r_in_75incongruentBlock'])
        ]
        condition_comparison = 'switch_type_by_congruency_proportion_comparison'
    elif args.conditions == experiment_conditions.stimulus_task_by_congruency_conditions:
        print("Setting up Task by Congruency visualization pairs...")
        viz_pairs = [
            (['Stimulus_i_taskG'], ['Stimulus_i_taskL']),
            (['Stimulus_c_taskG'], ['Stimulus_c_taskL'])
        ]
        condition_comparison = 'task_by_congruency_comparison'
    elif args.conditions == experiment_conditions.stimulus_task_by_switch_type_conditions:
        print("Setting up Task by Switch Type visualization pairs...")
        viz_pairs = [
            (['Stimulus_s_taskG'], ['Stimulus_s_taskL']),
            (['Stimulus_r_taskG'], ['Stimulus_r_taskL'])
        ]
        condition_comparison = 'task_by_switch_type_comparison'
    elif args.conditions == experiment_conditions.stimulus_task_by_congruency_proportion_conditions:
        print("Setting up Task by Congruency Proportion visualization pairs...")
        viz_pairs = [
            (['Stimulus_taskG_in_25incongruentBlock'], ['Stimulus_taskL_in_25incongruentBlock']),
            (['Stimulus_taskG_in_75incongruentBlock'], ['Stimulus_taskL_in_75incongruentBlock'])
        ]
        condition_comparison = 'task_by_congruency_proportion_comparison'
    elif args.conditions == experiment_conditions.stimulus_task_by_switch_proportion_conditions:
        print("Setting up Task by Switch Proportion visualization pairs...")
        viz_pairs = [
            (['Stimulus_taskG_in_25switchBlock'], ['Stimulus_taskL_in_25switchBlock']),
            (['Stimulus_taskG_in_75switchBlock'], ['Stimulus_taskL_in_75switchBlock'])
        ]
        condition_comparison = 'task_by_switch_proportion_comparison'
    if not viz_pairs:
        print("Warning: No visualization pairs defined for the current condition set. Skipping debug plots.")
    else:
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
        for roi in rois:
            if roi not in roi_labeled_arrays_viz:
                print(f"Skipping visualization for {roi}: No data found.")
                continue
                
            for pair in viz_pairs:
                viz_strings = pair
                pair_name = f"{viz_strings[0][0]}_vs_{viz_strings[1][0]}"
                print(f"\n--- Plotting for ROI: {roi}, Pair: {pair_name} ---")

                try:
                    # 4. Get balanced data and 'cats'
                    data, labels, cats = concatenate_and_balance_data_for_decoding(
                        roi_labeled_arrays_viz, roi, viz_strings, args.obs_axs,
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
                    
                    print(f"Fitting pipeline for {pair_name}...")
                    full_pipeline.fit(data_flat, labels)
                    print("Fit complete.")

                    # 6. Call the plotting function
                    plot_high_dim_decision_slice(
                        fitted_pipeline=full_pipeline,
                        X_data=data_flat,
                        y_labels=labels,
                        cats=cats,
                        roi=f"{roi} ({pair_name})", # Add pair info to title,
                        save_dir=os.path.join(save_dir, f"{condition_comparison}", f"{roi}")
                    )
                except Exception as e:
                    print(f"!! FAILED to generate plot for {roi} - {pair_name}: {e}")
                    
    print(f"\n{'='*20} ✅ VISUALIZATION DEBUG COMPLETE {'='*20}\n")

