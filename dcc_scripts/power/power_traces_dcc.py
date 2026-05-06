
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import sys
import os

# Get the absolute path to the directory containing the current script
try:
    current_file_path = os.path.abspath(__file__)
    current_script_dir = os.path.dirname(current_file_path)
except NameError:
    current_script_dir = os.getcwd()

# Navigate up two levels to get to the 'GlobalLocal' directory
project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..'))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ieeg.io import get_data
import mne
import numpy as np
import json

import matplotlib.pyplot as plt

from src.analysis.config import experiment_conditions
from src.analysis.config.plotting_parameters import plotting_parameters as plot_params
from src.analysis.config.condition_registry import (
    get_comparisons,
    get_conditions_obj,
    get_subtraction_pairs,
    get_anova_factors,
    get_anova_interactions,
)
import src.analysis.utils.general_utils as utils
from src.analysis.power.power_traces import (
    make_multi_channel_evokeds_for_all_conditions_and_rois,
    plot_power_traces_for_all_rois,
    create_subtracted_evokeds_dict,
    time_perm_cluster_between_two_evokeds,
    process_windowed_data_for_anova,
    run_windowed_anova_cluster_correction,
    anova_results_to_interaction_results_for_plotting,
    plot_anova_interaction_results,
)


def main(args):
    # ------------------------------------------------------------------
    # 1. Resolve LAB_root
    # ------------------------------------------------------------------
    if args.LAB_root is None:
        HOME = os.path.expanduser("~")
        USER = os.path.basename(HOME)

        if os.name == 'nt':  # Windows
            LAB_root = os.path.join(HOME, "Box", "CoganLab")
        elif sys.platform == 'darwin':  # macOS
            LAB_root = os.path.join(HOME, "Library", "CloudStorage",
                                    "Box-Box", "CoganLab")
        else:  # Linux (cluster)
            if os.path.exists(f"/cwork/{USER}"):
                LAB_root = f"/cwork/{USER}"
            else:
                LAB_root = os.path.join(HOME, "CoganLab")
    else:
        LAB_root = args.LAB_root

    # ------------------------------------------------------------------
    # 2. Resolve conditions / comparisons / subtraction_pairs / anova
    #    config from the registry, driven by condition_label.
    # ------------------------------------------------------------------
    condition_label = args.condition_label
    conditions = get_conditions_obj(condition_label)
    condition_names = list(conditions.keys())
    condition_comparisons = get_comparisons(condition_label)
    subtraction_pairs = get_subtraction_pairs(condition_label)
    anova_factors = get_anova_factors(condition_label)
    anova_interactions = get_anova_interactions(condition_label)

    conditions_save_name = utils.get_conditions_save_name(
        conditions, experiment_conditions, len(args.subjects)
    )

    # ------------------------------------------------------------------
    # 3. Build save dir, electrodes, mne objects
    # ------------------------------------------------------------------
    config_dir = os.path.join(project_root, 'src', 'analysis', 'config')
    subjects_electrodestoROIs_dict = utils.make_or_load_subjects_electrodes_to_ROIs_dict(
        subjects=args.subjects, save_dir=config_dir,
        filename='subjects_electrodestoROIs_dict.json'
    )

    layout = get_data(args.task, root=LAB_root)
    
    if args.save_dir:
        save_dir = args.save_dir
    else:
        save_dir = os.path.join(layout.root, 'derivatives', 'freqFilt', 'figs',
                                f"{args.epochs_root_file}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Save directory created or already exists at: {save_dir}")

    sig_chans_per_subject = utils.get_sig_chans_per_subject(
        args.subjects, args.epochs_root_file, task=args.task, LAB_root=LAB_root
    )

    rois = list(args.rois_dict.keys())
    all_electrodes_per_subject_roi, sig_electrodes_per_subject_roi = \
        utils.make_sig_electrodes_per_subject_and_roi_dict(
            args.rois_dict, subjects_electrodestoROIs_dict, sig_chans_per_subject
        )

    subjects_mne_objects = utils.create_subjects_mne_objects_dict(
        subjects=args.subjects, epochs_root_file=args.epochs_root_file,
        conditions=conditions, task=args.task,
        just_HG_ev1_rescaled=True, acc_trials_only=args.acc_trials_only
    )

    if args.electrodes == 'all':
        raw_electrodes = all_electrodes_per_subject_roi
        elec_string_to_add_to_filename = 'all_elecs'
    elif args.electrodes == 'sig':
        raw_electrodes = sig_electrodes_per_subject_roi
        elec_string_to_add_to_filename = 'sig_elecs'
    else:
        raise ValueError("electrodes input must be set to all or sig")

    electrodes = utils.filter_electrode_lists_against_subjects_mne_objects(
        rois, raw_electrodes, subjects_mne_objects
    )

    dropped_electrodes, _ = utils.find_difference_between_two_electrode_lists(
        raw_electrodes, electrodes
    )
    print("\n--- Summary of Dropped Electrodes ---")
    total_dropped = 0
    for roi, sub_dict in dropped_electrodes.items():
        if not sub_dict:
            continue
        print(f"ROI: {roi}")
        for sub, elec_list in sub_dict.items():
            if elec_list:
                print(f"  - Subject {sub}: Dropped {len(elec_list)} electrode(s)")
                total_dropped += len(elec_list)
    print(f"Total electrodes dropped across all subjects/ROIs: {total_dropped}")
    print("-------------------------------------\n")

    # ------------------------------------------------------------------
    # 4. Build evokeds for all conditions
    # ------------------------------------------------------------------
    evks_dict_elecs = make_multi_channel_evokeds_for_all_conditions_and_rois(
        subjects_mne_objects, args.subjects, rois, condition_names, electrodes
    )

    # ------------------------------------------------------------------
    # 5. Statistical testing
    # ------------------------------------------------------------------
    significant_clusters = {}
    interaction_results = None

    if args.statistical_method == 'time_perm_cluster':
        print("\nRunning statistical tests comparing TWO conditions")
        if len(condition_names) != 2:
            raise ValueError("Time perm cluster stats requires exactly two conditions.")

        p_values_dict = {}
        condition1_name, condition2_name = condition_names

        for roi in rois:
            print(f"-- Processing ROI: {roi} --")
            try:
                evoked_cond1_this_roi = evks_dict_elecs[condition1_name][roi]
                evoked_cond2_this_roi = evks_dict_elecs[condition2_name][roi]

                mask_roi, p_values = time_perm_cluster_between_two_evokeds(
                    evoked_cond1_this_roi, evoked_cond2_this_roi,
                    p_thresh=args.p_thresh_for_time_perm_cluster_stats,
                    p_cluster=args.p_cluster, n_perm=args.n_perm,
                    tails=args.tails, axis=0, stat_func=args.stat_func,
                    ignore_adjacency=None,
                    permutation_type=args.permutation_type,
                    vectorized=True, n_jobs=args.n_jobs, seed=None, verbose=True,
                )

                significant_clusters[roi] = mask_roi
                p_values_dict[roi] = p_values
            except KeyError:
                print(f"   Skipping ROI {roi}: Missing prepared evoked data for "
                      f"one or both conditions.")

    elif args.statistical_method == 'anova':
        if not anova_factors or not anova_interactions:
            raise ValueError(
                f"condition_label '{condition_label}' has no 'anova_factors' / "
                f"'anova_interactions' in the registry."
            )
        print(f"\nRunning full {len(anova_factors)}-way ANOVA cluster correction "
              f"across {len(rois)} ROI(s) with {args.n_perm} permutations")

        # Pull a representative evoked to get the full time vector + n_times
        ref_evk = next(
            (evks_dict_elecs[c][r] for c in condition_names for r in rois
             if evks_dict_elecs[c][r] is not None and evks_dict_elecs[c][r].data.shape[0] > 0),
            None,
        )
        if ref_evk is None:
            raise RuntimeError("No usable evoked data found for ANOVA path.")
        full_times = ref_evk.times

        windowed_data = process_windowed_data_for_anova(
            subjects_mne_objects, condition_names, rois, args.subjects,
            electrodes, window_size=args.window_size,
            step_size=args.step_size, sampling_rate=args.sampling_rate,
        )

        anova_cluster_results, window_centers = run_windowed_anova_cluster_correction(
            windowed_data, conditions, anova_factors, rois, args.subjects,
            electrodes_per_subject_roi=electrodes,
            times=full_times,
            window_size=args.window_size, step_size=args.step_size,
            sampling_rate=args.sampling_rate,
            n_perm=args.n_perm,
            percentile=int(100 * (1 - args.p_thresh_for_time_perm_cluster_stats)),
            cluster_percentile=int(100 * (1 - args.p_cluster)),
            seed=42, n_jobs=args.n_jobs, verbose=True,
        )

        # Adapt into the structure the mega-plot expects, picking out the four 2-ways
        interaction_results = anova_results_to_interaction_results_for_plotting(
            anova_cluster_results, anova_interactions,
        )

    # ------------------------------------------------------------------
    # 6. Plotting
    # ------------------------------------------------------------------
    plot_power_traces_for_all_rois(
        evks_dict_elecs, rois, condition_names, conditions_save_name, plot_params,
        args.window_size, args.sampling_rate,
        significant_clusters=significant_clusters or None,
        save_dir=save_dir,
        error_type='sem',
        plot_style=args.plot_style,
        save_name_suffix=elec_string_to_add_to_filename,
    )

    # If we ran the ANOVA cluster correction, draw the 16-condition mega-plot
    # with 4 stacked interaction-cluster bars + one 4-trace plot per interaction.
    if interaction_results is not None:
        plot_anova_interaction_results(
            evks_dict_elecs, conditions, condition_names, conditions_save_name,
            rois, anova_interactions, interaction_results,
            plot_style=args.plot_style, save_dir=save_dir,
            save_name_suffix=elec_string_to_add_to_filename, error_type='sem',
        )

    # ------------------------------------------------------------------
    # 7. Subtraction-pair plotting (driven by registry)
    # ------------------------------------------------------------------
    if subtraction_pairs:
        # All names needed must be present in evks_dict_elecs
        usable_pairs = [p for p in subtraction_pairs
                        if p[0] in evks_dict_elecs and p[1] in evks_dict_elecs]
        if usable_pairs:
            subtracted_evks_dict_elecs = create_subtracted_evokeds_dict(
                evks_dict_elecs, usable_pairs, rois
            )
            sub_pair_names = ['-'.join(pair) for pair in usable_pairs]
            sub_save_name = (f"{condition_label}_subtractions_"
                             f"{args.epochs_root_file}_"
                             f"{len(args.subjects)}_subjects")
            plot_power_traces_for_all_rois(
                subtracted_evks_dict_elecs, rois, sub_pair_names, sub_save_name,
                plot_params, save_dir=save_dir,
                window_size=args.window_size, sampling_rate=args.sampling_rate,
                error_type='sem',
                plot_style=args.plot_style,
                save_name_suffix=elec_string_to_add_to_filename,
            )
        else:
            print("[subtraction] No usable subtraction pairs (condition keys "
                  "missing from evks_dict).")

    # ------------------------------------------------------------------
    # 8. Save results
    # ------------------------------------------------------------------

    for condition_name in condition_names:
        for roi in rois:
            evk = evks_dict_elecs[condition_name][roi]
            if evk is not None:
                np.savez(
                    os.path.join(
                        save_dir,
                        f'{conditions_save_name}_{condition_name}_{roi}_evoked.npz'
                    ),
                    data=evk.data, times=evk.times, ch_names=evk.ch_names
                )

    if significant_clusters:
        np.savez(
            os.path.join(save_dir,
                         f'{conditions_save_name}_significant_clusters.npz'),
            **significant_clusters
        )

    if interaction_results is not None:
        # Persist as one file per ROI / interaction (masks + p values)
        for roi, by_inter in interaction_results.items():
            for inter_name, info in by_inter.items():
                np.savez(
                    os.path.join(
                        save_dir,
                        f'{conditions_save_name}_{roi}_{inter_name}_interaction_cluster.npz'
                    ),
                    mask=info['mask'],
                    t_obs=info['t_obs'],
                    cluster_p_values=info['cluster_p_values'],
                )
                
    if args.statistical_method == 'anova':
        # Save the full ANOVA F-traces too (all 16 effects, not just the 4 plotted)
        try:
            anova_save_dir = os.path.join(save_dir, 'anova_F_traces')
            os.makedirs(anova_save_dir, exist_ok=True)
            for roi, by_effect in anova_cluster_results.items():
                for eff, info in by_effect.items():
                    safe_eff = eff.replace(':', '_x_').replace('C(', '').replace(')', '')
                    np.savez(
                        os.path.join(anova_save_dir,
                                     f'{conditions_save_name}_{roi}_{safe_eff}.npz'),
                        observed_F=info['observed_F'],
                        null_F=info['null_F'],
                        window_mask=info['window_mask'],
                        sample_mask=info['sample_mask'],
                    )
        except NameError:
            pass  # anova_cluster_results not in scope (e.g. another method ran)

    metadata = {
        'condition_label': condition_label,
        'condition_names': condition_names,
        'rois': rois,
        'conditions_save_name': conditions_save_name,
        'sfreq': args.sampling_rate,
        'statistical_method': args.statistical_method,
        'anova_factors': anova_factors,
        'anova_interactions': [
            {'name': i['name'], 'factors': i['factors'],
             'label': i.get('label', i['name'])}
            for i in anova_interactions
        ] if anova_interactions else [],
    }
    with open(os.path.join(save_dir,
                           f'{conditions_save_name}_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        pass
    else:
        print("This script should be called via run_power_traces_dcc.py")
        print("Direct command-line execution is not supported with complex parameters.")
        sys.exit(1)