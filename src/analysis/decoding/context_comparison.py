"""Top-level context-comparison analysis and cross-block overlay plotting."""

import os
import numpy as np

from .accuracy_stats import get_pooled_accuracy_distributions_for_comparison, run_two_one_tailed_tests_with_time_perm_cluster
from .plots.accuracies import plot_accuracies_nature_style, plot_accuracies_with_multiple_sig_clusters

def run_context_comparison_analysis(
    condition_name,
    condition_comparison_1,
    condition_comparison_2,
    pooled_shuffle_key,
    colors,
    linestyles,
    ylabel,
    significance_label_1,
    significance_label_2,
    time_window_decoding_results,
    all_bootstrap_stats,
    master_results,
    args,
    rois,
    save_dir,
    analysis_params_str,
    electrode_set_desc_fn=None,
    display_name=None,
    trace_labels=None,
    electrode_set_label_fn=None,
):
    """
    Run comparison analysis between two conditions with pooled shuffle distribution.

    Parameters
    ----------
    condition_name : str
        Name of the overall comparison (e.g., 'LWPC', 'LWPS')
    condition_comparison_1 : str
        First condition comparison key (e.g., 'c25_vs_i25')
    condition_comparison_2 : str
        Second condition comparison key (e.g., 'c75_vs_i75')
    pooled_shuffle_key : str
        Key for accessing pooled shuffle data in results
    colors : dict
        Color mapping for plotting
    linestyles : dict
        Linestyle mapping for plotting
    ylabel : str
        Y-axis label for accuracy plots
    significance_label_1 : str
        Label for first significance cluster
    significance_label_2 : str
        Label for second significance cluster
    time_window_decoding_results : dict
        Results from time window decoding
    all_bootstrap_stats : dict
        Pooled bootstrap statistics
    master_results : dict
        Master results dictionary to update
    args : SimpleNamespace
        Arguments object
    rois : list
        List of ROIs to analyze
    save_dir : str
        Base save directory
    analysis_params_str : str
        String describing analysis parameters for filenames
    electrode_set_desc_fn : callable(roi) -> str, optional
        Long electrode-set description with the electrode and subject counts.
        No longer put on these panels -- ``electrode_set_label_fn`` supplies
        the short form that fits in a one-line title, and the counts stay in
        the filename. Still accepted so existing callers keep working.
    display_name : str, optional
        Short analysis name for the figure title, e.g. ``'LWPC'``. Falls back
        to ``condition_name``.
    trace_labels : dict, optional
        ``{comparison key: legend label}``. Without it the legend shows the
        raw comparison keys, which is how these panels ended up carrying
        ``lwpc_block_balanced_shuffle_accs_across_pooled_conditions_across_bootstraps``
        as a legend entry.
    electrode_set_label_fn : callable(roi) -> str, optional
        Raw electrode-set label for the one-line title. Falls back to nothing,
        in which case the title is just the analysis name.
    """
    from .anova_electrode_selection import short_decoding_figure_title

    display_name = display_name or condition_name
    trace_labels = dict(trace_labels or {})
    label_1 = trace_labels.get(condition_comparison_1, condition_comparison_1)
    label_2 = trace_labels.get(condition_comparison_2, condition_comparison_2)
    shuffle_key = f'{pooled_shuffle_key}_across_bootstraps'
    shuffle_label = 'Shuffle'

    def _short_title(roi):
        raw = electrode_set_label_fn(roi) if electrode_set_label_fn else None
        return short_decoding_figure_title(display_name, raw)

    # Display labels drive the legend, so the colour and linestyle dicts have
    # to be re-keyed to match them.
    display_colors = {
        label_1: colors.get(condition_comparison_1),
        label_2: colors.get(condition_comparison_2),
        shuffle_label: colors.get(shuffle_key, '#949494'),
    }
    display_linestyles = {
        label_1: linestyles.get(condition_comparison_1, '-'),
        label_2: linestyles.get(condition_comparison_2, '-'),
        shuffle_label: linestyles.get(shuffle_key, '--'),
    }

    print(f"\n--- Running {condition_name} Comparison Statistics ({condition_comparison_1} vs {condition_comparison_2}) using '{args.unit_of_analysis}' as unit of analysis ---")
    
    for roi in rois:
        if roi not in all_bootstrap_stats.get(condition_comparison_1, {}):
            print(f"Skipping plot for ROI {roi} due to missing data.")
            continue

        # --- Pool the pooled shuffle distributions from each bootstrap ---
        pooled_shuffle_accs = []
        for b_idx in range(args.bootstraps):
            if b_idx in time_window_decoding_results:
                shuffle_data = time_window_decoding_results[b_idx][pooled_shuffle_key][roi]
                pooled_shuffle_accs.append(shuffle_data.T)

        # Stack all samples from all bootstraps
        stacked_pooled_shuffle_accs = np.vstack(pooled_shuffle_accs)
        
        # Store in master results
        if 'pooled_shuffles' not in master_results['stats']:
            master_results['stats']['pooled_shuffles'] = {}
        if roi not in master_results['stats']['pooled_shuffles']:
            master_results['stats']['pooled_shuffles'][roi] = {}
        master_results['stats']['pooled_shuffles'][roi][condition_name.lower()] = stacked_pooled_shuffle_accs

        # 1. Get the pooled data using existing helper function
        pooled_accs_1, pooled_accs_2 = get_pooled_accuracy_distributions_for_comparison(
            time_window_decoding_results=time_window_decoding_results,
            n_bootstraps=args.bootstraps,
            condition_comparison_1=condition_comparison_1,
            condition_comparison_2=condition_comparison_2,
            roi=roi,
            unit_of_analysis=args.unit_of_analysis
        )

        # 2. Run the paired cluster test
        sig_clusters_1_over_2, sig_clusters_2_over_1, _, _ = run_two_one_tailed_tests_with_time_perm_cluster(
            accuracies1=pooled_accs_1,
            accuracies2=pooled_accs_2,
            p_thresh=args.p_thresh_for_time_perm_cluster_stats,
            p_cluster=args.p_cluster,
            stat_func=args.stat_func,
            permutation_type=args.permutation_type,
            n_perm=args.n_cluster_perms,
            random_state=args.random_state,
            n_jobs=args.n_jobs
        )
        
        # The two directions of the contrast share the top row of the
        # significance strip. They cannot overlap in time, so a solid and a
        # dashed bar in the two conditions' colours tell them apart without
        # spending a second row.
        significance_clusters_comparison = {
            '1_over_2': {
                'clusters': sig_clusters_1_over_2,
                'label': significance_label_1,
                'color': colors[condition_comparison_1],
                'linestyle': '-',
                'kind': 'contrast',
            },
            '2_over_1': {
                'clusters': sig_clusters_2_over_1,
                'label': significance_label_2,
                'color': colors[condition_comparison_2],
                'linestyle': '--',
                'kind': 'contrast',
            }
        }

        if roi not in master_results['comparison_clusters']:
            master_results['comparison_clusters'][roi] = {}
        master_results['comparison_clusters'][roi][condition_name.lower()] = significance_clusters_comparison
    
        # --- Get data for plotting from the main stats dictionary ---
        stats_1 = all_bootstrap_stats[condition_comparison_1][roi]
        stats_2 = all_bootstrap_stats[condition_comparison_2][roi]
        
        unit = stats_1['unit_of_analysis']
        time_window_centers = time_window_decoding_results[0][condition_comparison_1][roi]['time_window_centers']
        
        # Each trace's own test against chance goes on its own row under the
        # contrast bars, so the figure says both "these two differ" and "each
        # one beats chance" without the reader having to open a second panel.
        significance_clusters_plot = dict(significance_clusters_comparison)
        significance_clusters_plot.update({
            f'{condition_comparison_1}_vs_chance': {
                'clusters': stats_1['significant_clusters'],
                'color': colors[condition_comparison_1],
                'kind': 'chance',
            },
            f'{condition_comparison_2}_vs_chance': {
                'clusters': stats_2['significant_clusters'],
                'color': colors[condition_comparison_2],
                'kind': 'chance',
            },
        })

        # Main comparison plot
        plot_accuracies_with_multiple_sig_clusters(
            time_points=time_window_centers,
            accuracies_dict={
                label_1: stats_1[f'{unit}_true_accs'],
                label_2: stats_2[f'{unit}_true_accs'],
                shuffle_label: stacked_pooled_shuffle_accs,
            },
            significance_clusters_dict=significance_clusters_plot,
            window_size=args.window_size,
            step_size=args.step_size,
            sampling_rate=args.sampling_rate,
            comparison_name=f'bootstrap_{condition_name}_comparison',
            roi=roi,
            save_dir=os.path.join(save_dir, f"{condition_name}_comparison", f"{roi}"),
            timestamp=args.timestamp,
            p_thresh=args.percentile,
            colors=display_colors,
            linestyles=display_linestyles,
            single_column=args.single_column,
            show_legend=args.show_legend,
            ylim=(0.3, 0.8),
            ylabel=ylabel,
            show_chance_level=False,
            title=_short_title(roi),
            filename_suffix=analysis_params_str,
            show_sig_legend=True,
        )

        # Difference plot
        print(f"Plotting accuracy DIFFERENCE for {condition_name} in {roi}...")
        
        differences = pooled_accs_1 - pooled_accs_2
        
        mean_diff = np.mean(differences, axis=0)
        std_diff = np.std(differences, axis=0)
        max_abs_val = np.max(np.abs(mean_diff) + std_diff)
        diff_ylim = (-max_abs_val * 1.2, max_abs_val * 1.2)
        if diff_ylim[0] == 0: 
            diff_ylim = (-0.1, 0.1)

        diff_key = f'{label_1} − {label_2}'

        plot_accuracies_with_multiple_sig_clusters(
            time_points=time_window_centers,
            accuracies_dict={diff_key: differences},
            significance_clusters_dict=significance_clusters_comparison,
            window_size=args.window_size,
            step_size=args.step_size,
            sampling_rate=args.sampling_rate,
            comparison_name=f'bootstrap_{condition_name}_ACC_DIFFERENCE_plot',
            roi=roi,
            save_dir=os.path.join(save_dir, f"{condition_name}_comparison", f"{roi}"),
            timestamp=args.timestamp,
            p_thresh=args.percentile,
            colors={diff_key: '#404040'},
            linestyles={diff_key: '-'},
            single_column=args.single_column,
            show_legend=args.show_legend,
            ylim=diff_ylim,
            ylabel='Accuracy difference',
            show_chance_level=True,
            chance_level=0,
            title=f'{_short_title(roi)}: {label_1} − {label_2}',
            filename_suffix=analysis_params_str + "_ACC_DIFFERENCE",
            show_sig_legend=True,
        )


def plot_cross_block_overlay(
    variable_name,
    block_comparisons,
    pooled_shuffle_key,
    colors,
    linestyles,
    ylabel,
    time_window_decoding_results,
    all_bootstrap_stats,
    master_results,         
    args,
    rois,
    save_dir,
    analysis_params_str,
    electrode_set_desc_fn=None
):
    """
    Overlay decoding accuracy from multiple blocks on a single plot,
    and store pooled shuffle data in master_results for later re-plotting.

    ``electrode_set_desc_fn`` : callable(roi) -> str, optional -- electrode-set
    description for the figure title.
    """
    from .anova_electrode_selection import decoding_figure_title

    print(f"\n📊 Generating cross-block {variable_name.upper()} overlay plots...")

    # Pool shuffle distributions across bootstraps
    pooled_shuffle_by_roi = {}
    for roi in rois:
        shuffle_accs = []
        for b_idx in range(args.bootstraps):
            if (b_idx in time_window_decoding_results and
                pooled_shuffle_key in time_window_decoding_results[b_idx] and
                roi in time_window_decoding_results[b_idx][pooled_shuffle_key]):
                shuffle_data = time_window_decoding_results[b_idx][pooled_shuffle_key][roi]
                shuffle_accs.append(shuffle_data.T)
        if shuffle_accs:
            pooled_shuffle_by_roi[roi] = np.vstack(shuffle_accs)

    # Store pooled shuffles in master_results for notebook re-plotting
    if 'pooled_shuffles' not in master_results['stats']:
        master_results['stats']['pooled_shuffles'] = {}
    for roi, shuffle_data in pooled_shuffle_by_roi.items():
        if roi not in master_results['stats']['pooled_shuffles']:
            master_results['stats']['pooled_shuffles'][roi] = {}
        master_results['stats']['pooled_shuffles'][roi][f'{variable_name}_cross_block'] = shuffle_data

    # Plot
    for roi in rois:
        accuracies_dict = {}
        for display_name, comp_key in block_comparisons.items():
            if roi in all_bootstrap_stats.get(comp_key, {}):
                unit = all_bootstrap_stats[comp_key][roi]['unit_of_analysis']
                accuracies_dict[display_name] = all_bootstrap_stats[comp_key][roi][f'{unit}_true_accs']

        if roi in pooled_shuffle_by_roi:
            accuracies_dict['Pooled shuffle'] = pooled_shuffle_by_roi[roi]

        if not accuracies_dict:
            print(f"  Skipping ROI {roi}: no data for {variable_name} cross-block plot.")
            continue

        first_comp_key = list(block_comparisons.values())[0]
        time_window_centers = time_window_decoding_results[0][first_comp_key][roi]['time_window_centers']

        plot_accuracies_nature_style(
            time_points=time_window_centers,
            accuracies_dict=accuracies_dict,
            significant_clusters=None,
            window_size=args.window_size,
            step_size=args.step_size,
            sampling_rate=args.sampling_rate,
            comparison_name=f'{variable_name}_decoding_across_blocks',
            roi=roi,
            save_dir=os.path.join(save_dir, f"cross_block_{variable_name}", roi),
            timestamp=args.timestamp,
            colors=colors,
            linestyles=linestyles,
            ylim=(0.3, 0.8),
            show_chance_level=False,
            show_legend=True,
            ylabel=ylabel,
            title=decoding_figure_title(
                f"{variable_name} decoding across blocks", roi,
                electrode_set_desc_fn(roi) if electrode_set_desc_fn else None),
            filename_suffix=analysis_params_str,
            single_column=False,
        )

    print(f"✅ Cross-block {variable_name} overlay plots complete.")
