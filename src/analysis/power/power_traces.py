"""Facade for the power-traces package.

Historically this module held the entire power-trace pipeline (evoked building,
windowed ANOVA, and plotting) in one ~2,400-line file. It has been split into
focused submodules (see docs/refactoring_guide.md §5). This file now just
re-exports the public names so existing
``from src.analysis.power.power_traces import ...`` imports keep working.

The old ``sys.path`` juggling at the top of this file is gone — run
``pip install -e .`` once so ``src.analysis...`` is importable everywhere
(§7 of the refactoring guide).
"""

from .evoked_builders import (
    combine_single_channel_evokeds,
    get_subject_electrodes_for_roi,
    get_evoked_for_specific_subject_and_condition,
    extract_single_electrode_evokeds,
    create_list_of_single_channel_evokeds_across_subjects_for_roi_and_condition,
    make_evoked_electrode_lists_for_rois,
    make_evoked_electrode_lists_for_all_conditions_and_rois,
    make_multi_channel_evokeds_for_all_conditions_and_rois,
    create_roi_grand_average,
    subtract_evoked_conditions,
    create_subtracted_evokeds_dict,
    time_perm_cluster_between_two_evokeds,
)
from .windowed_anova import (
    run_within_electrode_windowed_anova_cluster_correction,
    load_significant_electrodes,
    process_windowed_data_for_anova,
    create_windowed_anova_dataframe,
    perform_windowed_anova,
    apply_fdr_correction_to_windowed_results,
    run_windowed_anova_cluster_correction,
)
from .plots import (
    DEFAULT_PLOT_STYLE,
    DIRECTION_TEST_SUBDIR,
    plot_power_trace_for_roi,
    plot_power_traces_for_all_rois,
    direction_test_save_name,
    plot_direction_test_traces,
    apply_plot_style,
    compute_subcell_evoked_data,
    plot_2way_interaction_for_roi,
    plot_16_conditions_with_interaction_clusters_for_roi,
    plot_anova_interaction_results,
    anova_results_to_interaction_results_for_plotting,
)

__all__ = [
    # evoked_builders
    "combine_single_channel_evokeds",
    "get_subject_electrodes_for_roi",
    "get_evoked_for_specific_subject_and_condition",
    "extract_single_electrode_evokeds",
    "create_list_of_single_channel_evokeds_across_subjects_for_roi_and_condition",
    "make_evoked_electrode_lists_for_rois",
    "make_evoked_electrode_lists_for_all_conditions_and_rois",
    "make_multi_channel_evokeds_for_all_conditions_and_rois",
    "create_roi_grand_average",
    "subtract_evoked_conditions",
    "create_subtracted_evokeds_dict",
    "time_perm_cluster_between_two_evokeds",
    # windowed_anova
    "run_within_electrode_windowed_anova_cluster_correction",
    "load_significant_electrodes",
    "process_windowed_data_for_anova",
    "create_windowed_anova_dataframe",
    "perform_windowed_anova",
    "apply_fdr_correction_to_windowed_results",
    "run_windowed_anova_cluster_correction",
    # plots
    "DEFAULT_PLOT_STYLE",
    "DIRECTION_TEST_SUBDIR",
    "plot_power_trace_for_roi",
    "plot_power_traces_for_all_rois",
    "direction_test_save_name",
    "plot_direction_test_traces",
    "apply_plot_style",
    "compute_subcell_evoked_data",
    "plot_2way_interaction_for_roi",
    "plot_16_conditions_with_interaction_clusters_for_roi",
    "plot_anova_interaction_results",
    "anova_results_to_interaction_results_for_plotting",
]
