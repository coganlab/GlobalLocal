"""Facade for the decoding package.

Historically this module held the entire decoding pipeline. It has been split
into focused submodules (see docs/code_maintenance.md#refactoring-guide). This file now just
re-exports the public names so existing
`from src.analysis.decoding.decoding import ...` imports keep working.
"""

from .data_prep import (
    concatenate_and_balance_data_for_decoding,
    mixup2,
    flatten_features,
    sample_fold,
)
from .decoder import (
    Decoder,
)
from .accuracy_stats import (
    compute_accuracies,
    perform_time_perm_cluster_test_for_accuracies,
    find_contiguous_clusters,
    make_pooled_shuffle_distribution,
    find_significant_clusters_of_series_vs_distribution_based_on_percentile,
    find_cluster_lengths,
    get_max_perm_cluster_lengths_based_on_percentile,
    compute_pooled_bootstrap_statistics,
    do_time_perm_cluster_comparing_two_true_bootstrap_accuracy_distributions_for_one_roi,
    get_pooled_accuracy_distributions_for_comparison,
    do_time_perm_cluster_comparing_two_true_bootstrap_accuracy_distributions,
    do_mne_paired_cluster_test,
    get_time_averaged_confusion_matrix,
    _run_single_permutation,
    cluster_perm_paired_ttest_by_duration,
    run_two_one_tailed_tests_with_time_perm_cluster,
)
from .tfr_cluster import (
    decode_on_sig_tfr_clusters,
    compute_sig_tfr_masks_from_roi_labeled_array,
    compute_sig_tfr_masks_for_specified_channels,
    compute_sig_tfr_masks_from_concatenated_data,
    apply_tfr_masks_and_flatten_to_make_decoding_matrix,
    get_confusion_matrix_for_rois_tfr_cluster,
)
from .roi_confusion import (
    get_and_plot_confusion_matrix_for_rois_jim,
    get_confusion_matrices_for_rois_time_window_decoding_jim,
)
from .plots.confusion import (
    get_display_labels_from_cats,
    plot_and_save_confusion_matrix,
    plot_and_save_tfr_masks,
    extract_pooled_cm_traces,
    plot_cm_traces_nature_style,
)
from .plots.accuracies import (
    plot_accuracies_nature_style,
    create_multipanel_nature_figure,
    plot_true_vs_shuffle_accuracies,
    plot_accuracies_with_multiple_sig_clusters,
)
from .plots.trajectories import (
    plot_static_pca_projection,
    plot_pca_over_time,
    plot_pca_3d_trajectory,
    plot_high_dim_decision_slice,
    plot_static_umap_projection,
    plot_umap_3d_trajectory,
)
from .context_comparison import (
    run_context_comparison_analysis,
    plot_cross_block_overlay,
)
from .plots.style import NATURE_STYLE
from src.analysis.utils.general_utils import windower

__all__ = [
    "concatenate_and_balance_data_for_decoding",
    "mixup2",
    "flatten_features",
    "sample_fold",
    "Decoder",
    "compute_accuracies",
    "perform_time_perm_cluster_test_for_accuracies",
    "find_contiguous_clusters",
    "find_cluster_lengths",
    "make_pooled_shuffle_distribution",
    "find_significant_clusters_of_series_vs_distribution_based_on_percentile",
    "get_max_perm_cluster_lengths_based_on_percentile",
    "compute_pooled_bootstrap_statistics",
    "do_time_perm_cluster_comparing_two_true_bootstrap_accuracy_distributions_for_one_roi",
    "get_pooled_accuracy_distributions_for_comparison",
    "do_time_perm_cluster_comparing_two_true_bootstrap_accuracy_distributions",
    "do_mne_paired_cluster_test",
    "get_time_averaged_confusion_matrix",
    "_run_single_permutation",
    "cluster_perm_paired_ttest_by_duration",
    "run_two_one_tailed_tests_with_time_perm_cluster",
    "decode_on_sig_tfr_clusters",
    "compute_sig_tfr_masks_from_roi_labeled_array",
    "compute_sig_tfr_masks_for_specified_channels",
    "compute_sig_tfr_masks_from_concatenated_data",
    "apply_tfr_masks_and_flatten_to_make_decoding_matrix",
    "get_confusion_matrix_for_rois_tfr_cluster",
    "get_and_plot_confusion_matrix_for_rois_jim",
    "get_confusion_matrices_for_rois_time_window_decoding_jim",
    "get_display_labels_from_cats",
    "plot_and_save_confusion_matrix",
    "plot_and_save_tfr_masks",
    "extract_pooled_cm_traces",
    "plot_cm_traces_nature_style",
    "plot_accuracies_nature_style",
    "create_multipanel_nature_figure",
    "plot_true_vs_shuffle_accuracies",
    "plot_accuracies_with_multiple_sig_clusters",
    "plot_static_pca_projection",
    "plot_pca_over_time",
    "plot_pca_3d_trajectory",
    "plot_high_dim_decision_slice",
    "plot_static_umap_projection",
    "plot_umap_3d_trajectory",
    "run_context_comparison_analysis",
    "plot_cross_block_overlay",
    "NATURE_STYLE",
    "windower",
]
