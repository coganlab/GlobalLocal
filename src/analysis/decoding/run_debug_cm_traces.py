"""
Module for extracting and plotting pooled confusion matrix traces from bootstrap decoding results.
"""
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

from src.analysis.decoding.decoding import (
    extract_pooled_cm_traces,
    plot_cm_traces_nature_style,
)


def run_debug_cm_traces(
    time_window_decoding_results,
    condition_comparisons,
    rois,
    cats_by_roi,
    args,
    save_dir,
    analysis_params_str,
):
    """
    Extract and plot pooled CM traces for debugging.

    Parameters
    ----------
    time_window_decoding_results : dict
        Bootstrap-indexed dict of decoding results.
    condition_comparisons : dict
        Dict of condition comparisons to plot.
    rois : list
        List of ROI names.
    cats_by_roi : dict
        Category labels per ROI (from bootstrap results).
    args : namespace
        Arguments including bootstraps, unit_of_analysis, timestamp, single_column, etc.
    save_dir : str
        Base save directory.
    analysis_params_str : str
        String to append to filenames.
    """
    import os

    print("\n Extracting and plotting pooled CM traces for debugging...")

    # 1. Extract the pooled traces
    pooled_cm_traces = extract_pooled_cm_traces(
        time_window_decoding_results=time_window_decoding_results,
        n_bootstraps=args.bootstraps,
        condition_comparisons=condition_comparisons,
        rois=rois,
        unit_of_analysis=args.unit_of_analysis,
        cats_by_roi=cats_by_roi,
    )

    # 2. Plot the traces for each comparison and ROI
    for condition_comparison, roi_data in pooled_cm_traces.items():
        for roi, traces_dict in roi_data.items():
            if not traces_dict:
                continue

            time_window_centers = time_window_decoding_results[0][condition_comparison][roi]["time_window_centers"]

            # Infer the two class labels from the trace dictionary keys
            labels = set()
            for key in traces_dict.keys():
                try:
                    parts = key.split(",")
                    true_part = parts[0].replace("True: ", "").strip()
                    pred_part = parts[1].replace("Pred: ", "").strip()
                    labels.add(true_part)
                    labels.add(pred_part)
                except Exception:
                    continue

            if len(labels) != 2:
                print(
                    f"Skipping plot for {condition_comparison}/{roi}: "
                    f"expected 2 labels, got {len(labels)} ({labels})"
                )
                continue

            label1, label2 = sorted(list(labels))

            color_correct = "green"
            color_incorrect = "red"

            trace_colors = {
                f"True: {label1}, Pred: {label1}": color_correct,
                f"True: {label1}, Pred: {label2}": color_incorrect,
                f"True: {label2}, Pred: {label2}": color_correct,
                f"True: {label2}, Pred: {label1}": color_incorrect,
            }

            trace_linestyles = {
                f"True: {label1}, Pred: {label1}": "-",
                f"True: {label1}, Pred: {label2}": "--",
                f"True: {label2}, Pred: {label2}": "--",
                f"True: {label2}, Pred: {label1}": "-",
            }

            plot_cm_traces_nature_style(
                time_points=time_window_centers,
                cm_traces_dict=traces_dict,
                comparison_name=f"DEBUG_CM_Traces_{condition_comparison}",
                roi=roi,
                save_dir=os.path.join(save_dir, f"{condition_comparison}", f"{roi}"),
                timestamp=args.timestamp,
                colors=trace_colors,
                linestyles=trace_linestyles,
                single_column=args.single_column,
                show_legend=True,
                ylabel="Mean Trial Count",
                filename_suffix=analysis_params_str,
            )