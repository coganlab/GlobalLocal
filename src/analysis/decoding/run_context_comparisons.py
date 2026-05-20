"""
Module that dispatches run_context_comparison_analysis calls based on which
experiment condition was selected at runtime.
"""
import sys
import os

try:
    current_file_path = os.path.abspath(__file__)
    current_script_dir = os.path.dirname(current_file_path)
except NameError:
    current_script_dir = os.getcwd()

project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..', '..'))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.analysis.config.condition_registry import get_context_comparison_kwargs
from src.analysis.decoding.decoding import run_context_comparison_analysis

def run_all_context_comparisons(
    args,
    time_window_decoding_results,
    all_bootstrap_stats,
    master_results,
    rois,
    save_dir,
    analysis_params_str,
):
    kwargs = get_context_comparison_kwargs(args.condition_label)
    if kwargs is not None:
        run_context_comparison_analysis(
            **kwargs,
            time_window_decoding_results=time_window_decoding_results,
            all_bootstrap_stats=all_bootstrap_stats,
            master_results=master_results,
            args=args,
            rois=rois,
            save_dir=save_dir,
            analysis_params_str=analysis_params_str,
        )