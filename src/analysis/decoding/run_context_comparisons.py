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

from src.analysis.config import experiment_conditions
from src.analysis.decoding.decoding import run_context_comparison_analysis


# Each entry maps an experiment_conditions attribute to the kwargs for
# run_context_comparison_analysis (minus the shared args passed at call time).
CONTEXT_COMPARISON_REGISTRY = [
    {
        'condition_attr': 'stimulus_lwpc_conditions',
        'kwargs': dict(
            condition_name='LWPC',
            condition_comparison_1='c25_vs_i25',
            condition_comparison_2='c75_vs_i75',
            pooled_shuffle_key='lwpc_shuffle_accs_across_pooled_conditions',
            colors={
                'c25_vs_i25': '#FF7E79',
                'c75_vs_i75': '#FF7E79',
                'lwpc_shuffle_accs_across_pooled_conditions_across_bootstraps': '#949494',
            },
            linestyles={
                'c25_vs_i25': '-',
                'c75_vs_i75': '--',
                'lwpc_shuffle_accs_across_pooled_conditions_across_bootstraps': '--',
            },
            ylabel='Congruency Decoding Accuracy',
            significance_label_1='25% > 75% I',
            significance_label_2='75% > 25% I',
        ),
    },
    {
        'condition_attr': 'stimulus_lwps_conditions',
        'kwargs': dict(
            condition_name='LWPS',
            condition_comparison_1='s25_vs_r25',
            condition_comparison_2='s75_vs_r75',
            pooled_shuffle_key='lwps_shuffle_accs_across_pooled_conditions',
            colors={
                's25_vs_r25': '#05B0F0',
                's75_vs_r75': '#05B0F0',
                'lwps_shuffle_accs_across_pooled_conditions_across_bootstraps': '#949494',
            },
            linestyles={
                's25_vs_r25': '-',
                's75_vs_r75': '--',
                'lwps_shuffle_accs_across_pooled_conditions_across_bootstraps': '--',
            },
            ylabel='Switch Type Decoding Accuracy',
            significance_label_1='25% > 75% S',
            significance_label_2='75% > 25% S',
        ),
    },
    {
        'condition_attr': 'stimulus_congruency_by_switch_proportion_conditions',
        'kwargs': dict(
            condition_name='congruency_by_switch_proportion',
            condition_comparison_1='c_in_25switchBlock_vs_i_in_25switchBlock',
            condition_comparison_2='c_in_75switchBlock_vs_i_in_75switchBlock',
            pooled_shuffle_key='congruency_by_switch_proportion_shuffle_accs_across_pooled_conditions',
            colors={
                'c_in_25switchBlock_vs_i_in_25switchBlock': '#05B0F0',
                'c_in_75switchBlock_vs_i_in_75switchBlock': '#05B0F0',
                'congruency_by_switch_proportion_shuffle_accs_across_pooled_conditions_across_bootstraps': '#949494',
            },
            linestyles={
                'c_in_25switchBlock_vs_i_in_25switchBlock': '-',
                'c_in_75switchBlock_vs_i_in_75switchBlock': '--',
                'congruency_by_switch_proportion_shuffle_accs_across_pooled_conditions_across_bootstraps': '--',
            },
            ylabel='Congruency Decoding Accuracy',
            significance_label_1='C/I (25% S) > C/I (75% S)',
            significance_label_2='C/I (75% S) > C/I (25% S)',
        ),
    },
    {
        'condition_attr': 'stimulus_switch_type_by_congruency_proportion_conditions',
        'kwargs': dict(
            condition_name='switch_type_by_congruency_proportion',
            condition_comparison_1='s_in_25incongruentBlock_vs_r_in_25incongruentBlock',
            condition_comparison_2='s_in_75incongruentBlock_vs_r_in_75incongruentBlock',
            pooled_shuffle_key='switch_type_by_congruency_proportion_shuffle_accs_across_pooled_conditions',
            colors={
                's_in_25incongruentBlock_vs_r_in_25incongruentBlock': '#FF7E79',
                's_in_75incongruentBlock_vs_r_in_75incongruentBlock': '#FF7E79',
                'switch_type_by_congruency_proportion_shuffle_accs_across_pooled_conditions_across_bootstraps': '#949494',
            },
            linestyles={
                's_in_25incongruentBlock_vs_r_in_25incongruentBlock': '-',
                's_in_75incongruentBlock_vs_r_in_75incongruentBlock': '--',
                'switch_type_by_congruency_proportion_shuffle_accs_across_pooled_conditions_across_bootstraps': '--',
            },
            ylabel='Switch Type Decoding Accuracy',
            significance_label_1='S/R (25% I) > S/R (75% I)',
            significance_label_2='S/R (75% I) > S/R (25% I)',
        ),
    },
    {
        'condition_attr': 'stimulus_task_by_congruency_conditions',
        'kwargs': dict(
            condition_name='task_by_congruency',
            condition_comparison_1='c_taskG_vs_c_taskL',
            condition_comparison_2='i_taskG_vs_i_taskL',
            pooled_shuffle_key='task_by_congruency_shuffle_accs_across_pooled_conditions',
            colors={
                'c_taskG_vs_c_taskL': '#05B0F0',
                'i_taskG_vs_i_taskL': '#05B0F0',
                'task_by_congruency_shuffle_accs_across_pooled_conditions_across_bootstraps': '#949494',
            },
            linestyles={
                'c_taskG_vs_c_taskL': '-',
                'i_taskG_vs_i_taskL': '--',
                'task_by_congruency_shuffle_accs_across_pooled_conditions_across_bootstraps': '--',
            },
            ylabel='Task Decoding Accuracy',
            significance_label_1='Task (C) > Task (I)',
            significance_label_2='Task (I) > Task (C)',
        ),
    },
    {
        'condition_attr': 'stimulus_task_by_switch_type_conditions',
        'kwargs': dict(
            condition_name='task_by_switch_type',
            condition_comparison_1='s_taskG_vs_s_taskL',
            condition_comparison_2='r_taskG_vs_r_taskL',
            pooled_shuffle_key='task_by_switch_type_shuffle_accs_across_pooled_conditions',
            colors={
                's_taskG_vs_s_taskL': '#05B0F0',
                'r_taskG_vs_r_taskL': '#05B0F0',
                'task_by_switch_type_shuffle_accs_across_pooled_conditions_across_bootstraps': '#949494',
            },
            linestyles={
                's_taskG_vs_s_taskL': '-',
                'r_taskG_vs_r_taskL': '--',
                'task_by_switch_type_shuffle_accs_across_pooled_conditions_across_bootstraps': '--',
            },
            ylabel='Task Decoding Accuracy',
            significance_label_1='Task (S) > Task (R)',
            significance_label_2='Task (R) > Task (S)',
        ),
    },
    {
        'condition_attr': 'stimulus_task_by_congruency_proportion_conditions',
        'kwargs': dict(
            condition_name='task_by_congruency_proportion',
            condition_comparison_1='taskG_in_25incongruentBlock_vs_taskL_in_25incongruentBlock',
            condition_comparison_2='taskG_in_75incongruentBlock_vs_taskL_in_75incongruentBlock',
            pooled_shuffle_key='task_by_congruency_proportion_shuffle_accs_across_pooled_conditions',
            colors={
                'taskG_in_25incongruentBlock_vs_taskL_in_25incongruentBlock': '#05B0F0',
                'taskG_in_75incongruentBlock_vs_taskL_in_75incongruentBlock': '#05B0F0',
                'task_by_congruency_proportion_shuffle_accs_across_pooled_conditions_across_bootstraps': '#949494',
            },
            linestyles={
                'taskG_in_25incongruentBlock_vs_taskL_in_25incongruentBlock': '-',
                'taskG_in_75incongruentBlock_vs_taskL_in_75incongruentBlock': '--',
                'task_by_congruency_proportion_shuffle_accs_across_pooled_conditions_across_bootstraps': '--',
            },
            ylabel='Task Decoding Accuracy',
            significance_label_1='Task (25% I) > Task (75% I)',
            significance_label_2='Task (75% I) > Task (25% I)',
        ),
    },
    {
        'condition_attr': 'stimulus_task_by_switch_proportion_conditions',
        'kwargs': dict(
            condition_name='task_by_switch_proportion',
            condition_comparison_1='taskG_in_25switchBlock_vs_taskL_in_25switchBlock',
            condition_comparison_2='taskG_in_75switchBlock_vs_taskL_in_75switchBlock',
            pooled_shuffle_key='task_by_switch_proportion_shuffle_accs_across_pooled_conditions',
            colors={
                'taskG_in_25switchBlock_vs_taskL_in_25switchBlock': '#05B0F0',
                'taskG_in_75switchBlock_vs_taskL_in_75switchBlock': '#05B0F0',
                'task_by_switch_proportion_shuffle_accs_across_pooled_conditions_across_bootstraps': '#949494',
            },
            linestyles={
                'taskG_in_25switchBlock_vs_taskL_in_25switchBlock': '-',
                'taskG_in_75switchBlock_vs_taskL_in_75switchBlock': '--',
                'task_by_switch_proportion_shuffle_accs_across_pooled_conditions_across_bootstraps': '--',
            },
            ylabel='Task Decoding Accuracy',
            significance_label_1='Task (25% S) > Task (75% S)',
            significance_label_2='Task (75% S) > Task (25% S)',
        ),
    },
]


def run_all_context_comparisons(
    args,
    time_window_decoding_results,
    all_bootstrap_stats,
    master_results,
    rois,
    save_dir,
    analysis_params_str,
):
    """
    Loop through the registry and run whichever context comparison matches
    the current args.conditions.
    """
    for entry in CONTEXT_COMPARISON_REGISTRY:
        expected_conditions = getattr(experiment_conditions, entry['condition_attr'])
        if args.conditions == expected_conditions:
            run_context_comparison_analysis(
                **entry['kwargs'],
                time_window_decoding_results=time_window_decoding_results,
                all_bootstrap_stats=all_bootstrap_stats,
                master_results=master_results,
                args=args,
                rois=rois,
                save_dir=save_dir,
                analysis_params_str=analysis_params_str,
            )