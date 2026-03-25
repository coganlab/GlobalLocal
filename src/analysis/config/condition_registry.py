# src/analysis/config/condition_registry.py
"""
Single source of truth for all condition configurations used across:
- build_condition_comparisons() in general_utils.py
- pooled shuffle settings in process_bootstrap.py
- context comparison registry in run_context_comparisons.py

To add a new condition:
1. Define it in experiment_conditions (as usual)
2. Add a single entry here with whichever keys apply:
   - 'comparisons' (required): used by build_condition_comparisons
   - 'pooled_shuffle' (optional): used by process_bootstrap for pooled shuffle distributions
   - 'context_comparison' (optional): used by run_context_comparisons for true-vs-true overlays
"""

from src.analysis.config import experiment_conditions

CONDITION_REGISTRY = {

    # =========================================================================
    # 1. Basic Stimulus & Task Comparisons
    # =========================================================================

    'stimulus_conditions': {
        'conditions_obj': experiment_conditions.stimulus_conditions,
        'comparisons': {
            'BigLetter': ['bigS', 'bigH'],
            'SmallLetter': ['smallS', 'smallH'],
            'Task': ['taskG', 'taskL'],
        },
    },

    'stimulus_big_letter_conditions': {
        'conditions_obj': experiment_conditions.stimulus_big_letter_conditions,
        'comparisons': {
            'BigLetter': ['bigS', 'bigH'],
        },
    },

    'stimulus_small_letter_conditions': {
        'conditions_obj': experiment_conditions.stimulus_small_letter_conditions,
        'comparisons': {
            'SmallLetter': ['smallS', 'smallH'],
        },
    },

    'stimulus_task_conditions': {
        'conditions_obj': experiment_conditions.stimulus_task_conditions,
        'comparisons': {
            'Task': ['taskG', 'taskL'],
        },
    },

    # =========================================================================
    # 2. Congruency, Switch, and Error Comparisons
    # =========================================================================

    'stimulus_congruency_conditions': {
        'conditions_obj': experiment_conditions.stimulus_congruency_conditions,
        'comparisons': {
            'congruency': [['Stimulus_c'], ['Stimulus_i']],
        },
    },

    'stimulus_switch_type_conditions': {
        'conditions_obj': experiment_conditions.stimulus_switch_type_conditions,
        'comparisons': {
            'switchType': [['Stimulus_r'], ['Stimulus_s']],
        },
    },

    'stimulus_err_corr_conditions': {
        'conditions_obj': experiment_conditions.stimulus_err_corr_conditions,
        'comparisons': {
            'err_vs_corr': [['Stimulus_err'], ['Stimulus_corr']],
        },
    },

    'stimulus_iR_cS_err_conditions': {
        'conditions_obj': experiment_conditions.stimulus_iR_cS_err_conditions,
        'comparisons': {
            'iR_err_vs_cS_err': [['Stimulus_err_iR'], ['Stimulus_err_cS']],
        },
    },

    # =========================================================================
    # 3. LWPC / LWPS (List-Wide Proportions)
    # =========================================================================

    'stimulus_lwpc_conditions': {
        'conditions_obj': experiment_conditions.stimulus_lwpc_conditions,
        'comparisons': {
            'c25_vs_i25': ['c25', 'i25'],
            'c75_vs_i75': ['c75', 'i75'],
            'c25_vs_i75': ['c25', 'i75'],
            'c75_vs_i25': ['c75', 'i25'],
            'c25_vs_c75': ['c25', 'c75'],
            'i25_vs_i75': ['i25', 'i75'],
        },
        'pooled_shuffle': [
            {
                'key': 'lwpc_shuffle_accs_across_pooled_conditions',
                'strings_to_find': [['c25', 'c75'], ['i25', 'i75']],
            },
        ],
        'context_comparison': {
            'condition_name': 'LWPC',
            'condition_comparison_1': 'c25_vs_i25',
            'condition_comparison_2': 'c75_vs_i75',
            'pooled_shuffle_key': 'lwpc_shuffle_accs_across_pooled_conditions',
            'colors': {
                'c25_vs_i25': '#FF7E79',
                'c75_vs_i75': '#FF7E79',
                'lwpc_shuffle_accs_across_pooled_conditions_across_bootstraps': '#949494',
            },
            'linestyles': {
                'c25_vs_i25': '-',
                'c75_vs_i75': '--',
                'lwpc_shuffle_accs_across_pooled_conditions_across_bootstraps': '--',
            },
            'ylabel': 'Congruency Decoding Accuracy',
            'significance_label_1': '25% > 75% I',
            'significance_label_2': '75% > 25% I',
        },
    },

    'stimulus_lwps_conditions': {
        'conditions_obj': experiment_conditions.stimulus_lwps_conditions,
        'comparisons': {
            's25_vs_r25': ['s25', 'r25'],
            's75_vs_r75': ['s75', 'r75'],
            's25_vs_r75': ['s25', 'r75'],
            's75_vs_r25': ['s75', 'r25'],
            's25_vs_s75': ['s25', 's75'],
            'r25_vs_r75': ['r25', 'r75'],
        },
        'pooled_shuffle': [
            {
                'key': 'lwps_shuffle_accs_across_pooled_conditions',
                'strings_to_find': [['s25', 's75'], ['r25', 'r75']],
            },
        ],
        'context_comparison': {
            'condition_name': 'LWPS',
            'condition_comparison_1': 's25_vs_r25',
            'condition_comparison_2': 's75_vs_r75',
            'pooled_shuffle_key': 'lwps_shuffle_accs_across_pooled_conditions',
            'colors': {
                's25_vs_r25': '#05B0F0',
                's75_vs_r75': '#05B0F0',
                'lwps_shuffle_accs_across_pooled_conditions_across_bootstraps': '#949494',
            },
            'linestyles': {
                's25_vs_r25': '-',
                's75_vs_r75': '--',
                'lwps_shuffle_accs_across_pooled_conditions_across_bootstraps': '--',
            },
            'ylabel': 'Switch Type Decoding Accuracy',
            'significance_label_1': '25% > 75% S',
            'significance_label_2': '75% > 25% S',
        },
    },

    # =========================================================================
    # 4. Interaction: Task by Congruency / Switch
    # =========================================================================

    'stimulus_task_by_congruency_conditions': {
        'conditions_obj': experiment_conditions.stimulus_task_by_congruency_conditions,
        'comparisons': {
            'c_taskG_vs_c_taskL': ['Stimulus_c_taskG', 'Stimulus_c_taskL'],
            'i_taskG_vs_i_taskL': ['Stimulus_i_taskG', 'Stimulus_i_taskL'],
            'c_taskG_vs_i_taskG': ['Stimulus_c_taskG', 'Stimulus_i_taskG'],
            'c_taskL_vs_i_taskL': ['Stimulus_c_taskL', 'Stimulus_i_taskL'],
        },
        'pooled_shuffle': [
            {
                'key': 'task_by_congruency_shuffle_accs_across_pooled_conditions',
                'strings_to_find': [['taskG'], ['taskL']],
            },
        ],
        'context_comparison': {
            'condition_name': 'task_by_congruency',
            'condition_comparison_1': 'c_taskG_vs_c_taskL',
            'condition_comparison_2': 'i_taskG_vs_i_taskL',
            'pooled_shuffle_key': 'task_by_congruency_shuffle_accs_across_pooled_conditions',
            'colors': {
                'c_taskG_vs_c_taskL': '#05B0F0',
                'i_taskG_vs_i_taskL': '#05B0F0',
                'task_by_congruency_shuffle_accs_across_pooled_conditions_across_bootstraps': '#949494',
            },
            'linestyles': {
                'c_taskG_vs_c_taskL': '-',
                'i_taskG_vs_i_taskL': '--',
                'task_by_congruency_shuffle_accs_across_pooled_conditions_across_bootstraps': '--',
            },
            'ylabel': 'Task Decoding Accuracy',
            'significance_label_1': 'Task (C) > Task (I)',
            'significance_label_2': 'Task (I) > Task (C)',
        },
    },

    'stimulus_task_by_switch_type_conditions': {
        'conditions_obj': experiment_conditions.stimulus_task_by_switch_type_conditions,
        'comparisons': {
            'r_taskG_vs_r_taskL': ['Stimulus_r_taskG', 'Stimulus_r_taskL'],
            's_taskG_vs_s_taskL': ['Stimulus_s_taskG', 'Stimulus_s_taskL'],
            'r_taskG_vs_s_taskG': ['Stimulus_r_taskG', 'Stimulus_s_taskG'],
            'r_taskL_vs_s_taskL': ['Stimulus_r_taskL', 'Stimulus_s_taskL'],
        },
        'pooled_shuffle': [
            {
                'key': 'task_by_switch_type_shuffle_accs_across_pooled_conditions',
                'strings_to_find': [['taskG'], ['taskL']],
            },
        ],
        'context_comparison': {
            'condition_name': 'task_by_switch_type',
            'condition_comparison_1': 's_taskG_vs_s_taskL',
            'condition_comparison_2': 'r_taskG_vs_r_taskL',
            'pooled_shuffle_key': 'task_by_switch_type_shuffle_accs_across_pooled_conditions',
            'colors': {
                's_taskG_vs_s_taskL': '#05B0F0',
                'r_taskG_vs_r_taskL': '#05B0F0',
                'task_by_switch_type_shuffle_accs_across_pooled_conditions_across_bootstraps': '#949494',
            },
            'linestyles': {
                's_taskG_vs_s_taskL': '-',
                'r_taskG_vs_r_taskL': '--',
                'task_by_switch_type_shuffle_accs_across_pooled_conditions_across_bootstraps': '--',
            },
            'ylabel': 'Task Decoding Accuracy',
            'significance_label_1': 'Task (S) > Task (R)',
            'significance_label_2': 'Task (R) > Task (S)',
        },
    },

    # =========================================================================
    # 5. Proportion Interactions (Congruency/Switch by Block Proportion)
    # =========================================================================

    'stimulus_congruency_by_switch_proportion_conditions': {
        'conditions_obj': experiment_conditions.stimulus_congruency_by_switch_proportion_conditions,
        'comparisons': {
            'c_in_25switchBlock_vs_i_in_25switchBlock': ['Stimulus_c_in_25switchBlock', 'Stimulus_i_in_25switchBlock'],
            'c_in_75switchBlock_vs_i_in_75switchBlock': ['Stimulus_c_in_75switchBlock', 'Stimulus_i_in_75switchBlock'],
            'c_in_25switchBlock_vs_i_in_75switchBlock': ['Stimulus_c_in_25switchBlock', 'Stimulus_i_in_75switchBlock'],
            'c_in_75switchBlock_vs_i_in_25switchBlock': ['Stimulus_c_in_75switchBlock', 'Stimulus_i_in_25switchBlock'],
            'c_in_25switchBlock_vs_c_in_75switchBlock': ['Stimulus_c_in_25switchBlock', 'Stimulus_c_in_75switchBlock'],
            'i_in_25switchBlock_vs_i_in_75switchBlock': ['Stimulus_i_in_25switchBlock', 'Stimulus_i_in_75switchBlock'],
        },
        'pooled_shuffle': [
            {
                'key': 'congruency_by_switch_proportion_shuffle_accs_across_pooled_conditions',
                'strings_to_find': [['c_in'], ['i_in']],
            },
        ],
        'context_comparison': {
            'condition_name': 'congruency_by_switch_proportion',
            'condition_comparison_1': 'c_in_25switchBlock_vs_i_in_25switchBlock',
            'condition_comparison_2': 'c_in_75switchBlock_vs_i_in_75switchBlock',
            'pooled_shuffle_key': 'congruency_by_switch_proportion_shuffle_accs_across_pooled_conditions',
            'colors': {
                'c_in_25switchBlock_vs_i_in_25switchBlock': '#05B0F0',
                'c_in_75switchBlock_vs_i_in_75switchBlock': '#05B0F0',
                'congruency_by_switch_proportion_shuffle_accs_across_pooled_conditions_across_bootstraps': '#949494',
            },
            'linestyles': {
                'c_in_25switchBlock_vs_i_in_25switchBlock': '-',
                'c_in_75switchBlock_vs_i_in_75switchBlock': '--',
                'congruency_by_switch_proportion_shuffle_accs_across_pooled_conditions_across_bootstraps': '--',
            },
            'ylabel': 'Congruency Decoding Accuracy',
            'significance_label_1': 'C/I (25% S) > C/I (75% S)',
            'significance_label_2': 'C/I (75% S) > C/I (25% S)',
        },
    },

    'stimulus_switch_type_by_congruency_proportion_conditions': {
        'conditions_obj': experiment_conditions.stimulus_switch_type_by_congruency_proportion_conditions,
        'comparisons': {
            's_in_25incongruentBlock_vs_r_in_25incongruentBlock': ['Stimulus_s_in_25incongruentBlock', 'Stimulus_r_in_25incongruentBlock'],
            's_in_75incongruentBlock_vs_r_in_75incongruentBlock': ['Stimulus_s_in_75incongruentBlock', 'Stimulus_r_in_75incongruentBlock'],
            's_in_25incongruentBlock_vs_r_in_75incongruentBlock': ['Stimulus_s_in_25incongruentBlock', 'Stimulus_r_in_75incongruentBlock'],
            's_in_75incongruentBlock_vs_r_in_25incongruentBlock': ['Stimulus_s_in_75incongruentBlock', 'Stimulus_r_in_25incongruentBlock'],
            's_in_25incongruentBlock_vs_s_in_75incongruentBlock': ['Stimulus_s_in_25incongruentBlock', 'Stimulus_s_in_75incongruentBlock'],
            'r_in_25incongruentBlock_vs_r_in_75incongruentBlock': ['Stimulus_r_in_25incongruentBlock', 'Stimulus_r_in_75incongruentBlock'],
        },
        'pooled_shuffle': [
            {
                'key': 'switch_type_by_congruency_proportion_shuffle_accs_across_pooled_conditions',
                'strings_to_find': [['s_in'], ['r_in']],
            },
        ],
        'context_comparison': {
            'condition_name': 'switch_type_by_congruency_proportion',
            'condition_comparison_1': 's_in_25incongruentBlock_vs_r_in_25incongruentBlock',
            'condition_comparison_2': 's_in_75incongruentBlock_vs_r_in_75incongruentBlock',
            'pooled_shuffle_key': 'switch_type_by_congruency_proportion_shuffle_accs_across_pooled_conditions',
            'colors': {
                's_in_25incongruentBlock_vs_r_in_25incongruentBlock': '#FF7E79',
                's_in_75incongruentBlock_vs_r_in_75incongruentBlock': '#FF7E79',
                'switch_type_by_congruency_proportion_shuffle_accs_across_pooled_conditions_across_bootstraps': '#949494',
            },
            'linestyles': {
                's_in_25incongruentBlock_vs_r_in_25incongruentBlock': '-',
                's_in_75incongruentBlock_vs_r_in_75incongruentBlock': '--',
                'switch_type_by_congruency_proportion_shuffle_accs_across_pooled_conditions_across_bootstraps': '--',
            },
            'ylabel': 'Switch Type Decoding Accuracy',
            'significance_label_1': 'S/R (25% I) > S/R (75% I)',
            'significance_label_2': 'S/R (75% I) > S/R (25% I)',
        },
    },

    # =========================================================================
    # 6. Task by Proportion Blocks
    # =========================================================================

    'stimulus_task_by_congruency_proportion_conditions': {
        'conditions_obj': experiment_conditions.stimulus_task_by_congruency_proportion_conditions,
        'comparisons': {
            'taskG_in_25incongruentBlock_vs_taskG_in_75incongruentBlock': ['Stimulus_taskG_in_25incongruentBlock', 'Stimulus_taskG_in_75incongruentBlock'],
            'taskL_in_25incongruentBlock_vs_taskL_in_75incongruentBlock': ['Stimulus_taskL_in_25incongruentBlock', 'Stimulus_taskL_in_75incongruentBlock'],
            'taskG_in_25incongruentBlock_vs_taskL_in_25incongruentBlock': ['Stimulus_taskG_in_25incongruentBlock', 'Stimulus_taskL_in_25incongruentBlock'],
            'taskG_in_75incongruentBlock_vs_taskL_in_75incongruentBlock': ['Stimulus_taskG_in_75incongruentBlock', 'Stimulus_taskL_in_75incongruentBlock'],
        },
        'pooled_shuffle': [
            {
                'key': 'task_by_congruency_proportion_shuffle_accs_across_pooled_conditions',
                'strings_to_find': [['taskG'], ['taskL']],
            },
        ],
        'context_comparison': {
            'condition_name': 'task_by_congruency_proportion',
            'condition_comparison_1': 'taskG_in_25incongruentBlock_vs_taskL_in_25incongruentBlock',
            'condition_comparison_2': 'taskG_in_75incongruentBlock_vs_taskL_in_75incongruentBlock',
            'pooled_shuffle_key': 'task_by_congruency_proportion_shuffle_accs_across_pooled_conditions',
            'colors': {
                'taskG_in_25incongruentBlock_vs_taskL_in_25incongruentBlock': '#05B0F0',
                'taskG_in_75incongruentBlock_vs_taskL_in_75incongruentBlock': '#05B0F0',
                'task_by_congruency_proportion_shuffle_accs_across_pooled_conditions_across_bootstraps': '#949494',
            },
            'linestyles': {
                'taskG_in_25incongruentBlock_vs_taskL_in_25incongruentBlock': '-',
                'taskG_in_75incongruentBlock_vs_taskL_in_75incongruentBlock': '--',
                'task_by_congruency_proportion_shuffle_accs_across_pooled_conditions_across_bootstraps': '--',
            },
            'ylabel': 'Task Decoding Accuracy',
            'significance_label_1': 'Task (25% I) > Task (75% I)',
            'significance_label_2': 'Task (75% I) > Task (25% I)',
        },
    },

    'stimulus_task_by_switch_proportion_conditions': {
        'conditions_obj': experiment_conditions.stimulus_task_by_switch_proportion_conditions,
        'comparisons': {
            'taskG_in_25switchBlock_vs_taskG_in_75switchBlock': ['Stimulus_taskG_in_25switchBlock', 'Stimulus_taskG_in_75switchBlock'],
            'taskL_in_25switchBlock_vs_taskL_in_75switchBlock': ['Stimulus_taskL_in_25switchBlock', 'Stimulus_taskL_in_75switchBlock'],
            'taskG_in_25switchBlock_vs_taskL_in_25switchBlock': ['Stimulus_taskG_in_25switchBlock', 'Stimulus_taskL_in_25switchBlock'],
            'taskG_in_75switchBlock_vs_taskL_in_75switchBlock': ['Stimulus_taskG_in_75switchBlock', 'Stimulus_taskL_in_75switchBlock'],
        },
        'pooled_shuffle': [
            {
                'key': 'task_by_switch_proportion_shuffle_accs_across_pooled_conditions',
                'strings_to_find': [['taskG'], ['taskL']],
            },
        ],
        'context_comparison': {
            'condition_name': 'task_by_switch_proportion',
            'condition_comparison_1': 'taskG_in_25switchBlock_vs_taskL_in_25switchBlock',
            'condition_comparison_2': 'taskG_in_75switchBlock_vs_taskL_in_75switchBlock',
            'pooled_shuffle_key': 'task_by_switch_proportion_shuffle_accs_across_pooled_conditions',
            'colors': {
                'taskG_in_25switchBlock_vs_taskL_in_25switchBlock': '#05B0F0',
                'taskG_in_75switchBlock_vs_taskL_in_75switchBlock': '#05B0F0',
                'task_by_switch_proportion_shuffle_accs_across_pooled_conditions_across_bootstraps': '#949494',
            },
            'linestyles': {
                'taskG_in_25switchBlock_vs_taskL_in_25switchBlock': '-',
                'taskG_in_75switchBlock_vs_taskL_in_75switchBlock': '--',
                'task_by_switch_proportion_shuffle_accs_across_pooled_conditions_across_bootstraps': '--',
            },
            'ylabel': 'Task Decoding Accuracy',
            'significance_label_1': 'Task (25% S) > Task (75% S)',
            'significance_label_2': 'Task (75% S) > Task (25% S)',
        },
    },

    # =========================================================================
    # 7. Full Experiment Conditions (multiple pooled shuffles, no context comparison)
    # =========================================================================

    'stimulus_experiment_conditions': {
        'conditions_obj': experiment_conditions.stimulus_experiment_conditions,
        'comparisons': {
            # NOTE: This entry exists in get_conditions_save_name but build_condition_comparisons
            # does not have an entry for it. Add comparisons here if/when needed.
        },
        'pooled_shuffle': [
            {
                'key': 'congruency_pooled_shuffle',
                'strings_to_find': [['c25', 'c75'], ['i25', 'i75']],
            },
            {
                'key': 'switchType_pooled_shuffle',
                'strings_to_find': [['s25', 's75'], ['r25', 'r75']],
            },
        ],
    },

    # =========================================================================
    # 8. Block-Specific Congruency (A, B, C, D)
    # =========================================================================

    'stimulus_congruency_blockA_conditions': {
        'conditions_obj': experiment_conditions.stimulus_congruency_blockA_conditions,
        'comparisons': {
            'Stimulus_c_blockA_vs_Stimulus_i_blockA': ['Stimulus_c_blockA', 'Stimulus_i_blockA'],
        },
        'pooled_shuffle': [
            {
                'key': 'congruency_blockA_shuffle',
                'strings_to_find': [['Stimulus_c'], ['Stimulus_i']],
            },
        ],
    },

    'stimulus_congruency_blockB_conditions': {
        'conditions_obj': experiment_conditions.stimulus_congruency_blockB_conditions,
        'comparisons': {
            'Stimulus_c_blockB_vs_Stimulus_i_blockB': ['Stimulus_c_blockB', 'Stimulus_i_blockB'],
        },
        'pooled_shuffle': [
            {
                'key': 'congruency_blockB_shuffle',
                'strings_to_find': [['Stimulus_c'], ['Stimulus_i']],
            },
        ],
    },

    'stimulus_congruency_blockC_conditions': {
        'conditions_obj': experiment_conditions.stimulus_congruency_blockC_conditions,
        'comparisons': {
            'Stimulus_c_blockC_vs_Stimulus_i_blockC': ['Stimulus_c_blockC', 'Stimulus_i_blockC'],
        },
        'pooled_shuffle': [
            {
                'key': 'congruency_blockC_shuffle',
                'strings_to_find': [['Stimulus_c'], ['Stimulus_i']],
            },
        ],
    },

    'stimulus_congruency_blockD_conditions': {
        'conditions_obj': experiment_conditions.stimulus_congruency_blockD_conditions,
        'comparisons': {
            'Stimulus_c_blockD_vs_Stimulus_i_blockD': ['Stimulus_c_blockD', 'Stimulus_i_blockD'],
        },
        'pooled_shuffle': [
            {
                'key': 'congruency_blockD_shuffle',
                'strings_to_find': [['Stimulus_c'], ['Stimulus_i']],
            },
        ],
    },

    # =========================================================================
    # 9. Block-Specific Switch Type (A, B, C, D)
    # =========================================================================

    'stimulus_switchType_blockA_conditions': {
        'conditions_obj': experiment_conditions.stimulus_switchType_blockA_conditions,
        'comparisons': {
            'Stimulus_s_blockA_vs_Stimulus_r_blockA': ['Stimulus_s_blockA', 'Stimulus_r_blockA'],
        },
        'pooled_shuffle': [
            {
                'key': 'switchType_blockA_shuffle',
                'strings_to_find': [['Stimulus_s'], ['Stimulus_r']],
            },
        ],
    },

    'stimulus_switchType_blockB_conditions': {
        'conditions_obj': experiment_conditions.stimulus_switchType_blockB_conditions,
        'comparisons': {
            'Stimulus_s_blockB_vs_Stimulus_r_blockB': ['Stimulus_s_blockB', 'Stimulus_r_blockB'],
        },
        'pooled_shuffle': [
            {
                'key': 'switchType_blockB_shuffle',
                'strings_to_find': [['Stimulus_s'], ['Stimulus_r']],
            },
        ],
    },

    'stimulus_switchType_blockC_conditions': {
        'conditions_obj': experiment_conditions.stimulus_switchType_blockC_conditions,
        'comparisons': {
            'Stimulus_s_blockC_vs_Stimulus_r_blockC': ['Stimulus_s_blockC', 'Stimulus_r_blockC'],
        },
        'pooled_shuffle': [
            {
                'key': 'switchType_blockC_shuffle',
                'strings_to_find': [['Stimulus_s'], ['Stimulus_r']],
            },
        ],
    },

    'stimulus_switchType_blockD_conditions': {
        'conditions_obj': experiment_conditions.stimulus_switchType_blockD_conditions,
        'comparisons': {
            'Stimulus_s_blockD_vs_Stimulus_r_blockD': ['Stimulus_s_blockD', 'Stimulus_r_blockD'],
        },
        'pooled_shuffle': [
            {
                'key': 'switchType_blockD_shuffle',
                'strings_to_find': [['Stimulus_s'], ['Stimulus_r']],
            },
        ],
    },
    
    # =========================================================================
    # 10. Main Effects (just names, no comparisons)
    # =========================================================================
    'stimulus_main_effect_conditions': {
        'conditions_obj': experiment_conditions.stimulus_main_effect_conditions,
        'comparisons': {},
    },
    'response_conditions': {
        'conditions_obj': experiment_conditions.response_conditions,
        'comparisons': {},
    },
    'response_experiment_conditions': {
        'conditions_obj': experiment_conditions.response_experiment_conditions,
        'comparisons': {},
    },
    'response_big_letter_conditions': {
        'conditions_obj': experiment_conditions.response_big_letter_conditions,
        'comparisons': {},
    },
    'response_small_letter_conditions': {
        'conditions_obj': experiment_conditions.response_small_letter_conditions,
        'comparisons': {},
    },
    'response_task_conditions': {
        'conditions_obj': experiment_conditions.response_task_conditions,
        'comparisons': {},
    },
    'response_congruency_conditions': {
        'conditions_obj': experiment_conditions.response_congruency_conditions,
        'comparisons': {},
    },
    'response_switch_type_conditions': {
        'conditions_obj': experiment_conditions.response_switch_type_conditions,
        'comparisons': {},
    },
    'response_err_corr_conditions': {
        'conditions_obj': experiment_conditions.response_err_corr_conditions,
        'comparisons': {
            'err_vs_corr': [['Response_err'], ['Response_corr']],
        },
    },
    'response_congruency_by_switch_proportion_conditions': {
        'conditions_obj': experiment_conditions.response_congruency_by_switch_proportion_conditions,
        'comparisons': {},
    },
    'response_switch_type_by_congruency_proportion_conditions': {
        'conditions_obj': experiment_conditions.response_switch_type_by_congruency_proportion_conditions,
        'comparisons': {},
    },
    'response_iR_cS_err_conditions': {
        'conditions_obj': experiment_conditions.response_iR_cS_err_conditions,
        'comparisons': {},
    }

}



# =============================================================================
# Helper functions for consumers
# =============================================================================

def _find_entry(conditions):
    """Look up the registry entry matching a conditions object."""
    for entry in CONDITION_REGISTRY.values():
        if conditions == entry['conditions_obj']:
            return entry
    return None


def get_comparisons(conditions):
    """
    Return the comparisons dict for a given conditions object.
    Replaces build_condition_comparisons().
    """
    entry = _find_entry(conditions)
    if entry is None:
        raise ValueError(f"No comparisons defined for {conditions}")
    return entry['comparisons']


def get_pooled_shuffle_settings(conditions):
    """
    Return a list of (key, strings_to_find) tuples for pooled shuffle.
    Replaces the if/elif chain in process_bootstrap.py.
    """
    entry = _find_entry(conditions)
    if entry is None:
        return []
    return [
        (s['key'], s['strings_to_find'])
        for s in entry.get('pooled_shuffle', [])
    ]


def get_context_comparison_kwargs(conditions):
    """
    Return the kwargs dict for run_context_comparison_analysis, or None.
    Replaces CONTEXT_COMPARISON_REGISTRY in run_context_comparisons.py.
    """
    entry = _find_entry(conditions)
    if entry is None:
        return None
    return entry.get('context_comparison', None)