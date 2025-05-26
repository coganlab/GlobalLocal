# %%
# this is an ongoing attempt at refactoring roi_analysis.ipynb 4/28. Perhaps split this into multiple scripts.
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import sys
import os
print(sys.path)

# Add parent directory to path to access modules in project root
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

# Path to IEEG_Pipelines
sys.path.append("C:/Users/jz421/Desktop/GlobalLocal/IEEG_Pipelines/") #need to do this cuz otherwise ieeg isn't added to path...

from ieeg.navigate import channel_outlier_marker, trial_ieeg, crop_empty_data, \
    outliers_to_nan
from ieeg.io import raw_from_layout, get_data
from ieeg.timefreq.utils import crop_pad
from ieeg.timefreq import gamma
from ieeg.calc.scaling import rescale
import mne
import os
import numpy as np
from ieeg.calc.reshape import make_data_same
from ieeg.calc.stats import time_perm_cluster, window_averaged_shuffle
from ieeg.viz.mri import gen_labels

# Import utils from the project root - we added the parent directory to sys.path above
from utils import make_subjects_electrodestoROIs_dict, load_subjects_electrodestoROIs_dict, load_acc_arrays, calculate_RTs, save_channels_to_file, save_sig_chans, \
      load_sig_chans, channel_names_to_indices, filter_and_average_epochs, permutation_test, perform_permutation_test_across_electrodes, perform_permutation_test_within_electrodes, \
      add_accuracy_to_epochs, load_mne_objects, create_subjects_mne_objects_dict, extract_significant_effects, convert_dataframe_to_serializable_format, \
      perform_modular_anova, make_plotting_parameters, plot_significance

import matplotlib.pyplot as plt
from collections import OrderedDict, defaultdict
import json
# still need to test if the permutation test functions load in properly.
import pandas as pd
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

# %% [markdown]
# choose which subjects you wanna run (has to be a list, even if just one subject)

# %%
subjects = ['D0057','D0059', 'D0063', 'D0065', 'D0069', 'D0071', 'D0077', 'D0090', 'D0094', 'D0100', 'D0102', 'D0103', 'D0107A', 'D0110']

# %% [markdown]
# ### make or load subjects electrodes to rois dict

# %%
# load in subjects electrodes to rois dict. If it doesn't already exist, make it and then load it.
filename = 'subjects_electrodestoROIs_dict.json'
subjects_electrodestoROIs_dict = utils.make_or_load_subjects_electrodes_to_rois_dict(filename, subjects)

# old code below as of 5/5
# subjects_electrodestoROIs_dict = utils.load_subjects_electrodestoROIs_dict(filename)

# if subjects_electrodestoROIs_dict is None:
#     utils.make_subjects_electrodestoROIs_dict(subjects)
#     subjects_electrodestoROIs_dict = utils.load_subjects_electrodestoROIs_dict(filename)

# %%
subjects_electrodestoROIs_dict

# %%
def count_electrodes_across_subjects(data):
    total_electrodes = 0
    for subject, details in data.items():
        total_electrodes += len(details['default_dict'])
    return total_electrodes

# Calculate the total number of electrodes in the 'default_dict' across subjects
total_electrodes = count_electrodes_across_subjects(subjects_electrodestoROIs_dict)
total_electrodes

# %% [markdown]
# load behavioral data (combinedData.csv)

# %%
combined_data = pd.read_csv(r'C:\Users\jz421\Box\CoganLab\D_Data\GlobalLocal\combinedData.csv')

# %% [markdown]
# map blockType to congruency and switch proportions in the behavioral data

# %%
combined_data[['congruencyProportion', 'switchProportion']] = combined_data.apply(utils.map_block_type, axis=1)

# %% [markdown]
# ### load evoked and stuff for all subjects in a dictionary

# %%
# # example of how to use this with multiple conditions, even matching any value in a list. Although I only ever have two conditions of a type so not super necessary.
# # make sure to use the correct column names and values that match with what combinedData uses.

# congruency
stimulus_congruency_conditions = {
    "Stimulus_c": {
        "BIDS_events": ["Stimulus/c25.0", "Stimulus/c75.0"],
        "congruency": "c"
    },
    "Stimulus_i": {
        "BIDS_events": ["Stimulus/i25.0", "Stimulus/i75.0"],
        "congruency": "i"
    }
}

# switch type
stimulus_switch_type_conditions = {
    "Stimulus_r": {
        "BIDS_events": ["Stimulus/r25.0", "Stimulus/r75.0"],
        "switchType": "r"
    },
    "Stimulus_s": {        
        "BIDS_events": ["Stimulus/s25.0", "Stimulus/s75.0"],
        "switchType": "s"
    }
}

# #  ir vs is
# output_names = ["Stimulus_ir_fixationCrossBase_1sec_mirror", "Stimulus_is_fixationCrossBase_1sec_mirror"]
# output_names_conditions = {
#     "Stimulus_ir_fixationCrossBase_1sec_mirror": {
#         "congruency": "i",
#         "switchType": "r"
#     },
#     "Stimulus_is_fixationCrossBase_1sec_mirror": {
#         "congruency": "i",
#         "switchType": "s"
#     }
# }

# #  cr vs cs
# output_names = ["Stimulus_cr_fixationCrossBase_1sec_mirror", "Stimulus_cs_fixationCrossBase_1sec_mirror"]
# output_names_conditions = {
#     "Stimulus_cr_fixationCrossBase_1sec_mirror": {
#         "congruency": "c",
#         "switchType": "r"
#     },
#     "Stimulus_cs_fixationCrossBase_1sec_mirror": {
#         "congruency": "c",
#         "switchType": "s"
#     }
# }

# #  is vs cs
# output_names = ["Stimulus_cs_fixationCrossBase_1sec_mirror", "Stimulus_is_fixationCrossBase_1sec_mirror"]
# output_names_conditions = {
#     "Stimulus_cs_fixationCrossBase_1sec_mirror": {
#         "congruency": "c",
#         "switchType": "s"
#     },
#     "Stimulus_is_fixationCrossBase_1sec_mirror": {
#         "congruency": "i",
#         "switchType": "s"
#     }
# }

# #  ir vs cr
# output_names = ["Stimulus_cr_fixationCrossBase_1sec_mirror", "Stimulus_ir_fixationCrossBase_1sec_mirror"]
# output_names_conditions = {
#     "Stimulus_cr_fixationCrossBase_1sec_mirror": {
#         "congruency": "c",
#         "switchType": "r"
#     },
#     "Stimulus_ir_fixationCrossBase_1sec_mirror": {
#         "congruency": "i",
#         "switchType": "r"
#     }
# }

# # # main effect interaction effects (run this with the anova code. Ugh make everything more modular later.)

stimulus_main_effect_conditions = {
    "Stimulus_ir": {
        "BIDS_events": ["Stimulus/i25.0/r25.0", "Stimulus/i25.0/r75.0", "Stimulus/i75.0/r25.0", "Stimulus/i75.0/r75.0"],
        "congruency": "i",
        "switchType": "r"
    },
    "Stimulus_is": {
        "BIDS_events": ["Stimulus/i25.0/s25.0", "Stimulus/i25.0/s75.0", "Stimulus/i75.0/s25.0", "Stimulus/i75.0/s75.0"],
        "congruency": "i",
        "switchType": "s"
    },
    "Stimulus_cr": {
        "BIDS_events": ["Stimulus/c25.0/r25.0", "Stimulus/c25.0/r75.0", "Stimulus/c75.0/r25.0", "Stimulus/c75.0/r75.0"],
        "congruency": "c",
        "switchType": "r"
    },
    "Stimulus_cs": {
        "BIDS_events": ["Stimulus/c25.0/s25.0", "Stimulus/c25.0/s75.0", "Stimulus/c75.0/s25.0", "Stimulus/c75.0/s75.0"],
        "congruency": "c",
        "switchType": "s"
    }
}

# # block interaction contrasts for lwpc

stimulus_lwpc_conditions = {
    "Stimulus_c25": {
        "BIDS_events": "Stimulus/c25.0",
        "congruency": "c",
        "congruencyProportion": "75%" #this is flipped because the BIDS events are saved in terms of incongruency proportion
    },
    "Stimulus_c75": {
        "BIDS_events": "Stimulus/c75.0",
        "congruency": "c",
        "congruencyProportion": "25%" #this is flipped because the BIDS events are saved in terms of incongruency proportion
    },
    "Stimulus_i25": {
        "BIDS_events": "Stimulus/i25.0",
        "congruency": "i",
        "congruencyProportion": "75%" #this is flipped because the BIDS events are saved in terms of incongruency proportion
    },
    "Stimulus_i75": {
        "BIDS_events": "Stimulus/i75.0",
        "congruency": "i",
        "congruencyProportion": "25%" #this is flipped because the BIDS events are saved in terms of incongruency proportion
    }
}

# # block interaction contrasts for lwps

stimulus_lwps_conditions = {
    "Stimulus_s25": {
        "BIDS_events": "Stimulus/s25.0",
        "switchType": "s",
        "switchProportion": "25%"
    },
    "Stimulus_s75": {
        "BIDS_events": "Stimulus/s75.0",
        "switchType": "s",
        "switchProportion": "75%"
    },
    "Stimulus_r25": {
        "BIDS_events": "Stimulus/r25.0",
        "switchType": "r",
        "switchProportion": "25%"
    },
    "Stimulus_r75": {
        "BIDS_events": "Stimulus/r75.0",
        "switchType": "r",
        "switchProportion": "75%"
    }
}

# all 16 trial types
stimulus_experiment_conditions = {
    "Stimulus_i25s25": {
        "BIDS_events": "Stimulus/i25.0/s25.0",
        "congruency": "i",
        "congruencyProportion": "75%",
        "switchType": "s",
        "switchProportion": "25%"
    },
    "Stimulus_i25s75": {
        "BIDS_events": "Stimulus/i25.0/s75.0",
        "congruency": "i",
        "congruencyProportion": "75%",
        "switchType": "s",
        "switchProportion": "75%"
    },
    "Stimulus_i75s25": {
        "BIDS_events": "Stimulus/i75.0/s25.0",
        "congruency": "i",
        "congruencyProportion": "25%",
        "switchType": "s",
        "switchProportion": "25%"
    },
    "Stimulus_i75s75": {
        "BIDS_events": "Stimulus/i75.0/s75.0",
        "congruency": "i",
        "congruencyProportion": "25%",
        "switchType": "s",
        "switchProportion": "75%"
    },
    "Stimulus_i25r25": {
        "BIDS_events": "Stimulus/i25.0/r25.0",
        "congruency": "i",
        "congruencyProportion": "75%",
        "switchType": "r",
        "switchProportion": "25%"
    },
    "Stimulus_i25r75": {
        "BIDS_events": "Stimulus/i25.0/r75.0",
        "congruency": "i",
        "congruencyProportion": "75%",
        "switchType": "r",
        "switchProportion": "75%"
    },
    "Stimulus_i75r25": {
        "BIDS_events": "Stimulus/i75.0/r25.0",
        "congruency": "i",
        "congruencyProportion": "25%",
        "switchType": "r",
        "switchProportion": "25%"
    },
    "Stimulus_i75r75": {
        "BIDS_events": "Stimulus/i75.0/r75.0",
        "congruency": "i",
        "congruencyProportion": "25%",
        "switchType": "r",
        "switchProportion": "75%"
    },
    "Stimulus_c25s25": {
        "BIDS_events": "Stimulus/c25.0/s25.0",
        "congruency": "c",
        "congruencyProportion": "75%",
        "switchType": "s",
        "switchProportion": "25%"
    },
    "Stimulus_c25s75": {
        "BIDS_events": "Stimulus/c25.0/s75.0",
        "congruency": "c",
        "congruencyProportion": "75%",
        "switchType": "s",
        "switchProportion": "75%"
    },
    "Stimulus_c75s25": {
        "BIDS_events": "Stimulus/c75.0/s25.0",
        "congruency": "c",
        "congruencyProportion": "25%",
        "switchType": "s",
        "switchProportion": "25%"
    },
    "Stimulus_c75s75": {
        "BIDS_events": "Stimulus/c75.0/s75.0",
        "congruency": "c",
        "congruencyProportion": "25%",
        "switchType": "s",
        "switchProportion": "75%"
    },
    "Stimulus_c25r25": {
        "BIDS_events": "Stimulus/c25.0/r25.0",
        "congruency": "c",
        "congruencyProportion": "75%",
        "switchType": "r",
        "switchProportion": "25%"
    },
    "Stimulus_c25r75": {
        "BIDS_events": "Stimulus/c25.0/r75.0",
        "congruency": "c",
        "congruencyProportion": "75%",
        "switchType": "r",
        "switchProportion": "75%"
    },
    "Stimulus_c75r25": {
        "BIDS_events": "Stimulus/c75.0/r25.0",
        "congruency": "c",
        "congruencyProportion": "25%",
        "switchType": "r",
        "switchProportion": "25%"
    },
    "Stimulus_c75r75": {
        "BIDS_events": "Stimulus/c75.0/r75.0",
        "congruency": "c",
        "congruencyProportion": "25%",
        "switchType": "r",
        "switchProportion": "75%"
    }
}

# stimulus details
stimulus_conditions = {
    "Stimulus_bigSsmallHtaskG": {
        "BIDS_events": "Stimulus/BigLetters/SmallLetterh/Taskg",
        "bigLetter": "s",
        "smallLetter": "h",
        "task": "g"
    },
    "Stimulus_bigSsmallHtaskL": {
        "BIDS_events": "Stimulus/BigLetters/SmallLetterh/Taskl",
        "bigLetter": "s",
        "smallLetter": "h",
        "task": "l"
    },
    "Stimulus_bigSsmallStaskG": {
        "BIDS_events": "Stimulus/BigLetters/SmallLetters/Taskg",
        "bigLetter": "s",
        "smallLetter": "s",
        "task": "g"
    },
    "Stimulus_bigSsmallStaskL": {
        "BIDS_events": "Stimulus/BigLetters/SmallLetters/Taskl",
        "bigLetter": "s",
        "smallLetter": "s",
        "task": "l"
    },
    "Stimulus_bigHsmallHtaskG": {
        "BIDS_events": "Stimulus/BigLetterh/SmallLetterh/Taskg",
        "bigLetter": "h",
        "smallLetter": "h",
        "task": "g"
    },
    "Stimulus_bigHsmallHtaskL": {
        "BIDS_events": "Stimulus/BigLetterh/SmallLetterh/Taskl",
        "bigLetter": "h",
        "smallLetter": "h",
        "task": "l"
    },
    "Stimulus_bigHsmallStaskG": {
        "BIDS_events": "Stimulus/BigLetterh/SmallLetters/Taskg",
        "bigLetter": "h",
        "smallLetter": "s",
        "task": "g"
    },
    "Stimulus_bigHsmallStaskL": {
        "BIDS_events": "Stimulus/BigLetterh/SmallLetters/Taskl",
        "bigLetter": "h",
        "smallLetter": "s",
        "task": "l"
    }
}


# big letter details
stimulus_big_letter_conditions = {
    "Stimulus_bigS": {
        "BIDS_events": "Stimulus/BigLetters",
        "bigLetter": "s",
    },
    "Stimulus_bigH": {
        "BIDS_events": "Stimulus/BigLetterh",
        "bigLetter": "h",
    }
}

# small letter details
stimulus_small_letter_conditions = {
    "Stimulus_smallS": {
        "BIDS_events": "Stimulus/SmallLetters",
        "smallLetter": "s",
    },
    "Stimulus_smallH": {
        "BIDS_events": "Stimulus/SmallLetterh",
        "smallLetter": "h",
    }
}

# task details
stimulus_task_conditions = {
    "Stimulus_taskG": {
        "BIDS_events": "Stimulus/Taskg",
        "task": "g",
    },
    "Stimulus_taskL": {
        "BIDS_events": "Stimulus/Taskl",
        "task": "l",
    }
}

# congruency
response_congruency_conditions = {
    "Response_c": {
        "BIDS_events": ["Response/c25.0", "Response/c75.0"],
        "congruency": "c"
    },
    "Response_i": {
        "BIDS_events": ["Response/i25.0", "Response/i75.0"],
        "congruency": "i"
    }
}

# switch type
response_switch_type_conditions = {
    "Response_r": {
        "BIDS_events": ["Response/r25.0", "Response/r75.0"],
        "switchType": "r"
    },
    "Response_s": {        
        "BIDS_events": ["Response/s25.0", "Response/s75.0"],
        "switchType": "s"
    }
}

response_experiment_conditions = {
    "Response_i25s25": {
        "BIDS_events": "Response/i25.0/s25.0",
        "congruency": "i",
        "congruencyProportion": "75%",
        "switchType": "s",
        "switchProportion": "25%"
    },
    "Response_i25s75": {
        "BIDS_events": "Response/i25.0/s75.0",
        "congruency": "i",
        "congruencyProportion": "75%",
        "switchType": "s",
        "switchProportion": "75%"
    },
    "Response_i75s25": {
        "BIDS_events": "Response/i75.0/s25.0",
        "congruency": "i",
        "congruencyProportion": "25%",
        "switchType": "s",
        "switchProportion": "25%"
    },
    "Response_i75s75": {
        "BIDS_events": "Response/i75.0/s75.0",
        "congruency": "i",
        "congruencyProportion": "25%",
        "switchType": "s",
        "switchProportion": "75%"
    },
    "Response_i25r25": {
        "BIDS_events": "Response/i25.0/r25.0",
        "congruency": "i",
        "congruencyProportion": "75%",
        "switchType": "r",
        "switchProportion": "25%"
    },
    "Response_i25r75": {
        "BIDS_events": "Response/i25.0/r75.0",
        "congruency": "i",
        "congruencyProportion": "75%",
        "switchType": "r",
        "switchProportion": "75%"
    },
    "Response_i75r25": {
        "BIDS_events": "Response/i75.0/r25.0",
        "congruency": "i",
        "congruencyProportion": "25%",
        "switchType": "r",
        "switchProportion": "25%"
    },
    "Response_i75r75": {
        "BIDS_events": "Response/i75.0/r75.0",
        "congruency": "i",
        "congruencyProportion": "25%",
        "switchType": "r",
        "switchProportion": "75%"
    },
    "Response_c25s25": {
        "BIDS_events": "Response/c25.0/s25.0",
        "congruency": "c",
        "congruencyProportion": "75%",
        "switchType": "s",
        "switchProportion": "25%"
    },
    "Response_c25s75": {
        "BIDS_events": "Response/c25.0/s75.0",
        "congruency": "c",
        "congruencyProportion": "75%",
        "switchType": "s",
        "switchProportion": "75%"
    },
    "Response_c75s25": {
        "BIDS_events": "Response/c75.0/s25.0",
        "congruency": "c",
        "congruencyProportion": "25%",
        "switchType": "s",
        "switchProportion": "25%"
    },
    "Response_c75s75": {
        "BIDS_events": "Response/c75.0/s75.0",
        "congruency": "c",
        "congruencyProportion": "25%",
        "switchType": "s",
        "switchProportion": "75%"
    },
    "Response_c25r25": {
        "BIDS_events": "Response/c25.0/r25.0",
        "congruency": "c",
        "congruencyProportion": "75%",
        "switchType": "r",
        "switchProportion": "25%"
    },
    "Response_c25r75": {
        "BIDS_events": "Response/c25.0/r75.0",
        "congruency": "c",
        "congruencyProportion": "75%",
        "switchType": "r",
        "switchProportion": "75%"
    },
    "Response_c75r25": {
        "BIDS_events": "Response/c75.0/r25.0",
        "congruency": "c",
        "congruencyProportion": "25%",
        "switchType": "r",
        "switchProportion": "25%"
    },
    "Response_c75r75": {
        "BIDS_events": "Response/c75.0/r75.0",
        "congruency": "c",
        "congruencyProportion": "25%",
        "switchType": "r",
        "switchProportion": "75%"
    }
}

response_conditions = {
    "Response_bigSsmallHtaskG": {
        "BIDS_events": "Response/BigLetters/SmallLetterh/Taskg",
        "bigLetter": "s",
        "smallLetter": "h",
        "task": "g"
    },
    "Response_bigSsmallHtaskL": {
        "BIDS_events": "Response/BigLetters/SmallLetterh/Taskl",
        "bigLetter": "s",
        "smallLetter": "h",
        "task": "l"
    },
    "Response_bigSsmallStaskG": {
        "BIDS_events": "Response/BigLetters/SmallLetters/Taskg",
        "bigLetter": "s",
        "smallLetter": "s",
        "task": "g"
    },
    "Response_bigSsmallStaskL": {
        "BIDS_events": "Response/BigLetters/SmallLetters/Taskl",
        "bigLetter": "s",
        "smallLetter": "s",
        "task": "l"
    },
    "Response_bigHsmallHtaskG": {
        "BIDS_events": "Response/BigLetterh/SmallLetterh/Taskg",
        "bigLetter": "h",
        "smallLetter": "h",
        "task": "g"
    },
    "Response_bigHsmallHtaskL": {
        "BIDS_events": "Response/BigLetterh/SmallLetterh/Taskl",
        "bigLetter": "h",
        "smallLetter": "h",
        "task": "l"
    },
    "Response_bigHsmallStaskG": {
        "BIDS_events": "Response/BigLetterh/SmallLetters/Taskg",
        "bigLetter": "h",
        "smallLetter": "s",
        "task": "g"
    },
    "Response_bigHsmallStaskL": {
        "BIDS_events": "Response/BigLetterh/SmallLetters/Taskl",
        "bigLetter": "h",
        "smallLetter": "s",
        "task": "l"
    }
}


# big letter details
response_big_letter_conditions = {
    "Response_bigS": {
        "BIDS_events": "Response/BigLetters",
        "bigLetter": "s",
    },
    "Response_bigH": {
        "BIDS_events": "Response/BigLetterh",
        "bigLetter": "h",
    }
}

# small letter details
response_small_letter_conditions = {
    "Response_smallS": {
        "BIDS_events": "Response/SmallLetters",
        "smallLetter": "s",
    },
    "Response_smallH": {
        "BIDS_events": "Response/SmallLetterh",
        "smallLetter": "h",
    }
}

# task details
response_task_conditions = {
    "Response_taskG": {
        "BIDS_events": "Response/Taskg",
        "task": "g",
    },
    "Response_taskL": {
        "BIDS_events": "Response/Taskl",
        "task": "l",
    }
}

# %%
task='GlobalLocal'
conditions = stimulus_main_effect_conditions # set this to whichever conditions you're running

stimulus_locked = True  #toggle
response_locked = not stimulus_locked

if stimulus_locked:
    epochs_root_file = "Stimulus_0.5sec_within1sec_randoffset_preStimulusBase_decFactor_8_outliers_10"
elif response_locked:
    epochs_root_file = "Response_0.5sec_within1sec_randoffset_preStimulusBase_decFactor_8_outliers_10"


output_names = [condition for condition in conditions.keys()]

# %%
if conditions == stimulus_conditions:
    conditions_save_name = 'stimulus_conditions'
elif conditions == stimulus_experiment_conditions:
    conditions_save_name = 'stimulus_experiment_conditions'
elif conditions == stimulus_main_effect_conditions:
    conditions_save_name = 'stimulus_main_effect_conditions'
elif conditions == stimulus_lwpc_conditions:
    conditions_save_name = 'stimulus_lwpc_conditions'
elif conditions == stimulus_lwps_conditions:
    conditions_save_name = 'stimulus_lwps_conditions'
elif conditions == stimulus_big_letter_conditions:
    conditions_save_name = 'stimulus_big_letter_conditions'
elif conditions == stimulus_small_letter_conditions:
    conditions_save_name = 'stimulus_small_letter_conditions'
elif conditions == stimulus_task_conditions:
    conditions_save_name = 'stimulus_task_conditions'
elif conditions == stimulus_congruency_conditions:
    conditions_save_name = 'stimulus_congruency_conditions'
elif conditions == stimulus_switch_type_conditions:
    conditions_save_name = 'stimulus_switch_type_conditions'

elif conditions == response_conditions:
    conditions_save_name = 'response_conditions'
elif conditions == response_experiment_conditions:
    conditions_save_name = 'response_experiment_conditions'
elif conditions == response_big_letter_conditions:
    conditions_save_name = 'response_big_letter_conditions'
elif conditions == response_small_letter_conditions:
    conditions_save_name = 'response_small_letter_conditions'
elif conditions == response_task_conditions:
    conditions_save_name = 'response_task_conditions'
elif conditions == response_congruency_conditions:
    conditions_save_name = 'response_congruency_conditions'
elif conditions == response_switch_type_conditions:
    conditions_save_name = 'response_switch_type_conditions'
    
# Assuming 'combined_data' is your DataFrame and 'subjects' is your list of subject IDs
subjects_mne_objects = utils.create_subjects_mne_objects_dict(subjects=subjects, epochs_root_file=epochs_root_file, conditions=conditions, task="GlobalLocal", just_HG_ev1_rescaled=True, acc_trials_only=True)

# %% [markdown]
# ### load stimulus significant channels. Compare ROI electrodes in next cell to these to see if they're included.
# 
# maybe do response significant channels too/instead?

# %%
sig_chans_per_subject = utils.get_sig_chans_per_subject(subjects, epochs_root_file, task='GlobalLocal', LAB_root=None)

# Now sig_chans_per_subject dictionary is populated with significant channels for each subject

# %%
sig_chans_per_subject

# %% [markdown]
# ### get the significant electrodes across subjects for each ROI of interest

# %% [markdown]
# dlPFC based on Yamagishi et al 2016 definition is G_front_middle, G_front_sup, S_front_inf, S_front_middle, S_front_sup
# ACC based on Destrieux et al 2010 definition is G_and_S_cingul-Ant

# %%
# def filter_electrodes_by_roi(subjects_electrodes_dict, sig_chans_per_subject, roi_list):
#     """
#     Filters electrodes based on specified ROIs and returns significant electrodes for each subject.

#     Args:
#     subjects_electrodes_dict (dict): A dictionary with subjects as keys and electrode-to-ROI mappings as values.
#     sig_chans_per_subject (dict): A dictionary with subjects as keys and lists of significant channels as values.
#     roi_list (list): A list of ROIs to filter electrodes.

#     Returns:
#     dict: A dictionary with subjects as keys and lists of significant electrodes in specified ROIs as values.
#     """
#     filtered_electrodes_per_subject = {}

#     for sub, electrodes_dict in subjects_electrodes_dict.items():
#         filtered = {key: value for key, value in electrodes_dict['filtROI_dict'].items() 
#                     if any(roi in key for roi in roi_list)}

#         # Aggregate electrodes into a list for each subject
#         filtered_electrodes = []
#         for electrodes in filtered.values():
#             filtered_electrodes.extend(electrodes)

#         filtered_electrodes_per_subject[sub] = filtered_electrodes
#         print(f'For subject {sub}, {", ".join(roi_list)} electrodes are: {filtered_electrodes}')

#     # Now filter for significant electrodes
#     sig_filtered_electrodes_per_subject = {}

#     for sub, filtered_electrodes in filtered_electrodes_per_subject.items():
#         # Retrieve the list of significant channels for the subject
#         sig_chans = sig_chans_per_subject.get(sub, [])

#         # Find the intersection of filtered electrodes and significant channels for the subject
#         sig_filtered_electrodes = [elec for elec in filtered_electrodes if elec in sig_chans]

#         # Store the significant filtered electrodes for the subject
#         sig_filtered_electrodes_per_subject[sub] = sig_filtered_electrodes
#         print(f"Subject {sub} significant {', '.join(roi_list)} electrodes: {sig_filtered_electrodes}")

#     return filtered_electrodes_per_subject, sig_filtered_electrodes_per_subject

# # Example usage:
# dlpfc_rois = ["G_front_middle", "G_front_sup", "S_front_inf", "S_front_middle", "S_front_sup"] #dorsolateral prefrontal cortex
# acc_rois = ["G_and_S_cingul-Ant", "G_and_S_cingul-Mid-Ant"] #anterior cingulate cortex
# parietal_rois = ["G_parietal_sup", "S_intrapariet_and_P_trans", "G_pariet_inf-Angular", "G_pariet_inf-Supramar"] #superior parietal lobule, intraparietal sulcus, and inferior parietal lobule (split into angular gyrus and supramarginal gyrus)

# dlpfc_electrodes_per_subject, sig_dlpfc_electrodes_per_subject = utils.filter_electrodes_by_roi(subjects_electrodestoROIs_dict, sig_chans_per_subject, dlpfc_rois)
# # acc_electrodes_per_subject, sig_acc_electrodes_per_subject = utils.filter_electrodes_by_roi(subjects_electrodestoROIs_dict, sig_chans_per_subject, acc_rois)
# # parietal_electrodes_per_subject, sig_parietal_electrodes_per_subject = utils.filter_electrodes_by_roi(subjects_electrodestoROIs_dict, sig_chans_per_subject, parietal_rois)

# sig_electrodes_per_subject_roi = {}
# sig_electrodes_per_subject_roi['dlpfc'] = sig_dlpfc_electrodes_per_subject
# sig_electrodes_per_subject_roi['acc'] = sig_acc_electrodes_per_subject
# sig_electrodes_per_subject_roi['parietal'] = sig_parietal_electrodes_per_subject

# %%
# def make_sig_electrodes_per_subject_and_roi_dict(rois_dict, subjects_electrodestoROIs_dict, sig_chans_per_subject):
#     """
#     Processes electrodes by ROI and filters significant electrodes.

#     Parameters:
#     - rois_dict: A dictionary mapping each region of interest (ROI) to a list of brain regions.
#     - subjects_electrodestoROIs_dict: A dictionary mapping subjects to their electrode-to-ROI assignments.
#     - sig_chans_per_subject: A dictionary indicating significant channels per subject.

#     Returns:
#     - A tuple of two dictionaries:
#       1. electrodes_per_subject_roi: Electrodes per subject for each ROI.
#       2. sig_electrodes_per_subject_roi: Significant electrodes per subject for each ROI.
#     """
#     electrodes_per_subject_roi = {}
#     sig_electrodes_per_subject_roi = {}

#     for roi_name, roi_regions in rois_dict.items():
#         # Apply the filter_electrodes_by_roi function for each set of ROI regions
#         electrodes_per_subject, sig_electrodes_per_subject = filter_electrodes_by_roi(
#             subjects_electrodestoROIs_dict, sig_chans_per_subject, roi_regions)
        
#         # Store the results in the respective dictionaries
#         electrodes_per_subject_roi[roi_name] = electrodes_per_subject
#         sig_electrodes_per_subject_roi[roi_name] = sig_electrodes_per_subject

#     return electrodes_per_subject_roi, sig_electrodes_per_subject_roi

# %%
# rois_dict = {
#     # 'dlpfc': ["G_front_middle", "G_front_sup", "S_front_inf", "S_front_middle", "S_front_sup"],
#     # 'acc': ["G_and_S_cingul-Ant", "G_and_S_cingul-Mid-Ant"],
#     # 'parietal': ["G_parietal_sup", "S_intrapariet_and_P_trans", "G_pariet_inf-Angular", "G_pariet_inf-Supramar"],
#     'lpfc': ["G_front_inf-Opercular", "G_front_inf-Orbital", "G_front_inf-Triangul", "G_front_middle", "G_front_sup", "Lat_Fis-ant-Horizont", "Lat_Fis-ant-Vertical", "S_circular_insula_ant", "S_circular_insula_sup", "S_front_inf", "S_front_middle", "S_front_sup"],
#     'v1': ["G_oc-temp_med-Lingual", "S_calcarine", "G_cuneus"],
#     'occ': ["G_cuneus", "G_and_S_occipital_inf", "G_occipital_middle", "G_occipital_sup", "G_oc-temp_lat-fusifor", "G_oc-temp_med-Lingual", "Pole_occipital", "S_calcarine", "S_oc_middle_and_Lunatus", "S_oc_sup_and_transversal", "S_occipital_ant"]
# }

# the cns 24/sfn 24 poster plots need just one roi. Fix all this code later. 10/1.
rois_dict = {
    'lpfc': ["G_front_inf-Opercular", "G_front_inf-Orbital", "G_front_inf-Triangul", "G_front_middle", "G_front_sup", "Lat_Fis-ant-Horizont", "Lat_Fis-ant-Vertical", "S_circular_insula_ant", "S_circular_insula_sup", "S_front_inf", "S_front_middle", "S_front_sup"]
}

rois = list(rois_dict.keys())
electrodes_per_subject_roi, sig_electrodes_per_subject_roi = utils.make_sig_electrodes_per_subject_and_roi_dict(rois_dict, subjects_electrodestoROIs_dict, sig_chans_per_subject)

# %% [markdown]
# get total number of electrodes (make this modular with roi later once everything works)

# %%
# def calculate_total_electrodes(sig_electrodes_per_subject_roi, electrodes_per_subject_roi):
#     """
#     Calculates the total number of significant and total electrodes for each ROI across all subjects.

#     Parameters:
#     - sig_electrodes_per_subject_roi: A dictionary containing significant electrodes per subject for each ROI.
#     - electrodes_per_subject_roi: A dictionary containing all electrodes per subject for each ROI.

#     Returns:
#     - A dictionary containing the counts of significant and total electrodes for each ROI.
#     """
#     total_electrodes_info = {}

#     for roi in sig_electrodes_per_subject_roi:
#         # Calculate total significant electrodes for the current ROI
#         total_sig_entries = sum(len(sig_electrodes_per_subject_roi[roi][sub]) for sub in sig_electrodes_per_subject_roi[roi])
#         # Calculate total electrodes for the current ROI
#         total_entries = sum(len(electrodes_per_subject_roi[roi][sub]) for sub in electrodes_per_subject_roi[roi])

#         # Store the results in the dictionary
#         total_electrodes_info[roi] = {
#             'total_significant_electrodes': total_sig_entries,
#             'total_electrodes': total_entries
#         }

#     return total_electrodes_info

# %%
# Example usage:
total_electrodes_info = utils.calculate_total_electrodes(sig_electrodes_per_subject_roi, electrodes_per_subject_roi)
for roi, counts in total_electrodes_info.items():
    print(f"Total number of significant {roi} electrodes across all subjects:", counts['total_significant_electrodes'])
    print(f"Total number of {roi} electrodes across all subjects:", counts['total_electrodes'])

# %% [markdown]
# check if any subjects have a different sampling rate

# %%
def check_sampling_rates(subjects_mne_objects, expected_sampling_rate=256):
    # This dictionary will store subjects with their sampling rates
    subject_sampling_rates = {}

    # Iterate through each subject and their corresponding data
    for subject, data in subjects_mne_objects.items():
        # Get the first epochs object from the dictionary
        if data:
            first_condition = list(data.keys())[0]
            mne_objects = data[first_condition]
            first_object = list(mne_objects.keys())[0]
            first_epochs = data[first_condition][first_object]
            actual_sampling_rate = first_epochs.info['sfreq']
            
            # Store the sampling rate in the dictionary
            subject_sampling_rates[subject] = actual_sampling_rate
    
    # Print the results
    for subject, rate in subject_sampling_rates.items():
        if rate != expected_sampling_rate:
            print(f"Subject {subject} has a different sampling rate: {rate} Hz.")
        else:
            print(f"Subject {subject} has the expected sampling rate: {rate} Hz.")
    
    return subject_sampling_rates

# Assuming 'subjects_mne_objects' is your dictionary containing MNE objects for each subject
sampling_rate = 256
subject_rates = check_sampling_rates(subjects_mne_objects, expected_sampling_rate=sampling_rate)


# %% [markdown]
# ### do stats
# 
# current approach is to run time_perm_cluster on significant dlpfc electrodes for each subject, comparing congruent and incongruent conditions. Then, average p-values across all subjects. Discuss this with Greg, probably wrong approach.
# 
# **1/23 new approach is to average across all trials for sig dlpfc electrodes, comparing incongruent and congruent conditions. Then, run stats on this new avg electrode value x time array.
# 
# Also, I'm using HG_ev1_rescaled instead of HG_ev1 to compare congruent and incongruent, so that they're normalized with a common baseline. I think this is better than comparing the raw HG traces directly.

# %% [markdown]
# ### this is 1/23 old approach of avg across trials first. Time perm cluster stats.
# 
# do stats and plotting together. Stats needs trial avg data, plotting just needs congruent_data without trial averaging (initially at least)  
# this code is so bad right now, turn into a function later  
# 
# trialAvg is for the time perm cluster stats  
# timeAvg_firstHalfSecond_firstHalfSecond_firstHalfSecond_firstHalfSecond_firstHalfSecond is for the window stats (not sure if this is even right)  
# 
# 

# %% [markdown]
# 4/30 try to make time perm stats more modular, and reusable  
# also remember that time perm cluster stats only compares two output names.  
# 
# these functions are now in utils.py. 5/6.

# %% [markdown]
# ### do 2x2 anova for interaction effects 
# this requires reloading in all four conditions (four this time cuz interaction contrasts).  
# ONLY RUN THIS WHEN LOADING IN THE FOUR INTERACTION CONTRASTS RIGHT NOW.  
# Integrate with other stats and plotting and stuff later.

# %%
# i should turn this all into a function too. 5/20.
# define time windows in terms of samples (this is cuz epochs are from -1 to 1.5 sec after stimulus onset)
# time_indices = {
#     'firstHalfSecond': (2048, 3072),
#     'secondHalfSecond': (3072, 4096),
#     'fullSecond': (2048, 4096)
# }

original_time_indices = {
    'firstHalfSecond': (2048, 3072),
    'secondHalfSecond': (3072, 4096)
}

sampling_rate_ratio = sampling_rate / 2048.0 # account for decimation factor

# Convert the time indices to the new sampling rate
time_indices = {
    key: (int(start * sampling_rate_ratio), int(end * sampling_rate_ratio))
    for key, (start, end) in original_time_indices.items()
}

condition_names = list(conditions.keys()) # get the condition names as a list

# # Select output names based on whether the processing is for a permutation test (first two outputs) or ANOVA (all outputs).
# for_perm_test = False
# relevant_output_names = output_names[:2] if for_perm_test else output_names

# Process the data
data_trialAvg_lists, data_trialStd_lists, data_timeAvg_lists, overall_electrode_mapping, electrode_mapping_per_roi = utils.process_data_for_roi(
    subjects_mne_objects, condition_names, rois, subjects, sig_electrodes_per_subject_roi, time_indices)

print("Data processing complete. Now let's concatenate the results and see what we've got!")

# Concatenate the data
concatenated_trialAvg_data = utils.concatenate_data(data_trialAvg_lists, rois, condition_names)

# Calculate means and sems
mean_and_sem = utils.calculate_mean_and_sem(concatenated_trialAvg_data, rois, condition_names)

print("Mean and SEM calculation done")


# %% [markdown]
# make dataframe for anova

# %%
LAB_root = None
# Determine LAB_root based on the operating system
if LAB_root is None:
    HOME = os.path.expanduser("~")
    LAB_root = os.path.join(HOME, "Box", "CoganLab") if os.name == 'nt' else os.path.join(HOME, "Library", "CloudStorage", "Box-Box", "CoganLab")

# Get data layout
layout = get_data(task, root=LAB_root)
save_dir = os.path.join(layout.root, 'derivatives', 'freqFilt', 'figs')

# Example structure for organizing data for ANOVA with four conditions

# Function to process and append data for ANOVA from time-averaged lists
# Adapted function to include Congruency and SwitchType
# modifying this to include time windows as another factor 4/4! Use code before 4/4 if don't want to include time windows.
def process_and_append_data_for_anova(time_averaged_lists_dict, conditions):
    data_for_anova = []
    for time_window, lists in time_averaged_lists_dict.items():
        for condition, condition_parameters in conditions.items():
            print('condition:', condition)
            # # Dynamically get condition parameters and their values for the current output_name
            # condition_parameters = conditions[condition]
            
            for roi in rois: #this is good cuz it loops through rois 3/6, the trial level one should copy this logic
                sig_electrodes_per_subject = sig_electrodes_per_subject_roi[roi]
                subjects_with_data = [subject for subject, electrodes in sig_electrodes_per_subject.items() if electrodes] # add this line to skip over subjects without data 4/1
                for subject_index, subject_data in enumerate(lists[condition][roi]):
                    subject_id = subjects_with_data[subject_index]

                    # Skip this subject if there are no significant electrodes for them in this ROI
                    if subject_id not in sig_electrodes_per_subject or not sig_electrodes_per_subject[subject_id]:
                        continue

                    # Calculate the mean across trials for each electrode
                    mean_activity_per_electrode = np.nanmean(subject_data, axis=0)
                    # untested making this more modular 2/27
                    for electrode_index, mean_activity in enumerate(mean_activity_per_electrode):
                        print('electrode index:', electrode_index)
                        electrode_name = sig_electrodes_per_subject[subject_id][electrode_index]
                        print(electrode_name)
                        # Prepare data dictionary, starting with fixed attributes
                        data_dict = {
                            'SubjectID': subject_id,
                            'Electrode': electrode_name,
                            'ROI': roi,
                            'TimeWindow': time_window,
                            'MeanActivity': mean_activity
                        }

                        # Dynamically add condition types and their values
                        data_dict.update(condition_parameters)

                        # Append the organized data to the list
                        data_for_anova.append(data_dict)
    return data_for_anova
# Create a time averaged lists dictionary to pass in to the process and append data for anova function
                    
# use this one to compare early vs late vs all time
# time_averaged_lists = {
#         "FirstHalfSecond": output_data_timeAvg_firstHalfSecond_lists,
#         "SecondHalfSecond": output_data_timeAvg_secondHalfSecond_lists,
#         "FullSecond": output_data_timeAvg_fullSecond_lists
# }
                    
# # use this one to just compare early and late time
# time_averaged_lists_dict = {
#         "FirstHalfSecond": output_data_timeAvg_firstHalfSecond_lists,
#         "SecondHalfSecond": output_data_timeAvg_secondHalfSecond_lists
# }
#

data_for_anova = process_and_append_data_for_anova(data_timeAvg_lists, conditions)
# Convert to DataFrame
df_for_anova = pd.DataFrame(data_for_anova)

# %%
df_for_anova

# %%
sig_electrodes_per_subject_roi

# %% [markdown]
# now actually run anova

# %%
df_for_anova

# %% [markdown]
# bruh THIS only works with multiple ROIs. Ugh. So the plotting doesn't work when this works...wow this code sucks. 10/1

# %%
def perform_modular_anova_all_time_windows(df, conditions, save_dir, save_name_prefix):
    # Dynamically construct the model formula based on condition keys and include TimeWindow and roi 5/20. Exclude BIDS_events.
    condition_keys = [key for key in conditions[next(iter(conditions))].keys() if key!= 'BIDS_events']
    formula_terms = ' + '.join([f'C({key})' for key in condition_keys] + ['C(TimeWindow)'] + ['C(ROI)'])
    interaction_terms = ' * '.join([f'C({key})' for key in condition_keys] + ['C(TimeWindow)'] + ['C(ROI)'])
    formula = f'MeanActivity ~ {formula_terms} + {interaction_terms}'

    # Define the model
    model = ols(formula, data=df).fit()

    # Perform the ANOVA
    anova_results = anova_lm(model, typ=2)

    # Define the base part of the results file name
    results_file_path = os.path.join(save_dir, f"{save_name_prefix}_ANOVAacrossElectrodes_allTimeWindows.txt")

    # Save the ANOVA results to a text file
    with open(results_file_path, 'w') as file:
        file.write(anova_results.__str__())

    # Optionally, print the path to the saved file and/or return it
    print(f"ANOVA results for all time windows saved to: {results_file_path}")

    # Print the results
    print(anova_results)

    return anova_results

# %%
# Join all the ROIs with '_' as the separator
rois_suffix = '_'.join(rois)

# %% [markdown]
# run anova

# %%
perform_modular_anova_all_time_windows(df_for_anova, conditions, save_dir, f'{conditions_save_name}_{rois_suffix}')

# %% [markdown]
# okay now do within-electrode anova too

# %%
def process_and_append_trial_data_for_anova(time_averaged_lists, conditions):
    data_for_anova = []
    for time_window, lists in time_averaged_lists.items():
        for condition, condition_parameters in conditions.items():
            for roi in rois:
                sig_electrodes_per_subject = sig_electrodes_per_subject_roi[roi]
                subjects_with_data = [subject for subject, electrodes in sig_electrodes_per_subject.items() if electrodes] # Skip over subjects without data
                for subject_index, subject_data in enumerate(lists[condition][roi]):
                    subject_id = subjects_with_data[subject_index]

                    if subject_id not in sig_electrodes_per_subject or not sig_electrodes_per_subject[subject_id]:
                        continue

                    for trial_index, trial_data in enumerate(subject_data):
                        # Skip trials with any missing data or incorrect length
                        if np.any(np.isnan(trial_data)) or len(trial_data) != len(sig_electrodes_per_subject[subject_id]):
                            continue

                        for electrode_index, electrode_name in enumerate(sig_electrodes_per_subject[subject_id]):
                            activity = trial_data[electrode_index] if electrode_index < len(trial_data) else np.nan

                            # Prepare the data dictionary
                            data_dict = {
                                'SubjectID': subject_id,
                                'Electrode': electrode_name,
                                'ROI': roi,
                                'TimeWindow': time_window,
                                'Trial': trial_index + 1,
                                'Activity': activity
                            }

                            # Dynamically add condition types and their values
                            data_dict.update(condition_parameters)

                            data_for_anova.append(data_dict)
    return data_for_anova
# # Example usage with the `time_averaged_lists` dictionary
# time_averaged_lists = {
#     "FirstHalfSecond": output_data_timeAvg_firstHalfSecond_lists,
#     "SecondHalfSecond": output_data_timeAvg_secondHalfSecond_lists,
#     # "FullSecond": output_data_timeAvg_fullSecond_lists  # Uncomment or comment based on your needs
# }

data_for_anova = process_and_append_trial_data_for_anova(data_timeAvg_lists, conditions)

# Convert to DataFrame
df_for_trial_level_anova = pd.DataFrame(data_for_anova)


# %%
df_for_trial_level_anova

# %%
# Assuming df_for_trial_level_anova is your DataFrame and it includes a 'SubjectID' column
def perform_modular_within_electrode_anova_roi(df, conditions, save_dir, save_name):
    '''
    This gets if an electrode is significant for specific time windows. It does not get their interaction.
    '''
    results = []
    significant_effects_structure = {}

    for subject_id in df['SubjectID'].unique():
        for electrode in df['Electrode'].unique():
            for time_window in df['TimeWindow'].unique():
                for roi in df['ROI'].unique():
                    df_filtered = df[(df['SubjectID'] == subject_id) & 
                                    (df['Electrode'] == electrode) & 
                                    (df['TimeWindow'] == time_window) &
                                    (df['ROI'] == roi)]
                    
                    if df_filtered.empty: #if this combination of subject, electrode, and time window doesn't exist, then move on.
                        continue

                    # Dynamically construct the formula based on condition keys present in the DataFrame
                    condition_keys = [key for key in conditions[next(iter(conditions))].keys() if key != 'BIDS_events']
                    formula_terms = ' + '.join([f'C({name})' for name in condition_keys])
                    interaction_terms = ' * '.join([f'C({name})' for name in condition_keys])
                    formula = f'Activity ~ {formula_terms} + {interaction_terms}'

                    # Perform the ANOVA
                    model = ols(formula, data=df_filtered).fit()
                    anova_results = anova_lm(model, typ=2)
                    
                    # Append the results
                    results.append({
                        'SubjectID': subject_id,
                        'Electrode': electrode,
                        'TimeWindow': time_window,
                        'ROI': roi,
                        'ANOVA_Results': anova_results
                    })

    # Join all the ROIs with '_' as the separator
    rois_suffix = '_'.join(rois)

    # Add the suffix '_onlySigElectrodes' to the base filename
    allElectrodesFilename = f"{save_name}_allElectrodes_{rois_suffix}.txt"
    onlySigElectrodesFilename = f"{save_name}_onlySigElectrodes_{rois_suffix}.txt"
    significantEffectsStructureFilename = f"{save_name}_significantEffectsStructure_{rois_suffix}.txt"

    # Define the full path for the results file
    results_file_path = os.path.join(save_dir, allElectrodesFilename)

    # Save the ANOVA results to a text file
    with open(results_file_path, 'w') as file:
        file.write(results.__str__())

    # Optionally, print the path to the saved file and/or return it
    print(f"results saved to: {results_file_path}")

    # Now process the significant results, including the subject ID in the output
    significant_results = []

    for result in results:
        anova_table = result['ANOVA_Results']
        subject_id = result['SubjectID']
        electrode = result['Electrode']
        time_window = result['TimeWindow']
        roi = result['ROI']
        significant_effects = anova_table[anova_table['PR(>F)'] < 0.05]
        
        if not significant_effects.empty:
            print(f"Significant effects found for Subject: {subject_id}, Electrode: {electrode}, Time Window: {time_window}, ROI: {roi}")
            print(significant_effects)
            print("\n")
            
            significant_results.append({
                'SubjectID': subject_id,
                'Electrode': electrode,
                'TimeWindow': time_window,
                'ROI': roi,
                'SignificantEffects': significant_effects
            })

        # Extract significant effects for the current result. Basically just get the p-value. 3/19.
        sig_effects_just_p_values = utils.extract_significant_effects(anova_table)
        
        if sig_effects_just_p_values:
            # Ensure subject_id and electrode keys exist
            if subject_id not in significant_effects_structure:
                significant_effects_structure[subject_id] = {}
            if electrode not in significant_effects_structure[subject_id]:
                significant_effects_structure[subject_id][electrode] = {}
            if electrode not in significant_effects_structure[subject_id][electrode]:
                significant_effects_structure[subject_id][electrode][roi] = {}

            # Assign the significant effects and their p-values to the correct structure
            significant_effects_structure[subject_id][electrode][roi][time_window] = sig_effects_just_p_values    

    # Define the full path for the results file
    significant_results_file_path = os.path.join(save_dir, onlySigElectrodesFilename)

    # Save the ANOVA results to a text file
    with open(significant_results_file_path, 'w') as file:
        file.write(significant_results.__str__())

    # Optionally, print the path to the saved file and/or return it
    print(f"significant_results saved to: {significant_results_file_path}")

    significant_effects_structure_file_path = os.path.join(save_dir, significantEffectsStructureFilename)
    # Save the ANOVA results to a json file (if this works, change the others to json files too)
    with open(significant_effects_structure_file_path, 'w') as file:
        json.dump(significant_effects_structure, file, indent=4)

    # Optionally, print the path to the saved file and/or return it
    print(f"significant_effects_structure saved to: {significant_effects_structure_file_path}")

    return results, significant_results, significant_effects_structure

# %%
# Assuming df_for_trial_level_anova is your DataFrame and it includes a 'SubjectID' column
def perform_modular_within_electrode_anova_roi_timeWindowInteractions(df, conditions, save_dir, save_name):
    '''
    This gets if the main and interaction effect of time window is significant for an electrode. AKA is overall or condition-specific activity different across differnet time windows?
    It does not tell you which time windows are significant. 
    '''
    import json
    results = []
    significant_effects_structure = {}

    for subject_id in df['SubjectID'].unique():
        for electrode in df['Electrode'].unique():
            for roi in df['ROI'].unique():
                df_filtered = df[(df['SubjectID'] == subject_id) & 
                                    (df['Electrode'] == electrode) & (df['ROI'] == roi)]
                
                if df_filtered.empty:
                    continue
                
                # Dynamically construct the formula based on condition keys present in the DataFrame, skipping BIDS_events
                condition_keys = [key for key in conditions[next(iter(conditions))].keys() if key != 'BIDS_events']
                formula_terms = ' + '.join([f'C({name})' for name in condition_keys] + ['C(TimeWindow)']) # maybe add roi too? 5/20.
                interaction_terms = ' * '.join([f'C({name})' for name in condition_keys] + ['C(TimeWindow)'])
                formula = f'Activity ~ {formula_terms} + {interaction_terms}'

                # Perform the ANOVA
                model = ols(formula, data=df_filtered).fit()
                anova_results = anova_lm(model, typ=2)
                
                # Append the results
                results.append({
                    'SubjectID': subject_id,
                    'Electrode': electrode,
                    'ROI': roi,
                    'ANOVA_Results': anova_results
                })
    
    # Join all the ROIs with '_' as the separator
    rois_suffix = '_'.join(rois)

    # Add the suffix '_onlySigElectrodes' to the base filename
    allElectrodesFilename = f"{save_name}_allElectrodes_{rois_suffix}.txt"
    onlySigElectrodesFilename = f"{save_name}_onlySigElectrodes_{rois_suffix}.txt"
    significantEffectsStructureFilename = f"{save_name}_significantEffectsStructure_timeWindowInteractions_{rois_suffix}.txt"

    # Define the full path for the results file
    results_file_path = os.path.join(save_dir, allElectrodesFilename)

    # Save the ANOVA results to a text file
    with open(results_file_path, 'w') as file:
        file.write(results.__str__())

    # Optionally, print the path to the saved file and/or return it
    print(f"results saved to: {results_file_path}")

    # Now process the significant results, including the subject ID in the output
    significant_results = []

    for result in results:
        anova_table = result['ANOVA_Results']
        subject_id = result['SubjectID']
        electrode = result['Electrode']
        roi = result['ROI']
        
        significant_effects = anova_table[anova_table['PR(>F)'] < 0.05]
        
        if not significant_effects.empty:
            print(f"Significant effects found for Subject: {subject_id}, Electrode: {electrode}, ROI: {roi}")
            print(significant_effects)
            print("\n")
            
            significant_results.append({
                'SubjectID': subject_id,
                'Electrode': electrode,
                'ROI': roi,
                'SignificantEffects': significant_effects
            })

        # Extract significant effects for the current result. Basically just get the p-value. 3/19.
        sig_effects_just_p_values = utils.extract_significant_effects(anova_table)
        
        if sig_effects_just_p_values:
            # Ensure subject_id and electrode keys exist
            if subject_id not in significant_effects_structure:
                significant_effects_structure[subject_id] = {}
            if electrode not in significant_effects_structure[subject_id]:
                significant_effects_structure[subject_id][electrode] = {}
            if electrode not in significant_effects_structure[subject_id][electrode]:
                significant_effects_structure[subject_id][electrode][roi] = {}

            # Assign the significant effects and their p-values to the correct structure
            significant_effects_structure[subject_id][electrode][roi] = sig_effects_just_p_values    

    # Define the full path for the results file
    significant_results_file_path = os.path.join(save_dir, onlySigElectrodesFilename)

    # Save the ANOVA results to a text file
    with open(significant_results_file_path, 'w') as file:
        file.write(significant_results.__str__())

    # Optionally, print the path to the saved file and/or return it
    print(f"significant_results saved to: {significant_results_file_path}")

    significant_effects_structure_file_path = os.path.join(save_dir, significantEffectsStructureFilename)
    # Save the ANOVA results to a json file (if this works, change the others to json files too)
    with open(significant_effects_structure_file_path, 'w') as file:
        json.dump(significant_effects_structure, file, indent=4)

    # Optionally, print the path to the saved file and/or return it
    print(f"significant_effects_structure saved to: {significant_effects_structure_file_path}")

    return results, significant_results, significant_effects_structure


# %% [markdown]
# run within electrode anova

# %%
results, significant_results, significant_effects_structure = perform_modular_within_electrode_anova_roi(df_for_trial_level_anova, conditions, save_dir, conditions_save_name)

# %% [markdown]
# 4/4 do both perform_modular_within_electrode_anova_roi and perform_modular_within_electrode_anova_roi_timeWindowInteractions. This will tell us which time windows have significant activity in an electrode AND if there is significant differences in activity across time windows in an electrode.

# %%
results_timeWindowInteraction, significant_results_timeWindowInteraction, significant_effects_structure_timeWindowInteraction = perform_modular_within_electrode_anova_roi_timeWindowInteractions(df_for_trial_level_anova, conditions, save_dir, conditions_save_name)

# %%
results

# %%
significant_results 

# %% [markdown]
# ### plot and QC stats

# %% [markdown]
# plot time perm cluster stats (don't run this immediately below cell if didn't do time perm cluster)

# %%
# # Plotting
# plt.figure(figsize=(10, 6))
# plt.plot(time_perm_cluster_results['dlpfc'])
# plt.xlabel('Timepoints')
# plt.ylabel('Significance (0 or 1)')
# plt.title('Permutation Test Significance Over Time')
# plt.show()

# %% [markdown]
# ### plot interaction effects (only do this when load in all four of them)

# %% [markdown]
# https://matplotlib.org/stable/gallery/color/named_colors.html

# %%
# # add the other conditions and give them condition names and colors too
plotting_parameters = {
    'Stimulus_r': {
        'condition_parameter': 'repeat',
        'color': 'blue',
        "line_style": "-"
    },
    'Stimulus_s': {
        'condition_parameter': 'switch',
        'color': 'blue',
        "line_style": "--"
    },
    'Stimulus_c': {
        'condition_parameter': 'congruent',
        'color': 'red',
        "line_style": "-"
    },
    'Stimulus_i': {
        'condition_parameter': 'incongruent',
        'color': 'red',
        "line_style": "--"
    },
    "Stimulus_ir": {
        "condition_parameter": "IR",
        "color": "blue",
        "line_style": "-"
    },
    "Stimulus_is": {
        "condition_parameter": "IS",
        "color": "blue",
        "line_style": "--"
    },
    "Stimulus_cr": {
        "condition_parameter": "CR",
        "color": "red",
        "line_style": "-"
    },
    "Stimulus_cs": {
        "condition_parameter": "CS",
        "color": "red",
        "line_style": "--"
    },
    "Stimulus_c25": {
        "condition_parameter": "c75",
        "color": "pink",
        "line_style": "-"
    },
    "Stimulus_c75": {
        "condition_parameter": "c25",
        "color": "orange",
        "line_style": "-"
    },
    "Stimulus_i25": {
        "condition_parameter": "i75",
        "color": "pink",
        "line_style": "--"
    },
    "Stimulus_i75": {
        "condition_parameter": "i25",
        "color": "orange",
        "line_style": "--"
    },
    "Stimulus_s25": {
        "condition_parameter": "s25",
        "color": "skyblue",
        "line_style": "--"
    },
    "Stimulus_s75": {
        "condition_parameter": "s75",
        "color": "purple",
        "line_style": "--"
    },
    "Stimulus_r25": {
        "condition_parameter": "r25",
        "color": "skyblue",
        "line_style": "-"
    },
    "Stimulus_r75": {
        "condition_parameter": "r75",
        "color": "purple",
        "line_style": "-"
    },
    "Stimulus_bigH": {
        "condition_parameter": "bigH",
        "color": "green",
        "line_style": "-"
    },
    "Stimulus_bigS": {
        "condition_parameter": "bigS",
        "color": "green",
        "line_style": "--"
    },
    "Stimulus_smallH": {
        "condition_parameter": "smallH",
        "color": "orange",
        "line_style": "-"
    },
    "Stimulus_smallS": {
        "condition_parameter": "smallS",
        "color": "orange",
        "line_style": "--"
    },
    "Stimulus_taskG": {
        "condition_parameter": "taskG",
        "color": "gray",
        "line_style": "-"
    },
    "Stimulus_taskL": {
        "condition_parameter": "taskL",
        "color": "gray",
        "line_style": "--"
    },
}

# # Save the dictionary to a file
# with open('plotting_parameters.json', 'w') as file:
#     json.dump(plotting_parameters, file, indent=4)

# %% [markdown]
# copy the same plotting parameters for response as stimulus

# %%
# Loop to add 'Response' keys
for key, value in list(plotting_parameters.items()):
    if key.startswith('Stimulus'):
        new_key = key.replace('Stimulus', 'Response')
        plotting_parameters[new_key] = value

# Optional: printing to verify
for key, value in plotting_parameters.items():
    print(f"{key}: {value}")


# %%
# utils.make_plotting_parameters() #make plotting parameters. Modify colors and line types in utils.

# # Load the dictionary from the file
# with open('plotting_parameters.json', 'r') as file:
#     plotting_parameters = json.load(file)

# print(plotting_parameters)

# %%
LAB_root = None
# Determine LAB_root based on the operating system
if LAB_root is None:
    HOME = os.path.expanduser("~")
    LAB_root = os.path.join(HOME, "Box", "CoganLab") if os.name == 'nt' else os.path.join(HOME, "Library", "CloudStorage", "Box-Box", "CoganLab")

# Get data layout
layout = get_data(task, root=LAB_root)
save_dir = os.path.join(layout.root, 'derivatives', 'freqFilt', 'figs')
condition_names = list(conditions.keys()) # get the condition names as a list

def plot_interact_effects_modular_roi(roi, save_dir, save_name, mean_and_sem, condition_names, plotting_parameters, font_size=14):
    # Set global font size
    plt.rcParams.update({'font.size': font_size})

    # Base setup for directories and file paths
    save_path = os.path.join(save_dir, f'avg_{roi}_{save_name}_interactEffects_power_zscore_roi.png')

    # Initialize plot
    plt.figure(figsize=(10, 6))

    # Dynamically select the first subject and use it to extract times
    first_subject_id = next(iter(subjects_mne_objects))
    example_condition_name = next(iter(subjects_mne_objects[first_subject_id]))
    times = subjects_mne_objects[first_subject_id][example_condition_name]['HG_ev1_power_rescaled'].times

    overall_averages_for_plotting = {}
    overall_sem_for_plotting = {}
    # Initialize variables to store the global min and max values
    global_min_val = float('inf')  # Set to infinity initially
    global_max_val = float('-inf')  # Set to negative infinity initially
    
    # Generate labels and plot each condition
    for index, condition_name in enumerate(condition_names):
        overall_averages_for_plotting[condition_name] = mean_and_sem[roi][condition_name]['mean']
        overall_sem_for_plotting[condition_name] = mean_and_sem[roi][condition_name]['sem']

        # Calculate the minimum value for this condition, including SEM
        current_min_val = min(overall_averages_for_plotting[condition_name] - overall_sem_for_plotting[condition_name])
        # Calculate the maximum value for this condition, including SEM
        current_max_val = max(overall_averages_for_plotting[condition_name] + overall_sem_for_plotting[condition_name])

        # Update the global min and max values if necessary
        global_min_val = min(global_min_val, current_min_val)
        global_max_val = max(global_max_val, current_max_val)

        # Optionally, add a small margin to the range
        margin = (global_max_val - global_min_val) * 0.05  # 5% of the range as margin
        global_min_val -= margin
        global_max_val += margin

        label = plotting_parameters[condition_name]['condition_parameter']  # extract label from plotting parameters dict
        color = plotting_parameters[condition_name]['color']
        line_style = plotting_parameters[condition_name]['line_style']

        plt.plot(times, overall_averages_for_plotting[condition_name], label=f'Average {roi} {label}', linestyle=line_style, color=color)
        plt.fill_between(times, overall_averages_for_plotting[condition_name] - overall_sem_for_plotting[condition_name], overall_averages_for_plotting[condition_name] + overall_sem_for_plotting[condition_name], alpha=0.3, color=color)

    plt.xlabel('Time (s)')
    plt.ylabel('Z-score Power')
    plt.title(f'Average {roi} signal for {save_name}')
    plt.legend()

    # Adjust the y-axis limits
    plt.ylim([global_min_val, global_max_val])

    # Remove the top and right borders (spines)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.savefig(save_path)
    plt.close()

# %% [markdown]
# plot average traces

# %%
for roi in rois:
    plot_interact_effects_modular_roi(roi, save_dir, conditions_save_name, mean_and_sem, condition_names, plotting_parameters)

# %% [markdown]
# try to plot different groups of electrodes based on significance for effects 4/4  

# %%
def get_significant_electrodes_by_effect(effect_structure, effect_type):
    '''
    Extracts electrodes with significant specified effects.

    :param effect_structure: Dictionary containing significant effects for each electrode, subject, and roi.
    :param effect_type: Single string or list of strings specifying the effect(s) of interest ("congruency", "switchType", "congruency:switchType", etc).
    :return: A dictionary mapping subject IDs to lists of electrodes with the specified significant effect(s).
    '''
    significant_electrodes = {}

    # Ensure effect_type is a list to simplify the logic
    if isinstance(effect_type, str):
        effect_type = [effect_type]

    # Loop over each subject in the effect structure
    for subject_id, electrodes in effect_structure.items():
        significant_electrodes_for_subject = []  # Temporary list to hold significant electrodes for the current subject

        # Loop over each electrode and its associated ROIs
        for electrode, rois in electrodes.items():
            for roi, effects in rois.items():
                # Loop over each effect and p-value tuple
                for effect, p_value in effects:
                    if effect in effect_type and p_value < 0.05:  # Check if the effect is of interest and is significant
                        significant_electrodes_for_subject.append(electrode)
                        break  # Stop checking further effects for this electrode once a significant effect is found

        # Add the subject's significant electrodes to the result if any were found
        if significant_electrodes_for_subject:
            significant_electrodes[subject_id] = significant_electrodes_for_subject

    return significant_electrodes


# %% [markdown]
# this is for congruency x switch type interaction

# %%
output_names

# %%
significant_effects_structure_timeWindowInteraction

# %%
congruencySigElectrodes

# %%
if 'Stimulus_cr' in output_names:
    congruencySigElectrodes = get_significant_electrodes_by_effect(significant_effects_structure_timeWindowInteraction, 'congruency')
    switchTypeSigElectrodes = get_significant_electrodes_by_effect(significant_effects_structure_timeWindowInteraction, 'switchType')
    congruencySwitchTypeInteractionSigElectrodes = get_significant_electrodes_by_effect(significant_effects_structure_timeWindowInteraction, 'congruency:switchType')
    allEffectSensitiveElectrodes = get_significant_electrodes_by_effect(significant_effects_structure_timeWindowInteraction, ['congruency', 'switchType', 'congruency:switchType'])

# %% [markdown]
# this one is for congruency x congruency proportion interaction

# %%
if 'Stimulus_c25' in output_names:
    congruencySigElectrodes = get_significant_electrodes_by_effect(significant_effects_structure_timeWindowInteraction, 'congruency')
    congruencyProportionSigElectrodes = get_significant_electrodes_by_effect(significant_effects_structure_timeWindowInteraction, 'congruencyProportion')
    congruencyCongruencyProportionInteractionSigElectrodes = get_significant_electrodes_by_effect(significant_effects_structure_timeWindowInteraction, 'congruency:congruencyProportion')
    allEffectSensitiveElectrodes = get_significant_electrodes_by_effect(significant_effects_structure_timeWindowInteraction, ['congruency', 'congruencyProportion', 'congruency:congruencyProportion'])

# %% [markdown]
# this is for switch type x switch proportion interaction

# %%
if 'Stimulus_s25' in output_names:
    switchTypeSigElectrodes = get_significant_electrodes_by_effect(significant_effects_structure_timeWindowInteraction, 'switchType')
    switchProportionSigElectrodes = get_significant_electrodes_by_effect(significant_effects_structure_timeWindowInteraction, 'switchProportion')
    switchTypeSwitchProportionInteractionSigElectrodes = get_significant_electrodes_by_effect(significant_effects_structure_timeWindowInteraction, 'switchType:switchProportion')
    allEffectSensitiveElectrodes = get_significant_electrodes_by_effect(significant_effects_structure_timeWindowInteraction, ['switchType', 'switchProportion', 'switchType:switchProportion'])

# %% [markdown]
# remake overall averages but just for the chosen electrodes. This should probably just be incorporated into process_data_for_roi (can pass in sig_electrodes as an input) but too lazy to do that right now..5/20.

# %%
# Time windows

original_time_indices = {
    'firstHalfSecond': (2048, 3072),
    'secondHalfSecond': (3072, 4096)
}

sampling_rate_ratio = sampling_rate / 2048.0 # account for decimation factor

# Convert the time indices to the new sampling rate
time_indices = {
    key: (int(start * sampling_rate_ratio), int(end * sampling_rate_ratio))
    for key, (start, end) in original_time_indices.items()
}

start_idx_firstHalfSecond, end_idx_firstHalfSecond = time_indices['firstHalfSecond']
start_idx_secondHalfSecond, end_idx_secondHalfSecond = time_indices['secondHalfSecond']
start_idx_fullSecond, end_idx_fullSecond = start_idx_firstHalfSecond, end_idx_secondHalfSecond

def is_dict_of_dicts(d):
    """Check if the input is a dictionary of dictionaries."""
    return isinstance(d, dict) and all(isinstance(val, dict) for val in d.values())

def get_sig_electrodes(sig_electrodes, roi, sub):
    """Get significant electrodes based on the structure of sig_electrodes. Significant effects structure removes roi info, but sig_electrodes_per_subject_roi keeps it.
    maybe I should make these consistent... 4/5"""
    if is_dict_of_dicts(sig_electrodes):
        return sig_electrodes[roi].get(sub, [])
    else:
        return sig_electrodes.get(sub, [])
    
# this massive function needs to be split up and replace its above non-function form once it works 4/5.
def get_average_data_for_specific_electrodes(subjects, rois, output_names, sig_electrodes):
    # Assuming output_names contains all four conditions
    # Initializing dictionaries
    output_data_trialAvg_lists = utils.initialize_output_data(rois, output_names)
    output_data_trialStd_lists = utils.initialize_output_data(rois, output_names)
    output_data_timeAvg_firstHalfSecond_lists = utils.initialize_output_data(rois, output_names)
    output_data_timeAvg_secondHalfSecond_lists = utils.initialize_output_data(rois, output_names)
    output_data_timeAvg_fullSecond_lists = utils.initialize_output_data(rois, output_names)

    # Initialize a dictionary to hold mappings
    overall_electrode_mapping = []

    # Initialize a dictionary to hold mappings for each ROI
    electrode_mapping_per_roi = {roi: [] for roi in rois}
    for sub in subjects:
        for roi in rois:
            for output_name in output_names:
                # Determine significant electrodes for the current ROI and subject. Hmm not sure if this will work if I have more than one ROI for the 
                # get_significant_electrodes_by_effect electrodes.. 4/5
                sig_electrodes_this_sub = get_sig_electrodes(sig_electrodes, roi, sub)
                print('sub', sub)
                print('sig elecs:', sig_electrodes_this_sub)
                
                if not sig_electrodes_this_sub:  # Skip if no significant electrodes
                    continue
                            
                for electrode in sig_electrodes_this_sub:
                    # For each significant electrode, append a tuple to the mapping list
                    # Tuple format: (Subject ID, ROI, Electrode Name, Index in List)
                    # The index can be the current length of the list before appending
                    index = len(overall_electrode_mapping)
                    overall_electrode_mapping.append((sub, roi, electrode, output_name, index))  

                    # For each significant electrode, append a tuple to the mapping list of the corresponding ROI
                    # Tuple format: (Subject ID, Electrode Name, Index in List for this ROI)
                    index = len(electrode_mapping_per_roi[roi])  # Get the current length of the list for this ROI
                    electrode_mapping_per_roi[roi].append((sub, electrode, output_name, index))
                    
                # Load trial-level data for the current condition and pick significant electrodes
                epochs = subjects_mne_objects[sub][output_name]['HG_ev1_power_rescaled'].copy().pick_channels(sig_electrodes_this_sub)
                # print(epochs.get_data().shape)
                # Calculate averages for each time window
                trial_avg, trial_std, time_avg_firstHalfSecond = utils.filter_and_average_epochs(epochs, start_idx_firstHalfSecond, end_idx_firstHalfSecond)
                _, _, time_avg_secondHalfSecond = utils.filter_and_average_epochs(epochs, start_idx_secondHalfSecond, end_idx_secondHalfSecond)
                _, _, time_avg_fullSecond = utils.filter_and_average_epochs(epochs, start_idx_fullSecond, end_idx_fullSecond)
                print('time avg full second shape:', time_avg_fullSecond.shape)

                # Append the results to their respective lists
                output_data_trialAvg_lists[output_name][roi].append(trial_avg)
                output_data_trialStd_lists[output_name][roi].append(trial_std)
                output_data_timeAvg_firstHalfSecond_lists[output_name][roi].append(time_avg_firstHalfSecond)
                output_data_timeAvg_secondHalfSecond_lists[output_name][roi].append(time_avg_secondHalfSecond)
                output_data_timeAvg_fullSecond_lists[output_name][roi].append(time_avg_fullSecond)


    # After collecting all data, concatenate across subjects for each roi and condition
    concatenated_trialAvg_data = {}
    concatenated_trialStd_data = {}

    for roi in rois:
        concatenated_trialAvg_data[roi] = {}
        concatenated_trialStd_data[roi] = {}

        for output_name in output_names:
            concatenated_trialAvg_data[roi][output_name] = np.concatenate(output_data_trialAvg_lists[output_name][roi], axis=0)
            concatenated_trialStd_data[roi][output_name] = np.concatenate(output_data_trialStd_lists[output_name][roi], axis=0)


    # Calculate mean and SEM across electrodes for all time windows and rois
    overall_averages = {}
    overall_sems = {}
    mean_and_sem = {roi: {output_name: {} for output_name in output_names} for roi in rois}

    for roi in rois:
        overall_averages[roi] = {}
        overall_sems[roi] = {}
        for output_name in output_names:
            trialAvg_data = concatenated_trialAvg_data[roi][output_name]
            overall_averages[roi][output_name] = np.nanmean(trialAvg_data, axis=0)
            overall_sems[roi][output_name] = np.std(trialAvg_data, axis=0, ddof=1) / np.sqrt(trialAvg_data.shape[0])
            mean_and_sem[roi][output_name] = {'mean': overall_averages[roi][output_name], 'sem': overall_sems[roi][output_name]}

    return concatenated_trialAvg_data, concatenated_trialStd_data, mean_and_sem, output_data_timeAvg_fullSecond_lists

# %% [markdown]
# this is for congruency x switch type interaction

# %%
if 'Stimulus_cr' in output_names:
    concatenated_trialAvg_data_congruencySigElectrodes, concatenated_trialStd_data_congruencySigElectrodes, congruencySigElectrodesMeanAndSEM, congruencySigElectrodes_timeAvg_fullSecondLists = get_average_data_for_specific_electrodes(subjects, rois, output_names, congruencySigElectrodes)
    concatenated_trialAvg_data_switchTypeSigElectrodes, concatenated_trialStd_data_switchTypeSigElectrodes, switchTypeSigElectrodesMeanAndSEM, switchTypeSigElectrodes_timeAvg_fullSecondLists = get_average_data_for_specific_electrodes(subjects, rois, output_names, switchTypeSigElectrodes)
    concatenated_trialAvg_data_congruencySwitchTypeInteractionSigElectrodes, concatenated_trialStd_data_congruencySwitchTypeInteractionSigElectrodes, congruencySwitchTypeInteractionSigElectrodesMeanAndSEM, congruencySwitchTypeInteractionSigElectrodes_timeAvg_fullSecondLists = get_average_data_for_specific_electrodes(subjects, rois, output_names, congruencySwitchTypeInteractionSigElectrodes)
    concatenated_trialAvg_data_congruencySwitchTypeInteractionSigElectrodes, concatenated_trialStd_data_congruencySwitchTypeInteractionSigElectrodes, allEffectSensitiveElectrodesMeanAndSEM, allEffectSensitiveElectrodes_timeAvg_fullSecondLists = get_average_data_for_specific_electrodes(subjects, rois, output_names, allEffectSensitiveElectrodes)

# %% [markdown]
# this is for congruency x congruency proportion interaction

# %%
if 'Stimulus_c25' in output_names:
    concatenated_trialAvg_data_congruencySigElectrodes, concatenated_trialStd_data_congruencySigElectrodes, congruencySigElectrodesMeanAndSEM, congruencySigElectrodes_timeAvg_fullSecondLists = get_average_data_for_specific_electrodes(subjects, rois, output_names, congruencySigElectrodes)
    concatenated_trialAvg_data_congruencyProportionSigElectrodes, concatenated_trialStd_data_congruencyProportionSigElectrodes, congruencyProportionSigElectrodesMeanAndSEM, congruencyProportionSigElectrodes_timeAvg_fullSecondLists = get_average_data_for_specific_electrodes(subjects, rois, output_names, congruencyProportionSigElectrodes)
    concatenated_trialAvg_data_congruencyCongruencyProportionInteractionSigElectrodes, concatenated_trialStd_data_congruencyCongruencyProportionInteractionSigElectrodes, congruencyCongruencyProportionInteractionSigElectrodesMeanAndSEM, congruencyCongruencyProportionInteractionSigElectrodes_timeAvg_fullSecondLists = get_average_data_for_specific_electrodes(subjects, rois, output_names, congruencyCongruencyProportionInteractionSigElectrodes)
    concatenated_trialAvg_data_allEffectSensitiveSigElectrodes, concatenated_trialStd_data_allEffectSensitiveSigElectrodes, allEffectSensitiveElectrodesMeanAndSEM, allEffectSensitiveElectrodes_timeAvg_fullSecondLists = get_average_data_for_specific_electrodes(subjects, rois, output_names, allEffectSensitiveElectrodes)

# %% [markdown]
# this for switch type x switch proportion interaction

# %%
if 'Stimulus_s25' in output_names:
    concatenated_trialAvg_data_switchTypeSigElectrodes, concatenated_trialStd_data_switchTypeSigElectrodes, switchTypeSigElectrodesMeanAndSEM, switchTypeSigElectrodes_timeAvg_fullSecondLists = get_average_data_for_specific_electrodes(subjects, rois, output_names, switchTypeSigElectrodes)
    concatenated_trialAvg_data_switchProportionSigElectrodes, concatenated_trialStd_data_switchProportionSigElectrodes, switchProportionSigElectrodesMeanAndSEM, switchProportionSigElectrodes_timeAvg_fullSecondLists = get_average_data_for_specific_electrodes(subjects, rois, output_names, switchProportionSigElectrodes)
    concatenated_trialAvg_data_switchTypeSwitchProportionInteractionSigElectrodes, concatenated_trialStd_data_switchTypeSwitchProportionInteractionSigElectrodes, switchTypeSwitchProportionInteractionSigElectrodesMeanAndSEM, switchTypeSwitchProportionInteractionSigElectrodes_timeAvg_fullSecondLists = get_average_data_for_specific_electrodes(subjects, rois, output_names, switchTypeSwitchProportionInteractionSigElectrodes)
    concatenated_trialAvg_data_allEffectSensitiveSigElectrodes, concatenated_trialStd_data_allEffectSensitiveSigElectrodes, allEffectSensitiveElectrodesMeanAndSEM, allEffectSensitiveElectrodes_timeAvg_fullSecondLists = get_average_data_for_specific_electrodes(subjects, rois, output_names, allEffectSensitiveElectrodes)

# %% [markdown]
# finally plot the average traces for each output name for these chosen electrodes (this is for congruency x switch type)
# 
# *NOTE: This code only works for one roi. So go back and rerun everything with just one ROI for now. And fix this later (or just redo it all) 10/1.

# %%
if len(rois) > 1:
    raise AssertionError("This conflates multiple ROIs right now, just load in one ROI at a time. Sorry, 5/20. Make better later.")
elif 'Stimulus_cr' in output_names:
    plot_interact_effects_modular_roi(f'{rois[0]}', save_dir, 'congruency_switchType_congruencySigElectrodes', congruencySigElectrodesMeanAndSEM, output_names, plotting_parameters)

# %%
if len(rois) > 1:
    raise AssertionError("This conflates multiple ROIs right now, just load in one ROI at a time. Sorry, 5/20. Make better later.")
if 'Stimulus_cr' in output_names:
    plot_interact_effects_modular_roi(f'{rois[0]}', save_dir, 'congruency_switchType_switchTypeSigElectrodes', switchTypeSigElectrodesMeanAndSEM, output_names, plotting_parameters)

# %%
if len(rois) > 1:
    raise AssertionError("This conflates multiple ROIs right now, just load in one ROI at a time. Sorry, 5/20. Make better later.")
if 'Stimulus_cr' in output_names:
    plot_interact_effects_modular_roi(f'{rois[0]}', save_dir, 'congruency_switchType_congruencySwitchTypeInteractionSigElectrodes', congruencySwitchTypeInteractionSigElectrodesMeanAndSEM, output_names, plotting_parameters)

# %%
if len(rois) > 1:
    raise AssertionError("This conflates multiple ROIs right now, just load in one ROI at a time. Sorry, 5/20. Make better later.")
if 'Stimulus_cr' in output_names:
    plot_interact_effects_modular_roi(f'{rois[0]}', save_dir, 'congruency_switchType_allEffectSensitiveElectrodes', allEffectSensitiveElectrodesAverage, allEffectSensitiveElectrodesSEM, output_names, plotting_parameters)

# %% [markdown]
# plot the four traces for congruency x congruency proportion for electrodes sensitive to the interaction

# %%
if len(rois) > 1:
    raise AssertionError("This conflates multiple ROIs right now, just load in one ROI at a time. Sorry, 5/20. Make better later.")
elif 'Stimulus_c25' in output_names:
    plot_interact_effects_modular_roi(f'{rois[0]}', save_dir, 'congruency_congruencyProportion_congruencyCongruencyProportionInteractionSigElectrodes', congruencyCongruencyProportionInteractionSigElectrodesMeanAndSEM, output_names, plotting_parameters)

# %% [markdown]
# plot four traces for switch x switch proportion for electrodes sensitive to the interaction

# %%
if len(rois) > 1:
    raise AssertionError("This conflates multiple ROIs right now, just load in one ROI at a time. Sorry, 5/20. Make better later.")
elif 'Stimulus_s25' in output_names:
    plot_interact_effects_modular_roi(f'{rois[0]}', save_dir, 'switchType_switchProportion_switchTypeSwitchProportionInteractionSigElectrodes', switchTypeSwitchProportionInteractionSigElectrodesMeanAndSEM, output_names, plotting_parameters)

# %% [markdown]
# plot the four traces for switch type x switch proportion for electrodes sensitive to the interaction

# %% [markdown]
# plot (ir - cr) vs (is - cs)  
# 4/5 - functionize this stuff later

# %% [markdown]
# SEM diff = sqrt(SEM1^2 + SEM2^2)

# %%
diff_ir_cr = {}  # Difference between IR and CR
diff_is_cs = {}  # Difference between IS and CS

for roi in rois:
    diff_ir_cr[roi] = congruencySwitchTypeInteractionSigElectrodesMeanAndSEM[roi]['Stimulus_ir']['mean'] - congruencySwitchTypeInteractionSigElectrodesMeanAndSEM[roi]['Stimulus_cr']['mean']
    diff_is_cs[roi] = congruencySwitchTypeInteractionSigElectrodesMeanAndSEM[roi]['Stimulus_is']['mean'] - congruencySwitchTypeInteractionSigElectrodesMeanAndSEM[roi]['Stimulus_cs']['mean']

sem_diff_ir_cr = {}
sem_diff_is_cs = {}

for roi in rois:
    sem_diff_ir_cr[roi] = np.sqrt(np.power(congruencySwitchTypeInteractionSigElectrodesMeanAndSEM[roi]['Stimulus_ir']['sem'], 2) + np.power(congruencySwitchTypeInteractionSigElectrodesMeanAndSEM[roi]['Stimulus_cr']['sem'], 2))
    sem_diff_is_cs[roi] = np.sqrt(np.power(congruencySwitchTypeInteractionSigElectrodesMeanAndSEM[roi]['Stimulus_is']['sem'], 2) + np.power(congruencySwitchTypeInteractionSigElectrodesMeanAndSEM[roi]['Stimulus_cs']['sem'], 2))

# Dynamically select the first subject and use it to extract times
first_subject_id = next(iter(subjects_mne_objects))
example_output_name = next(iter(subjects_mne_objects[first_subject_id]))
times = subjects_mne_objects[first_subject_id][example_output_name]['HG_ev1_power_rescaled'].times

roi = 'lpfc'

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(times, diff_ir_cr[roi], label='IR - CR', color='black', linestyle='-')
ax.plot(times, diff_is_cs[roi], label='IS - CS', color='black', linestyle='--')

ax.fill_between(times, diff_ir_cr[roi] - sem_diff_ir_cr[roi], diff_ir_cr[roi] + sem_diff_ir_cr[roi], alpha=0.2, color='black')
ax.fill_between(times, diff_is_cs[roi] - sem_diff_is_cs[roi], diff_is_cs[roi] + sem_diff_is_cs[roi], alpha=0.2, color='black')

# # Overlay a dotted vertical line at time = 0.5
# ax.axvline(x=0.5, color='k', linestyle='--', linewidth=1)

# Remove top and right borders
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ax.set_xlabel('Time from Stimulus Onset (s)', fontsize=20)
# ax.set_ylabel('Z-score Difference', fontsize=20)
# ax.legend(fontsize=20, loc='upper left')

# Make the x and y ticks bigger
ax.tick_params(axis='x', labelsize=24)  # Adjust x-axis tick label size
ax.tick_params(axis='y', labelsize=24)  # Adjust y-axis tick label size

plt.tight_layout()
plt.legend()
save_path = os.path.join(save_dir, f'avg_{roi}_IR-CRvsIS-CS_power.png')
plt.savefig(save_path)
plt.close()

# %% [markdown]
# now let's do switch cost as a function of congruency

# %%
diff_is_ir = {}  # Difference between IR and CR
diff_cs_cr = {}  # Difference between IS and CS

# need to change these to the mean and sem format. As shown above.
for roi in rois:
    diff_is_ir[roi] = congruencySwitchTypeInteractionSigElectrodesMeanAndSEM[roi]['Stimulus_is']['mean'] - congruencySwitchTypeInteractionSigElectrodesMeanAndSEM[roi]['Stimulus_ir']['mean']
    diff_cs_cr[roi] = congruencySwitchTypeInteractionSigElectrodesMeanAndSEM[roi]['Stimulus_cs']['mean'] - congruencySwitchTypeInteractionSigElectrodesMeanAndSEM[roi]['Stimulus_cr']['mean']

sem_diff_is_ir = {}
sem_diff_cs_cr = {}

for roi in rois:
    sem_diff_is_ir[roi] = np.sqrt(np.power(congruencySwitchTypeInteractionSigElectrodesMeanAndSEM[roi]['Stimulus_is']['sem'], 2) + np.power(congruencySwitchTypeInteractionSigElectrodesMeanAndSEM[roi]['Stimulus_ir']['sem'], 2))
    sem_diff_cs_cr[roi] = np.sqrt(np.power(congruencySwitchTypeInteractionSigElectrodesMeanAndSEM[roi]['Stimulus_cs']['sem'], 2) + np.power(congruencySwitchTypeInteractionSigElectrodesMeanAndSEM[roi]['Stimulus_cr']['sem'], 2))

# Dynamically select the first subject and use it to extract times
first_subject_id = next(iter(subjects_mne_objects))
example_output_name = next(iter(subjects_mne_objects[first_subject_id]))
times = subjects_mne_objects[first_subject_id][example_output_name]['HG_ev1_power_rescaled'].times

roi = 'lpfc'

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(times, diff_is_ir[roi], label='IS - IR', color='black', linestyle='-')
ax.plot(times, diff_cs_cr[roi], label='CS - CR', color='black', linestyle='--')

ax.fill_between(times, diff_is_ir[roi] - sem_diff_is_ir[roi], diff_is_ir[roi] + sem_diff_is_ir[roi], alpha=0.2, color='black')
ax.fill_between(times, diff_cs_cr[roi] - sem_diff_cs_cr[roi], diff_cs_cr[roi] + sem_diff_cs_cr[roi], alpha=0.2, color='black')

# # Overlay a dotted vertical line at time = 0.5
# ax.axvline(x=0.5, color='k', linestyle='--', linewidth=1)

# Remove top and right borders
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.set_xlabel('Time from Stimulus Onset (s)', fontsize=20)
ax.set_ylabel('Z-score Difference', fontsize=20)
ax.legend(fontsize=20, loc='upper left')

# Make the x and y ticks bigger
ax.tick_params(axis='x', labelsize=20)  # Adjust x-axis tick label size
ax.tick_params(axis='y', labelsize=20)  # Adjust y-axis tick label size

plt.tight_layout()
plt.legend()
save_path = os.path.join(save_dir, f'avg_{roi}_IS-IRvsCS-CR_power.png')
plt.savefig(save_path)
plt.close()

# %% [markdown]
# let's also plot i vs c 4/5

# %%
avg_ir_is = {}  # Average of IR and IS
avg_cr_cs = {}  # Average of CR and CS

# change to mean and sem formatting as shown above 5/20. But more realistically, make a function to do all of this.
for roi in rois:
    avg_ir_is[roi] = (congruencySigElectrodesMeanAndSEM[roi]['Stimulus_ir']['mean'] + congruencySigElectrodesMeanAndSEM[roi]['Stimulus_is']['mean']) / 2
    avg_cr_cs[roi] = (congruencySigElectrodesMeanAndSEM[roi]['Stimulus_cr']['mean'] + congruencySigElectrodesMeanAndSEM[roi]['Stimulus_cs']['mean']) / 2

# assuming equal sample sizes, which i think we should have
avg_sem_ir_is = {}
avg_sem_cr_cs = {}

for roi in rois:
    avg_sem_ir_is[roi] = np.sqrt((np.power(congruencySigElectrodesMeanAndSEM[roi]['Stimulus_ir']['sem'], 2) + np.power(congruencySigElectrodesMeanAndSEM[roi]['Stimulus_is']['sem'], 2)) / 2)
    avg_sem_cr_cs[roi] = np.sqrt((np.power(congruencySigElectrodesMeanAndSEM[roi]['Stimulus_cr']['sem'], 2) + np.power(congruencySigElectrodesMeanAndSEM[roi]['Stimulus_cs']['sem'], 2)) / 2)

# Dynamically select the first subject and use it to extract times
first_subject_id = next(iter(subjects_mne_objects))
example_output_name = next(iter(subjects_mne_objects[first_subject_id]))
times = subjects_mne_objects[first_subject_id][example_output_name]['HG_ev1_power_rescaled'].times

roi = 'lpfc'

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(times, avg_cr_cs[roi], label='Congruent', color='red', linestyle='-')
ax.plot(times, avg_ir_is[roi], label='Incongruent', color='red', linestyle='--')

ax.fill_between(times, avg_ir_is[roi] - avg_sem_ir_is[roi], avg_ir_is[roi] + avg_sem_ir_is[roi], alpha=0.2, color='red')
ax.fill_between(times, avg_cr_cs[roi] - avg_sem_cr_cs[roi], avg_cr_cs[roi] + avg_sem_cr_cs[roi], alpha=0.2, color='red')

# # Overlay a dotted vertical line at time = 0.5
# ax.axvline(x=0.5, color='k', linestyle='--', linewidth=1)

# Remove top and right borders
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ax.set_xlabel('Time from Stimulus Onset (s)', fontsize=20)
# ax.set_ylabel('Z-score Difference', fontsize=20)
# ax.legend(fontsize=20, loc='upper left')

# Make the x and y ticks bigger
ax.tick_params(axis='x', labelsize=24)  # Adjust x-axis tick label size
ax.tick_params(axis='y', labelsize=24)  # Adjust y-axis tick label size

plt.tight_layout()
# plt.legend()
save_path = os.path.join(save_dir, f'avg_{roi}_IvsC_congruencySigElectrodes_power.png')
plt.savefig(save_path)
plt.close()

# %% [markdown]
# now lets do i - c for the switch type main effect electrodes

# %%
congruencySigElectrodesMeanAndSEM

# %%
switchTypeSigElectrodesMeanAndSEM

# %%
avg_ir_is = {}  # Average of IR and IS
avg_cr_cs = {}  # Average of CR and CS

for roi in rois:
    avg_ir_is[roi] = (switchTypeSigElectrodesMeanAndSEM[roi]['Stimulus_ir']['mean'] + switchTypeSigElectrodesMeanAndSEM[roi]['Stimulus_is']['mean']) / 2
    avg_cr_cs[roi] = (switchTypeSigElectrodesMeanAndSEM[roi]['Stimulus_cr']['mean'] + switchTypeSigElectrodesMeanAndSEM[roi]['Stimulus_cs']['mean']) / 2

# assuming equal sample sizes, which i think we should have
avg_sem_ir_is = {}
avg_sem_cr_cs = {}

for roi in rois:
    avg_sem_ir_is[roi] = np.sqrt((np.power(switchTypeSigElectrodesMeanAndSEM[roi]['Stimulus_ir']['sem'], 2) + np.power(switchTypeSigElectrodesMeanAndSEM[roi]['Stimulus_is']['sem'], 2)) / 2)
    avg_sem_cr_cs[roi] = np.sqrt((np.power(switchTypeSigElectrodesMeanAndSEM[roi]['Stimulus_cr']['sem'], 2) + np.power(switchTypeSigElectrodesMeanAndSEM[roi]['Stimulus_cs']['sem'], 2)) / 2)

# Dynamically select the first subject and use it to extract times
first_subject_id = next(iter(subjects_mne_objects))
example_output_name = next(iter(subjects_mne_objects[first_subject_id]))
times = subjects_mne_objects[first_subject_id][example_output_name]['HG_ev1_power_rescaled'].times

roi = 'lpfc'

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(times, avg_ir_is[roi], label='Congruent', color='red', linestyle='-')
ax.plot(times, avg_cr_cs[roi], label='Incongruent', color='red', linestyle='--')

ax.fill_between(times, avg_ir_is[roi] - avg_sem_ir_is[roi], avg_ir_is[roi] + avg_sem_ir_is[roi], alpha=0.2, color='red')
ax.fill_between(times, avg_cr_cs[roi] - avg_sem_cr_cs[roi], avg_cr_cs[roi] + avg_sem_cr_cs[roi], alpha=0.2, color='red')

# # Overlay a dotted vertical line at time = 0.5
# ax.axvline(x=0.5, color='k', linestyle='--', linewidth=1)

# Remove top and right borders
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.set_xlabel('Time from Stimulus Onset (s)', fontsize=20)
ax.set_ylabel('Z-score Difference', fontsize=20)
ax.legend(fontsize=20, loc='upper left')

# Make the x and y ticks bigger
ax.tick_params(axis='x', labelsize=20)  # Adjust x-axis tick label size
ax.tick_params(axis='y', labelsize=20)  # Adjust y-axis tick label size

plt.tight_layout()
# plt.legend()
save_path = os.path.join(save_dir, f'avg_{roi}_IvsC_switchTypeSigElectrodes_power.png')
plt.savefig(save_path)
plt.close()

# %% [markdown]
# now let's plot switch vs repeat for the switch type main effect electrodes

# %%
avg_ir_cr = {}  # Average of IR and CR
avg_is_cs = {}  # Average of IS and CS

for roi in rois:
    avg_ir_cr[roi] = (switchTypeSigElectrodesMeanAndSEM[roi]['Stimulus_ir']['mean'] + switchTypeSigElectrodesMeanAndSEM[roi]['Stimulus_cr']['mean']) / 2
    avg_is_cs[roi] = (switchTypeSigElectrodesMeanAndSEM[roi]['Stimulus_is']['mean'] + switchTypeSigElectrodesMeanAndSEM[roi]['Stimulus_cs']['mean']) / 2

# assuming equal sample sizes, which i think we should have
avg_sem_ir_cr = {}
avg_sem_is_cs = {}

for roi in rois:
    avg_sem_ir_cr[roi] = np.sqrt((np.power(switchTypeSigElectrodesMeanAndSEM[roi]['Stimulus_ir']['sem'], 2) + np.power(switchTypeSigElectrodesMeanAndSEM[roi]['Stimulus_cr']['sem'], 2)) / 2)
    avg_sem_is_cs[roi] = np.sqrt((np.power(switchTypeSigElectrodesMeanAndSEM[roi]['Stimulus_is']['sem'], 2) + np.power(switchTypeSigElectrodesMeanAndSEM[roi]['Stimulus_cs']['sem'], 2)) / 2)

# Dynamically select the first subject and use it to extract times
first_subject_id = next(iter(subjects_mne_objects))
example_output_name = next(iter(subjects_mne_objects[first_subject_id]))
times = subjects_mne_objects[first_subject_id][example_output_name]['HG_ev1_power_rescaled'].times

roi = 'lpfc'

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(times, avg_ir_cr[roi], label='Repeat', color='blue', linestyle='-')
ax.plot(times, avg_is_cs[roi], label='Switch', color='blue', linestyle='--')

ax.fill_between(times, avg_ir_cr[roi] - avg_sem_ir_cr[roi], avg_ir_cr[roi] + avg_sem_ir_cr[roi], alpha=0.2, color='blue')
ax.fill_between(times, avg_is_cs[roi] - avg_sem_is_cs[roi], avg_is_cs[roi] + avg_sem_is_cs[roi], alpha=0.2, color='blue')

# # Overlay a dotted vertical line at time = 0.5
# ax.axvline(x=0.5, color='k', linestyle='--', linewidth=1)

# Remove top and right borders
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ax.set_xlabel('Time from Stimulus Onset (s)', fontsize=20)
# ax.set_ylabel('Z-score Difference', fontsize=20)
# ax.legend(fontsize=20, loc='upper left')

# Make the x and y ticks bigger
ax.tick_params(axis='x', labelsize=24)  # Adjust x-axis tick label size
ax.tick_params(axis='y', labelsize=24)  # Adjust y-axis tick label size

plt.tight_layout()
# plt.legend()
save_path = os.path.join(save_dir, f'avg_{roi}_SvsR_switchTypeSigElectrodes_power.png')
plt.savefig(save_path)
plt.close()

# %%
congruencySigElectrodes

# %%
switchTypeSigElectrodes

# %%
congruencySwitchTypeInteractionSigElectrodes

# %% [markdown]
# interestingly, the switch vs repeat for congruency main effect electrodes is quite different too

# %%
avg_ir_cr = {}  # Average of IR and CR
avg_is_cs = {}  # Average of IS and CS

for roi in rois:
    avg_ir_cr[roi] = (congruencySigElectrodesMeanAndSEM[roi]['Stimulus_ir']['mean'] + congruencySigElectrodesMeanAndSEM[roi]['Stimulus_cr']['mean']) / 2
    avg_is_cs[roi] = (congruencySigElectrodesMeanAndSEM[roi]['Stimulus_is']['mean'] + congruencySigElectrodesMeanAndSEM[roi]['Stimulus_cs']['mean']) / 2

# assuming equal sample sizes, which i think we should have
avg_sem_ir_cr = {}
avg_sem_is_cs = {}

for roi in rois:
    avg_sem_ir_cr[roi] = np.sqrt((np.power(congruencySigElectrodesMeanAndSEM[roi]['Stimulus_ir']['sem'], 2) + np.power(congruencySigElectrodesMeanAndSEM[roi]['Stimulus_cr']['sem'], 2)) / 2)
    avg_sem_is_cs[roi] = np.sqrt((np.power(congruencySigElectrodesMeanAndSEM[roi]['Stimulus_is']['sem'], 2) + np.power(congruencySigElectrodesMeanAndSEM[roi]['Stimulus_cs']['sem'], 2)) / 2)

# Dynamically select the first subject and use it to extract times
first_subject_id = next(iter(subjects_mne_objects))
example_output_name = next(iter(subjects_mne_objects[first_subject_id]))
times = subjects_mne_objects[first_subject_id][example_output_name]['HG_ev1_power_rescaled'].times

roi = 'lpfc'

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(times, avg_ir_cr[roi], label='repeat', color='blue', linestyle='-')
ax.plot(times, avg_is_cs[roi], label='switch', color='blue', linestyle='--')

ax.fill_between(times, avg_ir_cr[roi] - avg_sem_ir_cr[roi], avg_ir_cr[roi] + avg_sem_ir_cr[roi], alpha=0.2, color='blue')
ax.fill_between(times, avg_is_cs[roi] - avg_sem_is_cs[roi], avg_is_cs[roi] + avg_sem_is_cs[roi], alpha=0.2, color='blue')

# # Overlay a dotted vertical line at time = 0.5
# ax.axvline(x=0.5, color='k', linestyle='--', linewidth=1)

# Remove top and right borders
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.set_xlabel('Time from Stimulus Onset (s)', fontsize=20)
ax.set_ylabel('Z-score Difference', fontsize=20)
ax.legend(fontsize=20, loc='upper left')

# Make the x and y ticks bigger
ax.tick_params(axis='x', labelsize=20)  # Adjust x-axis tick label size
ax.tick_params(axis='y', labelsize=20)  # Adjust y-axis tick label size

plt.tight_layout()
# plt.legend()
save_path = os.path.join(save_dir, f'avg_{roi}_SvsR_congruencySigElectrodes_power.png')
plt.savefig(save_path)
plt.close()

# %% [markdown]
# ### make bar plots windowed from 0 to 1 s of my two conditions 4/9  
# greg wants to see these

# %% [markdown]
# first get mean and sem across electrodes from 0 to 1 for various conditions

# %%
def calculate_means_sems_from_fullSecond_averages(timeAvg_fullSecond_lists, rois, output_names):
    means = {}
    sems = {}
    
    for roi in rois:
        means[roi] = {}
        sems[roi] = {}
        for output_name in output_names:
            # Handle each electrode's full-second average individually
            full_second_averages = timeAvg_fullSecond_lists[output_name][roi]
            # Assuming full_second_averages is a list of numpy arrays, one per electrode
            
            # Initialize lists to store the mean for each electrode
            electrode_means = []
            
            for electrode_data in full_second_averages:
                # Calculate the mean for this electrode in the full second window
                electrode_mean = np.nanmean(electrode_data)
                electrode_means.append(electrode_mean)
            
            # Convert the list of means to a NumPy array for further calculation
            electrode_means = np.array(electrode_means)
            
            # Calculate the overall mean and SEM across electrodes
            condition_mean = np.nanmean(electrode_means)
            condition_sem = np.std(electrode_means, ddof=1) / np.sqrt(len(electrode_means))
            
            means[roi][output_name] = condition_mean
            sems[roi][output_name] = condition_sem
    
    return means, sems

# Now, calculate the means and SEMs
means_windowed_full_second, sems_windowed_full_second = calculate_means_sems_from_fullSecond_averages(congruencySwitchTypeInteractionSigElectrodes_timeAvg_fullSecondLists, rois, output_names)
congruency_means_windowed_full_second, congruency_sems_windowed_full_second = calculate_means_sems_from_fullSecond_averages(congruencySigElectrodes_timeAvg_fullSecondLists, rois, output_names)
switchType_means_windowed_full_second, switchType_sems_windowed_full_second = calculate_means_sems_from_fullSecond_averages(switchTypeSigElectrodes_timeAvg_fullSecondLists, rois, output_names)

# %% [markdown]
# now plot all four conditions as bars on one plot for the interaction effect 4/10

# %%
import matplotlib.pyplot as plt
import numpy as np

# Corrected condition names based on your dataset
conditions = [
    'Stimulus_cr',  # Congruent Repeat
    'Stimulus_ir',  # Incongruent Repeat
    'Stimulus_cs',  # Congruent Switch
    'Stimulus_is'   # Incongruent Switch
]

# Retrieve the means and SEMs for each condition for plotting
means = [means_windowed_full_second[roi][cond] for cond in conditions]
sems = [sems_windowed_full_second[roi][cond] for cond in conditions]

# Colors for each condition for plotting
group_colors = ['pink', 'red', 'pink', 'red']  # Mapping colors to Congruent/Incongruent

# Plotting
fig, ax = plt.subplots()
bar_width = 0.35  # Width of the bars
index = np.arange(2)  # Two groups: Repeat and Switch

# Creating bars for each group
for i, (mean, sem, color) in enumerate(zip(means, sems, group_colors)):
    position = index[i // 2] + (i % 2 - 0.5) * bar_width
    ax.bar(position, mean, yerr=sem, capsize=5, color=color, width=bar_width)

# Ensure no labels or ticks are shown on the x-axis
ax.set_xticks([])  # No x-tick marks
ax.tick_params(axis='x', which='both', length=0)  # No x-tick marks
ax.tick_params(axis='y', labelsize=32)  # Adjust y-tick label size as needed

# Customizing the plot (commented sections are optional customizations)
# ax.set_ylabel('Average Z-score', fontsize=14)
# ax.set_title('Average Z-score From Baseline by Condition and Type (Full Second)', fontsize=16)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
# plt.legend()
save_path = os.path.join(save_dir, f'avg_{roi}_CSvsCRvsIRvsIS_fullSecond.png')
plt.savefig(save_path)
plt.close()


# %% [markdown]
# now plot congruent vs incongruent 4/10

# %%
import matplotlib.pyplot as plt
import numpy as np

# Assuming roi, congruency_means_windowed_full_second, and congruency_sems_windowed_full_second are defined

# Conditions based on your specification
conditions = [
    'Stimulus_cr_fixationCrossBase_1sec_mirror',  # Congruent Repeat
    'Stimulus_ir_fixationCrossBase_1sec_mirror',  # Incongruent Repeat
    'Stimulus_cs_fixationCrossBase_1sec_mirror',  # Congruent Switch
    'Stimulus_is_fixationCrossBase_1sec_mirror'   # Incongruent Switch
]

# Calculating combined means and SEMs for Congruent and Incongruent conditions
combined_means = [
    np.mean([
        congruency_means_windowed_full_second[roi][conditions[0]],  # CR mean
        congruency_means_windowed_full_second[roi][conditions[2]]   # CS mean
    ]),
    np.mean([
        congruency_means_windowed_full_second[roi][conditions[1]],  # IR mean
        congruency_means_windowed_full_second[roi][conditions[3]]   # IS mean
    ])
]

combined_sems = [
    np.sqrt(np.mean([
        congruency_sems_windowed_full_second[roi][conditions[0]]**2,  # CR SEM^2
        congruency_sems_windowed_full_second[roi][conditions[2]]**2   # CS SEM^2
    ])),
    np.sqrt(np.mean([
        congruency_sems_windowed_full_second[roi][conditions[1]]**2,  # IR SEM^2
        congruency_sems_windowed_full_second[roi][conditions[3]]**2   # IS SEM^2
    ]))
]

# Plotting adjustments for the first plot
fig, ax = plt.subplots()
bar_width = 0.015  # Narrower bars

# Adjust the index to create slight separation
index = np.array([0.4375, 0.4625])  # Adjusted for slight separation

bars = ax.bar(index, combined_means, yerr=combined_sems, capsize=5, color=['pink', 'red'], width=bar_width)

# Customizing the plot
ax.set_xticks(index)
ax.set_xticklabels(['Congruent', 'Incongruent'], fontsize=24)
ax.tick_params(axis='y', labelsize=24)

plt.tight_layout()
plt.show()

# Plotting adjustments for the second plot with same formatting
fig, ax = plt.subplots(figsize=(8, 6))  # Adjust figure size as needed

# Reusing the same bar width and indices for consistency
bars = ax.bar(index, combined_means, 
              yerr=combined_sems, capsize=5, 
              color=['pink', 'red'], width=bar_width)

ax.set_xticks(index)
ax.set_xticklabels(['Congruent', 'Incongruent'], fontsize=32)
ax.tick_params(axis='y', labelsize=32)

# Optional: Uncomment the next line if you want to see the effect without plt.tight_layout()
# plt.tight_layout()

plt.show()


# %% [markdown]
# now plot switch vs repeat windowed from 0 to 1

# %%
combined_means_switch_repeat = [
    np.mean([
        switchType_means_windowed_full_second[roi][conditions[0]],  # CR mean
        switchType_means_windowed_full_second[roi][conditions[1]]   # IR mean
    ]),
    np.mean([
        switchType_means_windowed_full_second[roi][conditions[2]],  # CS mean
        switchType_means_windowed_full_second[roi][conditions[3]]   # IS mean
    ])
]

combined_sems_switch_repeat = [
    np.sqrt(np.mean([
        switchType_sems_windowed_full_second[roi][conditions[0]]**2,  # CR SEM^2
        switchType_sems_windowed_full_second[roi][conditions[1]]**2   # IR SEM^2
    ])),
    np.sqrt(np.mean([
        switchType_sems_windowed_full_second[roi][conditions[2]]**2,  # CS SEM^2
        switchType_sems_windowed_full_second[roi][conditions[3]]**2   # IS SEM^2
    ]))
]

fig, ax = plt.subplots(figsize=(8, 6))  # Adjust figure size as needed
bar_width = 0.015  # Even narrower bars

# Closer indices, but ensure they're distinct enough to not overlap
index_switch_repeat = np.array([0.4375, 0.4625])

bars = ax.bar(index_switch_repeat, combined_means_switch_repeat, 
              yerr=combined_sems_switch_repeat, capsize=5, 
              color=['lightblue', 'blue'], width=bar_width)

ax.set_xticks(index_switch_repeat)
ax.set_xticklabels(['Repeat', 'Switch'], fontsize=24)
ax.tick_params(axis='y', labelsize=24)

# Uncomment the next line if you want to see the effect without plt.tight_layout()
# plt.tight_layout()

plt.show()


# %% [markdown]
# ### plot individual electrodes for interaction effects
# i think this will just work regardless of the output names 3/5

# %% [markdown]
# test this new plot significance function that offsets for each significance bar 4/7

# %%
def plot_significance(ax, times, sig_effects, y_offset=0.1):
    """
    Plot significance bars for the effects on top of the existing axes, adjusted for time windows.

    Parameters:
    - ax: The matplotlib Axes object to plot on.
    - times: Array of time points for the x-axis.
    - sig_effects: Dictionary with time windows as keys and lists of tuples (effect, p-value) as values.
    - y_offset: The vertical offset between different time window significance bars.
    """
    y_pos_base = ax.get_ylim()[1]  # Get the top y-axis limit to place significance bars

    time_windows = {
        'firstHalfSecond': (0, 0.5),
        'secondHalfSecond': (0.5, 1),
        'fullSecond': (0, 1)
    }

    window_offsets = {window: 0 for window in time_windows}  # Initialize offsets for each time window

    # Sort time windows to ensure 'FullSecond' bars are plotted last (on top)
    for time_window, effects in sorted(sig_effects.items(), key=lambda x: x[0] == 'fullSecond'):
        base_y_pos = y_pos_base + y_offset * list(time_windows).index(time_window)
        for effect, p_value in effects:
            start_time, end_time = time_windows[time_window]
            # Adjust y_pos based on how many bars have already been plotted in this window
            y_pos = base_y_pos + y_offset * window_offsets[time_window]

            # Update the color selection logic as per your requirement
            color = 'black'  # Default color for unmatched conditions
                        
            if 'congruency' in effect and 'congruencyProportion' not in effect and 'switchType' not in effect and 'congruency:congruencyProportion' not in effect and 'congruency:switchType' not in effect:
                color = 'red'
            elif 'congruencyProportion' in effect and 'switchType' not in effect and 'congruency:congruencyProportion' not in effect and 'congruency:switchType' not in effect:
                color = 'pink'
            elif 'switchType' in effect and 'switchProportion' not in effect and 'congruency' not in effect and 'switchType:switchProportion' not in effect and 'congruency:switchType' not in effect:
                color = 'blue'
            elif 'switchProportion' in effect and 'congruency' not in effect and 'switchType:switchProportion' not in effect and 'congruency:switchType' not in effect:
                color = 'skyblue'
            elif 'congruency:congruencyProportion' in effect:
                color = 'hotpink'
            elif 'switchType:switchProportion' in effect:
                color = 'gray'
            elif 'congruency:switchType' in effect:
                color = 'black'

            num_asterisks = '*' * (1 if p_value < 0.05 else 2 if p_value < 0.01 else 3)
            ax.plot([start_time, end_time], [y_pos, y_pos], color=color, lw=4)
            ax.text((start_time + end_time) / 2, y_pos, num_asterisks, ha='center', va='bottom', color=color)

            window_offsets[time_window] += 1  # Increment the offset for this time window

# %%
LAB_root = None
channels = None

if LAB_root is None:
    HOME = os.path.expanduser("~")
    if os.name == 'nt':  # windows
        LAB_root = os.path.join(HOME, "Box", "CoganLab")
    else:  # mac
        LAB_root = os.path.join(HOME, "Library", "CloudStorage", "Box-Box",
                                "CoganLab")

layout = get_data(task, root=LAB_root)
save_dir = os.path.join(layout.root, 'derivatives', 'freqFilt', 'figs')

    
# Dynamically select the first subject and use it to extract times
first_subject_id = next(iter(subjects_mne_objects))
example_condition_name = next(iter(subjects_mne_objects[first_subject_id]))
times = subjects_mne_objects[first_subject_id][example_condition_name]['HG_ev1_rescaled'].times

# Define the time windows
time_windows = ['firstHalfSecond', 'secondHalfSecond', 'fullSecond']

# port over the plot_electrodes_grid_whole_brain_analysis here, but replace wherever the save name is wholebrainanalysis with the roi names. 3/25.
def plot_electrodes_grid_roi(electrodes_data, significant_effects_structure, grid_num, roi, condition_names, times, save_dir, save_name, plotting_parameters):

    fig, axes = plt.subplots(4, 4, figsize=(20, 12))  # Adjust figure size as needed
    axes = axes.flatten()  # Flatten the axes array for easy indexing

    for i, (data, sub, electrode) in enumerate(electrodes_data):
        ax = axes[i]
        for condition_name in condition_names:
            color = plotting_parameters[condition_name]['color']
            line_style = plotting_parameters[condition_name]['line_style']
            ax.plot(times, data[condition_name], label=f'{roi}_{condition_name}', color=color, linestyle=line_style)
            ax.fill_between(times, 
                            data[condition_name] - np.std(data[condition_name], ddof=1) / np.sqrt(len(data[condition_name])),
                            data[condition_name] + np.std(data[condition_name], ddof=1) / np.sqrt(len(data[condition_name])), alpha=0.3)

        # Overlay a dotted vertical line at time = 0.5
        # ax.axvline(x=0.5, color='k', linestyle='--', linewidth=1)

        # Remove top and right borders
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        ax.set_title(f'Subject {sub}, Electrode {electrode}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Z-score')

        # Retrieve significant effects for the current subject and electrode
        sig_effects = significant_effects_structure.get(sub, {}).get(electrode, {}).get(roi, {})
        if sig_effects:
            # Adjust y_offset based on plotting needs. This used to not be assigned to a variable. 3/20.
            plot_significance(ax, times, sig_effects, y_offset=0.1)

    # Create the legend at the top center of the figure
    handles, labels = ax.get_legend_handles_labels()  # Get handles and labels from the last subplot
    fig.legend(handles, labels, loc='lower center', ncol=2)

    plt.tight_layout()  # Adjust the layout to make room for the legend
    plt.savefig(os.path.join(save_dir, f'{roi}_{save_name}_electrodes_plot_grid_{grid_num+1}.png'))
    plt.close()


# %% [markdown]
# test looping over subjects and electrodes as a function 4/1

# %% [markdown]
# THIS NEEDS TO GRAB SPECIFIC ROIS FROM THE SIGNIFICANT EFFECTS STRUCTURE. ALSO NEED TO GRAB THE CORRECT SAVE NAME. 6/11

# %%
def plot_electrodes_grid_roi_loop(subjects, sig_electrodes_per_subject_roi, roi, rois_suffix, concatenated_trialAvg_data, condition_names, grid_size, save_dir, save_name, times, plotting_parameters):
    electrodes_data = []
    electrode_counter = 0
    grid_num = 0

    # Load in significant effects structure
    significant_effects_structure_file_path = os.path.join(save_dir, f'{save_name}_significantEffectsStructure_{rois_suffix}.txt')
    with open(significant_effects_structure_file_path, 'r') as file:
        significant_effects_structure = json.load(file)

    for sub in subjects:
        if sub in sig_electrodes_per_subject_roi[roi]:
            for electrode in sig_electrodes_per_subject_roi[roi][sub]:
                electrode_data = {}
                for condition_name in condition_names:
                    # Ensure the index is correctly used here for your data structure
                    electrode_data[condition_name] = concatenated_trialAvg_data[condition_name][roi][electrode_counter]

                electrodes_data.append((electrode_data, sub, electrode))
                electrode_counter += 1
                if len(electrodes_data) == grid_size:
                    plot_electrodes_grid_roi(electrodes_data, significant_effects_structure, grid_num, roi, condition_names, times, save_dir, save_name, plotting_parameters)
                    electrodes_data = []  # Reset for the next grid
                    grid_num += 1

    # Plot remaining electrodes in the last grid
    if electrodes_data:
        plot_electrodes_grid_roi(electrodes_data, significant_effects_structure, grid_num, roi, condition_names, times, save_dir, save_name, plotting_parameters)


# %% [markdown]
# loop through rois and plot individual electrodes with color-coded significance bars

# %%
for roi in rois:
    plot_electrodes_grid_roi_loop(subjects, sig_electrodes_per_subject_roi, roi, rois_suffix, concatenated_trialAvg_data, condition_names, 16, save_dir, conditions_save_name, times, plotting_parameters)

# %% [markdown]
# awful godforsaken code to get individual electrode plots for congruency main effects, switch main effects, and interaction effects for lpfc 4/8  
# replace with function later after CNS...    

# %%
LAB_root = None
channels = None

if LAB_root is None:
    HOME = os.path.expanduser("~")
    if os.name == 'nt':  # windows
        LAB_root = os.path.join(HOME, "Box", "CoganLab")
    else:  # mac
        LAB_root = os.path.join(HOME, "Library", "CloudStorage", "Box-Box",
                                "CoganLab")

layout = get_data(task, root=LAB_root)
save_dir = os.path.join(layout.root, 'derivatives', 'freqFilt', 'figs')

# Dynamically select the first subject and use it to extract times
first_subject_id = next(iter(subjects_mne_objects))
example_output_name = next(iter(subjects_mne_objects[first_subject_id]))
times = subjects_mne_objects[first_subject_id][example_output_name]['HG_ev1_evoke_rescaled'].times

# %%
import numpy as np
import matplotlib.pyplot as plt
import os
import json

def plot_electrodes_grid_roi_switchTypeWithInteractionOutputNames(electrodes_data, significant_effects_structure, grid_num, roi, output_names, times, save_dir, save_name, plotting_parameters):
    fig, axes = plt.subplots(4, 4, figsize=(20, 12))  # Adjust figure size as needed
    axes = axes.flatten()  # Flatten the axes array for easy indexing

    for i, (data, sub, electrode) in enumerate(electrodes_data):
        ax = axes[i]
        # Calculate S and R for the electrode
        avg_ir_cr = (data['Stimulus_ir_fixationCrossBase_1sec_mirror'] + data['Stimulus_cr_fixationCrossBase_1sec_mirror']) / 2
        avg_is_cs = (data['Stimulus_is_fixationCrossBase_1sec_mirror'] + data['Stimulus_cs_fixationCrossBase_1sec_mirror']) / 2
        # avg_sem_is_cs = np.sqrt((np.power(switchTypeSigElectrodesSEM[roi]['Stimulus_ir_fixationCrossBase_1sec_mirror'], 2) + np.power(switchTypeSigElectrodesSEM[roi]['Stimulus_cr_fixationCrossBase_1sec_mirror'], 2)) / 2)
        # avg_sem_cr_cs = np.sqrt((np.power(switchTypeSigElectrodesSEM[roi]['Stimulus_is_fixationCrossBase_1sec_mirror'], 2) + np.power(switchTypeSigElectrodesSEM[roi]['Stimulus_cs_fixationCrossBase_1sec_mirror'], 2)) / 2)

        # Plotting S and R
        ax.plot(times, avg_ir_cr, label='repeat', color='blue', linestyle='-')
        ax.plot(times, avg_is_cs, label='switch', color='blue', linestyle='--')
        
        # Overlay a dotted vertical line at time = 0.5
        ax.axvline(x=0.5, color='k', linestyle='--', linewidth=1)

        # Remove top and right borders
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        ax.set_title(f'Subject {sub}, Electrode {electrode}')
        ax.set_xlabel('Time from Stimulus Onset (s)')
        ax.set_ylabel('Z-score From Baseline')

        # Retrieve and plot significant effects
        sig_effects = significant_effects_structure.get(sub, {}).get(electrode, {})
        if sig_effects:
            utils.plot_significance(ax, times, sig_effects, y_offset=0.1)

    # Create the legend at the top center of the figure
    handles, labels = ax.get_legend_handles_labels()  # Get handles and labels from the last subplot
    fig.legend(handles, labels, loc='lower center', ncol=2)

    plt.tight_layout()  # Adjust the layout to make room for the legend
    plt.savefig(os.path.join(save_dir, f'{roi}_{save_name}_electrodes_plot_grid_{grid_num+1}.png'))
    plt.close()


def plot_electrodes_grid_roi_congruencyWithInteractionOutputNames(electrodes_data, significant_effects_structure, grid_num, roi, output_names, times, save_dir, save_name, plotting_parameters):
    fig, axes = plt.subplots(4, 4, figsize=(20, 12))  # Adjust figure size as needed
    axes = axes.flatten()  # Flatten the axes array for easy indexing

    for i, (data, sub, electrode) in enumerate(electrodes_data):
        ax = axes[i]
        # Calculate I-C for the electrode
        avg_ir_is = (data['Stimulus_ir_fixationCrossBase_1sec_mirror'] + data['Stimulus_is_fixationCrossBase_1sec_mirror']) / 2
        avg_cr_cs = (data['Stimulus_cr_fixationCrossBase_1sec_mirror'] + data['Stimulus_cs_fixationCrossBase_1sec_mirror']) / 2
        # avg_sem_is_cs = np.sqrt((np.power(switchTypeSigElectrodesSEM[roi]['Stimulus_ir_fixationCrossBase_1sec_mirror'], 2) + np.power(switchTypeSigElectrodesSEM[roi]['Stimulus_cr_fixationCrossBase_1sec_mirror'], 2)) / 2)
        # avg_sem_cr_cs = np.sqrt((np.power(switchTypeSigElectrodesSEM[roi]['Stimulus_is_fixationCrossBase_1sec_mirror'], 2) + np.power(switchTypeSigElectrodesSEM[roi]['Stimulus_cs_fixationCrossBase_1sec_mirror'], 2)) / 2)

        # Plotting I-C difference
        ax.plot(times, avg_cr_cs, label='congruent', color='red', linestyle='-')
        ax.plot(times, avg_ir_is, label='incongruent', color='red', linestyle='--')
        
        # Overlay a dotted vertical line at time = 0.5
        ax.axvline(x=0.5, color='k', linestyle='--', linewidth=1)

        # Remove top and right borders
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.set_title(f'Subject {sub}, Electrode {electrode}')
        ax.set_xlabel('Time from Stimulus Onset (s)')
        ax.set_ylabel('Z-score From Baseline')

        # Retrieve and plot significant effects
        sig_effects = significant_effects_structure.get(sub, {}).get(electrode, {})
        if sig_effects:
            utils.plot_significance(ax, times, sig_effects, y_offset=0.1)

    # Create the legend at the top center of the figure
    handles, labels = ax.get_legend_handles_labels()  # Get handles and labels from the last subplot
    fig.legend(handles, labels, loc='lower center', ncol=2)

    plt.tight_layout()  # Adjust the layout to make room for the legend
    plt.savefig(os.path.join(save_dir, f'{roi}_{save_name}_electrodes_plot_grid_{grid_num+1}.png'))
    plt.close()



def plot_electrodes_grid_roi_congruencyEffectSwitchTypeWithInteractionOutputNames(electrodes_data, significant_effects_structure, grid_num, roi, output_names, times, save_dir, save_name, plotting_parameters):
    fig, axes = plt.subplots(4, 4, figsize=(20, 12))  # Adjust figure size as needed
    axes = axes.flatten()  # Flatten the axes array for easy indexing

    for i, (data, sub, electrode) in enumerate(electrodes_data):
        ax = axes[i]
        # Calculate congruency effect as a function of switch type for the electrode

        avg_diff_ir_cr = data['Stimulus_ir_fixationCrossBase_1sec_mirror'] - data['Stimulus_cr_fixationCrossBase_1sec_mirror']
        avg_diff_is_cs = data['Stimulus_is_fixationCrossBase_1sec_mirror'] - data['Stimulus_cs_fixationCrossBase_1sec_mirror']


        ax.plot(times, avg_diff_ir_cr, label='IR - CR', color='black', linestyle='-')
        ax.plot(times, avg_diff_is_cs, label='IS - CS', color='black', linestyle='--')
        
        # Overlay a dotted vertical line at time = 0.5
        ax.axvline(x=0.5, color='k', linestyle='--', linewidth=1)

        # Remove top and right borders
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        ax.set_title(f'Subject {sub}, Electrode {electrode}')
        ax.set_xlabel('Time from Stimulus Onset (s)')
        ax.set_ylabel('Z-score From Baseline')

        # Retrieve and plot significant effects
        sig_effects = significant_effects_structure.get(sub, {}).get(electrode, {})
        if sig_effects:
            plot_significance_justInteraction_delete_after_poster(ax, times, sig_effects, y_offset=0.1) #change back to plot_significance after poster 4/8

    # Create the legend at the top center of the figure
    handles, labels = ax.get_legend_handles_labels()  # Get handles and labels from the last subplot
    fig.legend(handles, labels, loc='lower center', ncol=2)

    plt.tight_layout()  # Adjust the layout to make room for the legend
    plt.savefig(os.path.join(save_dir, f'{roi}_{save_name}_electrodes_plot_grid_{grid_num+1}.png'))
    plt.close()



def plot_electrodes_grid_roi_switchCostCongruencyWithInteractionOutputNames(electrodes_data, significant_effects_structure, grid_num, roi, output_names, times, save_dir, save_name, plotting_parameters):
    fig, axes = plt.subplots(4, 4, figsize=(20, 12))  # Adjust figure size as needed
    axes = axes.flatten()  # Flatten the axes array for easy indexing

    for i, (data, sub, electrode) in enumerate(electrodes_data):
        ax = axes[i]
        # Calculate congruency effect as a function of switch type for the electrode

        avg_diff_is_ir = data['Stimulus_is_fixationCrossBase_1sec_mirror'] - data['Stimulus_ir_fixationCrossBase_1sec_mirror']
        avg_diff_cs_cr = data['Stimulus_cs_fixationCrossBase_1sec_mirror'] - data['Stimulus_cr_fixationCrossBase_1sec_mirror']


        ax.plot(times, avg_diff_is_ir, label='IS - IR', color='black', linestyle='-')
        ax.plot(times, avg_diff_cs_cr, label='CS - CR', color='black', linestyle='--')
        
        # Overlay a dotted vertical line at time = 0.5
        ax.axvline(x=0.5, color='k', linestyle='--', linewidth=1)

        # Remove top and right borders
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        ax.set_title(f'Subject {sub}, Electrode {electrode}')
        ax.set_xlabel('Time from Stimulus Onset (s)')
        ax.set_ylabel('Z-score From Baseline')

        # Retrieve and plot significant effects
        sig_effects = significant_effects_structure.get(sub, {}).get(electrode, {})
        if sig_effects:
            plot_significance_justInteraction_delete_after_poster(ax, times, sig_effects, y_offset=0.1) #change back to plot_significance after poster 4/8

    # Create the legend at the top center of the figure
    handles, labels = ax.get_legend_handles_labels()  # Get handles and labels from the last subplot
    fig.legend(handles, labels, loc='lower center', ncol=2)

    plt.tight_layout()  # Adjust the layout to make room for the legend
    plt.savefig(os.path.join(save_dir, f'{roi}_{save_name}_electrodes_plot_grid_{grid_num+1}.png'))
    plt.close()



# %% [markdown]
# try this for lpfc 4/8  
# just for poster, clean this up!! this is hella hard-coded 

# %%
electrodes_data = []
electrode_counter = 0
grid_size = 16  # Number of electrodes per grid
grid_num = 0
roi = 'lpfc'
save_name = 'congruencySigElectrodesCongruencyComparison'

# load in significant effects structure
significant_effects_structure_file_path = os.path.join(save_dir, f'congruency_switchType_ANOVAwithinElectrodes_significantEffectsStructure_{roi}.txt')
with open(significant_effects_structure_file_path, 'r') as file:
    significant_effects_structure = json.load(file)
for sub in subjects:

    # Use .get() to safely access congruencySigElectrodes for sub
    # If sub is not a key, congruencySigElectrodes_for_sub will be None
    congruencySigElectrodes_for_sub = congruencySigElectrodes.get(sub)

    # Check if congruencySigElectrodes_for_sub is None (i.e., if sub was not a key in congruencySigElectrodes)
    if congruencySigElectrodes_for_sub is None:
        continue  # Skip this sub and move to the next one

    # If we reach here, it means congruencySigElectrodes_for_sub is not None, and we can safely use it
    for electrode in congruencySigElectrodes_for_sub:
        electrode_data = {}
        for output_name in output_names:
            electrode_data[output_name] = concatenated_trialAvg_data_congruencySigElectrodes[roi][output_name][electrode_counter]

        electrodes_data.append((electrode_data, sub, electrode))
        electrode_counter += 1
        if len(electrodes_data) == grid_size:
            plot_electrodes_grid_roi_congruencyWithInteractionOutputNames(electrodes_data, significant_effects_structure, grid_num, roi, output_names, times, save_dir, save_name, plotting_parameters)
            electrodes_data = []  # Reset for the next grid
            grid_num += 1

# Plot remaining electrodes in the last grid
if electrodes_data:
    plot_electrodes_grid_roi_congruencyWithInteractionOutputNames(electrodes_data, significant_effects_structure, grid_num, roi, output_names, times, save_dir, save_name, plotting_parameters)

# %% [markdown]
# okay now do individual for switch type

# %%
electrodes_data = []
electrode_counter = 0
grid_size = 16  # Number of electrodes per grid
grid_num = 0
roi = 'lpfc'
save_name = 'switchTypeSigElectrodesSwitchTypeComparison'

# load in significant effects structure
significant_effects_structure_file_path = os.path.join(save_dir, f'congruency_switchType_ANOVAwithinElectrodes_significantEffectsStructure_{roi}.txt')
with open(significant_effects_structure_file_path, 'r') as file:
    significant_effects_structure = json.load(file)
for sub in subjects:

    # Use .get() to safely access switchType for sub
    # If sub is not a key, switchTypeSigElectrodes_for_sub will be None
    switchTypeSigElectrodes_for_sub = switchTypeSigElectrodes.get(sub)

    # Check if congruencySigElectrodes_for_sub is None (i.e., if sub was not a key in congruencySigElectrodes)
    if switchTypeSigElectrodes_for_sub is None:
        continue  # Skip this sub and move to the next one

    # If we reach here, it means congruencySigElectrodes_for_sub is not None, and we can safely use it
    for electrode in switchTypeSigElectrodes_for_sub:
        electrode_data = {}
        for output_name in output_names:
            electrode_data[output_name] = concatenated_trialAvg_data_switchTypeSigElectrodes[roi][output_name][electrode_counter]

        electrodes_data.append((electrode_data, sub, electrode))
        electrode_counter += 1
        if len(electrodes_data) == grid_size:
            plot_electrodes_grid_roi_switchTypeWithInteractionOutputNames(electrodes_data, significant_effects_structure, grid_num, roi, output_names, times, save_dir, save_name, plotting_parameters)
            electrodes_data = []  # Reset for the next grid
            grid_num += 1

# Plot remaining electrodes in the last grid
if electrodes_data:
    plot_electrodes_grid_roi_switchTypeWithInteractionOutputNames(electrodes_data, significant_effects_structure, grid_num, roi, output_names, times, save_dir, save_name, plotting_parameters)

# %% [markdown]
# now do individual for interaction (congruency effect by switch type)

# %%
electrodes_data = []
electrode_counter = 0
grid_size = 16  # Number of electrodes per grid
grid_num = 0
roi = 'lpfc'
save_name = 'congruencySwitchTypeInteractionSigElectrodesSwitchTypeComparison'

# load in significant effects structure
significant_effects_structure_file_path = os.path.join(save_dir, f'congruency_switchType_ANOVAwithinElectrodes_significantEffectsStructure_{roi}.txt')
with open(significant_effects_structure_file_path, 'r') as file:
    significant_effects_structure = json.load(file)
for sub in subjects:

    # Use .get() to safely access switchType for sub
    # If sub is not a key, switchTypeSigElectrodes_for_sub will be None
    congruencySwitchTypeInteractionSigElectrodes_for_sub = congruencySwitchTypeInteractionSigElectrodes.get(sub)

    # Check if congruencySigElectrodes_for_sub is None (i.e., if sub was not a key in congruencySigElectrodes)
    if congruencySwitchTypeInteractionSigElectrodes_for_sub is None:
        continue  # Skip this sub and move to the next one

    # If we reach here, it means congruencySigElectrodes_for_sub is not None, and we can safely use it
    for electrode in congruencySwitchTypeInteractionSigElectrodes_for_sub:
        electrode_data = {}
        for output_name in output_names:
            electrode_data[output_name] = concatenated_trialAvg_data_congruencySwitchTypeInteractionSigElectrodes[roi][output_name][electrode_counter]

        electrodes_data.append((electrode_data, sub, electrode))
        electrode_counter += 1
        if len(electrodes_data) == grid_size:
            plot_electrodes_grid_roi_congruencyEffectSwitchTypeWithInteractionOutputNames(electrodes_data, significant_effects_structure, grid_num, roi, output_names, times, save_dir, save_name, plotting_parameters)
            electrodes_data = []  # Reset for the next grid
            grid_num += 1

# Plot remaining electrodes in the last grid
if electrodes_data:
    plot_electrodes_grid_roi_congruencyEffectSwitchTypeWithInteractionOutputNames(electrodes_data, significant_effects_structure, grid_num, roi, output_names, times, save_dir, save_name, plotting_parameters)

# %% [markdown]
# now do individual for interaction (switch cost by congruency)

# %%
electrodes_data = []
electrode_counter = 0
grid_size = 16  # Number of electrodes per grid
grid_num = 0
roi = 'lpfc'
save_name = 'switchCostCongruencyInteractionSigElectrodes'

# load in significant effects structure
significant_effects_structure_file_path = os.path.join(save_dir, f'congruency_switchType_ANOVAwithinElectrodes_significantEffectsStructure_{roi}.txt')
with open(significant_effects_structure_file_path, 'r') as file:
    significant_effects_structure = json.load(file)
for sub in subjects:

    # Use .get() to safely access switchType for sub
    # If sub is not a key, switchTypeSigElectrodes_for_sub will be None
    congruencySwitchTypeInteractionSigElectrodes_for_sub = congruencySwitchTypeInteractionSigElectrodes.get(sub)

    # Check if congruencySigElectrodes_for_sub is None (i.e., if sub was not a key in congruencySigElectrodes)
    if congruencySwitchTypeInteractionSigElectrodes_for_sub is None:
        continue  # Skip this sub and move to the next one

    # If we reach here, it means congruencySigElectrodes_for_sub is not None, and we can safely use it
    for electrode in congruencySwitchTypeInteractionSigElectrodes_for_sub:
        electrode_data = {}
        for output_name in output_names:
            electrode_data[output_name] = concatenated_trialAvg_data_congruencySwitchTypeInteractionSigElectrodes[roi][output_name][electrode_counter]

        electrodes_data.append((electrode_data, sub, electrode))
        electrode_counter += 1
        if len(electrodes_data) == grid_size:
            plot_electrodes_grid_roi_switchCostCongruencyWithInteractionOutputNames(electrodes_data, significant_effects_structure, grid_num, roi, output_names, times, save_dir, save_name, plotting_parameters)
            electrodes_data = []  # Reset for the next grid
            grid_num += 1

# Plot remaining electrodes in the last grid
if electrodes_data:
    plot_electrodes_grid_roi_switchCostCongruencyWithInteractionOutputNames(electrodes_data, significant_effects_structure, grid_num, roi, output_names, times, save_dir, save_name, plotting_parameters)

# %% [markdown]
# okay once we choose the electrodes we want, plot the single example electrode

# %% [markdown]
# for congruency

# %%
def plot_single_electrode_data_congruency(data, times, electrode, subject_id, sig_effects, save_dir, save_name):
    fig, ax = plt.subplots(figsize=(10, 6))  # A more focused figure size
    avg_ir_is = (data['Stimulus_ir_fixationCrossBase_1sec_mirror'] + data['Stimulus_is_fixationCrossBase_1sec_mirror']) / 2
    avg_cr_cs = (data['Stimulus_cr_fixationCrossBase_1sec_mirror'] + data['Stimulus_cs_fixationCrossBase_1sec_mirror']) / 2
    
    ax.plot(times, avg_cr_cs, label='Congruent', color='red', linestyle='-')
    ax.plot(times, avg_ir_is, label='Incongruent', color='red', linestyle='--')

    # ax.axvline(x=0.5, color='k', linestyle='--', linewidth=1)

    # Remove top and right borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ax.set_xlabel('Time from Stimulus Onset (s)', fontsize=20)
    # ax.set_ylabel('Z-score Difference', fontsize=20)
    # ax.legend(fontsize=20, loc='upper left')

    # Make the x and y ticks bigger
    ax.tick_params(axis='x', labelsize=24)  # Adjust x-axis tick label size
    ax.tick_params(axis='y', labelsize=24)  # Adjust y-axis tick label size

    plt.tight_layout()

    # Incorporate significance plotting
    if sig_effects:
        utils.plot_significance(ax, times, sig_effects, y_offset=0.1)

    plt.tight_layout()
    plt.show()

    if save_dir and save_name:
        fig.savefig(os.path.join(save_dir, f'{save_name}_{subject_id}_{electrode}.png'))
    plt.close(fig)

# Example usage setup
sub, example_elec = 'D0063', 'RMMF13'
roi = 'lpfc'
electrode_index = 5 # right now just manually count this from the grid plot BUT make this real after CNS 4/9

save_name = 'congruencySigElectrodesCongruencyComparison'
congruencySigElectrodes_for_sub = congruencySigElectrodes.get(sub)

# Load in significant effects structure
significant_effects_structure_file_path = os.path.join(save_dir, f'congruency_switchType_ANOVAwithinElectrodes_significantEffectsStructure_{roi}.txt')
with open(significant_effects_structure_file_path, 'r') as file:
    significant_effects_structure = json.load(file)

# Extract significant effects for the specific electrode and subject
sig_effects = significant_effects_structure.get(sub, {}).get(example_elec, {})

electrode_data = {}
for output_name in output_names:
    electrode_data[output_name] = concatenated_trialAvg_data_congruencySigElectrodes[roi][output_name][electrode_index]

plot_single_electrode_data_congruency(electrode_data, times, example_elec, sub, sig_effects, save_dir, save_name)

# %% [markdown]
# for switchType

# %%
def plot_single_electrode_data_switchType(data, times, electrode, subject_id, sig_effects, save_dir, save_name):
    fig, ax = plt.subplots(figsize=(10, 6))  # A more focused figure size
    avg_ir_cr = (data['Stimulus_ir_fixationCrossBase_1sec_mirror'] + data['Stimulus_cr_fixationCrossBase_1sec_mirror']) / 2
    avg_is_cs = (data['Stimulus_is_fixationCrossBase_1sec_mirror'] + data['Stimulus_cs_fixationCrossBase_1sec_mirror']) / 2
    
    ax.plot(times, avg_ir_cr, label='Repeat', color='blue', linestyle='-')
    ax.plot(times, avg_is_cs, label='Switch', color='blue', linestyle='--')

    # ax.axvline(x=0.5, color='k', linestyle='--', linewidth=1)

    # Remove top and right borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ax.set_xlabel('Time from Stimulus Onset (s)', fontsize=20)
    # ax.set_ylabel('Z-score Difference', fontsize=20)
    # ax.legend(fontsize=20, loc='upper left')

    # Make the x and y ticks bigger
    ax.tick_params(axis='x', labelsize=24)  # Adjust x-axis tick label size
    ax.tick_params(axis='y', labelsize=24)  # Adjust y-axis tick label size

    plt.tight_layout()

    # Incorporate significance plotting
    if sig_effects:
        plot_significance(ax, times, sig_effects, y_offset=0.1)

    plt.tight_layout()
    plt.show()

    if save_dir and save_name:
        fig.savefig(os.path.join(save_dir, f'{save_name}_{subject_id}_{electrode}.png'))
    plt.close(fig)

# Example usage setup
sub, example_elec = 'D0059', 'LMMF9'
roi = 'lpfc'
electrode_index = 1 # right now just manually get this from the grid plot BUT make this real after CNS 4/9

save_name = 'switchTypeSigElectrodesCongruencyComparison'
switchTypeSigElectrodes_for_sub = switchTypeSigElectrodes.get(sub)

# Load in significant effects structure
significant_effects_structure_file_path = os.path.join(save_dir, f'congruency_switchType_ANOVAwithinElectrodes_significantEffectsStructure_{roi}.txt')
with open(significant_effects_structure_file_path, 'r') as file:
    significant_effects_structure = json.load(file)

# Extract significant effects for the specific electrode and subject
sig_effects = significant_effects_structure.get(sub, {}).get(example_elec, {})

electrode_data = {}
for output_name in output_names:
    electrode_data[output_name] = concatenated_trialAvg_data_switchTypeSigElectrodes[roi][output_name][electrode_index]

plot_single_electrode_data_switchType(electrode_data, times, example_elec, sub, sig_effects, save_dir, save_name)

# %% [markdown]
# for interaction effect, congruency effect by switch type

# %%
def plot_single_electrode_data_interaction_effect(data, times, electrode, subject_id, sig_effects, save_dir, save_name):
    fig, ax = plt.subplots(figsize=(10, 6))  # A more focused figure size

    avg_diff_ir_cr = data['Stimulus_ir_fixationCrossBase_1sec_mirror'] - data['Stimulus_cr_fixationCrossBase_1sec_mirror']
    avg_diff_is_cs = data['Stimulus_is_fixationCrossBase_1sec_mirror'] - data['Stimulus_cs_fixationCrossBase_1sec_mirror']

    ax.plot(times, avg_diff_ir_cr, label='IR - CR', color='black', linestyle='-')
    ax.plot(times, avg_diff_is_cs, label='IS - CS', color='black', linestyle='--')

    # ax.axvline(x=0.5, color='k', linestyle='--', linewidth=1)
    # Remove top and right borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # ax.set_xlabel('Time from Stimulus Onset (s)', fontsize=20)
    # ax.set_ylabel('Z-score Difference', fontsize=20)
    # ax.legend(fontsize=20, loc='upper left')

    # Make the x and y ticks bigger
    ax.tick_params(axis='x', labelsize=24)  # Adjust x-axis tick label size
    ax.tick_params(axis='y', labelsize=24)  # Adjust y-axis tick label size

    plt.tight_layout()

    # Incorporate significance plotting
    if sig_effects:
        plot_significance_justInteraction_delete_after_poster(ax, times, sig_effects, y_offset=0.1)

    plt.tight_layout()
    plt.show()

    if save_dir and save_name:
        fig.savefig(os.path.join(save_dir, f'{save_name}_{subject_id}_{electrode}.png'))
    plt.close(fig)

# Example usage setup
sub, example_elec = 'D0094', 'LFAI9'
electrode_index = 8 # right now just manually get this from the grid plot BUT make this real after CNS 4/9
roi = 'lpfc'
save_name = 'congruencySwitchTypeInteractionSigElectrodesSwitchTypeComparison'
congruencySwitchTypeInteractionSigElectrodes_for_sub = congruencySwitchTypeInteractionSigElectrodes.get(sub)

# Load in significant effects structure
significant_effects_structure_file_path = os.path.join(save_dir, f'congruency_switchType_ANOVAwithinElectrodes_significantEffectsStructure_{roi}.txt')
with open(significant_effects_structure_file_path, 'r') as file:
    significant_effects_structure = json.load(file)

# Extract significant effects for the specific electrode and subject
sig_effects = significant_effects_structure.get(sub, {}).get(example_elec, {})

electrode_data = {}
for output_name in output_names:
    electrode_data[output_name] = concatenated_trialAvg_data_congruencySwitchTypeInteractionSigElectrodes[roi][output_name][electrode_index]
plot_single_electrode_data_interaction_effect(electrode_data, times, example_elec, sub, sig_effects, save_dir, save_name)

# %%
congruencySwitchTypeInteractionSigElectrodes

# %% [markdown]
# interaction effect for switch cost by congruency

# %%
def plot_single_electrode_data_interaction_effect(data, times, electrode, subject_id, sig_effects, save_dir, save_name):
    fig, ax = plt.subplots(figsize=(10, 6))  # A more focused figure size

    avg_diff_is_ir = data['Stimulus_is_fixationCrossBase_1sec_mirror'] - data['Stimulus_ir_fixationCrossBase_1sec_mirror']
    avg_diff_cs_cr = data['Stimulus_cs_fixationCrossBase_1sec_mirror'] - data['Stimulus_cr_fixationCrossBase_1sec_mirror']

    ax.plot(times, avg_diff_is_ir, label='IS - IR', color='black', linestyle='-')
    ax.plot(times, avg_diff_cs_cr, label='CS - CR', color='black', linestyle='--')

    # ax.axvline(x=0.5, color='k', linestyle='--', linewidth=1)
    # Remove top and right borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.set_xlabel('Time from Stimulus Onset (s)', fontsize=20)
    ax.set_ylabel('Z-score Difference', fontsize=20)
    ax.legend(fontsize=20, loc='upper left')

    # Make the x and y ticks bigger
    ax.tick_params(axis='x', labelsize=20)  # Adjust x-axis tick label size
    ax.tick_params(axis='y', labelsize=20)  # Adjust y-axis tick label size

    # Incorporate significance plotting
    if sig_effects:
        plot_significance_justInteraction_delete_after_poster(ax, times, sig_effects, y_offset=0.1)

    plt.tight_layout()
    plt.show()

    if save_dir and save_name:
        fig.savefig(os.path.join(save_dir, f'{save_name}_{subject_id}_{electrode}.png'))
    plt.close(fig)

# Example usage setup
sub, example_elec = 'D0094', 'LFAI9'
electrode_index = 8 # right now just manually get this from the grid plot BUT make this real after CNS 4/9
roi = 'lpfc'
save_name = 'congruencySwitchTypeInteractionSigElectrodesSwitchTypeComparison'
congruencySwitchTypeInteractionSigElectrodes_for_sub = congruencySwitchTypeInteractionSigElectrodes.get(sub)

# Load in significant effects structure
significant_effects_structure_file_path = os.path.join(save_dir, f'congruency_switchType_ANOVAwithinElectrodes_significantEffectsStructure_{roi}.txt')
with open(significant_effects_structure_file_path, 'r') as file:
    significant_effects_structure = json.load(file)

# Extract significant effects for the specific electrode and subject
sig_effects = significant_effects_structure.get(sub, {}).get(example_elec, {})

electrode_data = {}
for output_name in output_names:
    electrode_data[output_name] = concatenated_trialAvg_data_congruencySwitchTypeInteractionSigElectrodes[roi][output_name][electrode_index]
plot_single_electrode_data_interaction_effect(electrode_data, times, example_elec, sub, sig_effects, save_dir, save_name)

# %% [markdown]
# plot example electrode with all four conditions  
# note this index is in the overall sig electrodes

# %%
def plot_single_electrode_data_interaction_effect_all_four_conditions(data, times, electrode, subject_id, sig_effects, save_dir, save_name):
    fig, ax = plt.subplots(figsize=(10, 6))  # A more focused figure size

    avg_is = data['Stimulus_is_fixationCrossBase_1sec_mirror']
    avg_ir = data['Stimulus_ir_fixationCrossBase_1sec_mirror']
    avg_cs = data['Stimulus_cs_fixationCrossBase_1sec_mirror']
    avg_cr = data['Stimulus_cr_fixationCrossBase_1sec_mirror']

    ax.plot(times, avg_is, label='IS', color='red', linestyle='--')
    ax.plot(times, avg_ir, label='IR', color='red', linestyle='-')
    ax.plot(times, avg_cs, label='CS', color='pink', linestyle='--')
    ax.plot(times, avg_cr, label='CR', color='pink', linestyle='-')

    # Remove top and right borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # ax.set_xlabel('Time from Stimulus Onset (s)', fontsize=20)
    # ax.set_ylabel('Z-score Difference', fontsize=20)
    # ax.legend(fontsize=20, loc='upper left')

    # Make the x and y ticks bigger
    ax.tick_params(axis='x', labelsize=24)  # Adjust x-axis tick label size
    ax.tick_params(axis='y', labelsize=24)  # Adjust y-axis tick label size

    # Incorporate significance plotting
    if sig_effects:
        plot_significance_justInteraction_delete_after_poster(ax, times, sig_effects, y_offset=0.1)

    plt.tight_layout()
    plt.show()

    if save_dir and save_name:
        fig.savefig(os.path.join(save_dir, f'{save_name}_{subject_id}_{electrode}.png'))
    plt.close(fig)

# Example usage setup
sub, example_elec = 'D0094', 'LPAI9'
electrode_index = 31 # right now just manually get this from the grid plot BUT make this real after CNS 4/9
roi = 'lpfc'
save_name = 'sigElectrodesPerSubjectROI_D0094LPAI9'
sig_electrodes_for_sub = sig_electrodes_per_subject_roi.get(sub)

# Load in significant effects structure
significant_effects_structure_file_path = os.path.join(save_dir, f'congruency_switchType_ANOVAwithinElectrodes_significantEffectsStructure_{roi}.txt')
with open(significant_effects_structure_file_path, 'r') as file:
    significant_effects_structure = json.load(file)

# Extract significant effects for the specific electrode and subject
sig_effects = significant_effects_structure.get(sub, {}).get(example_elec, {})

electrode_data = {}
for output_name in output_names:
    electrode_data[output_name] = concatenated_trialAvg_data[roi][output_name][electrode_index]
plot_single_electrode_data_interaction_effect_all_four_conditions(electrode_data, times, example_elec, sub, sig_effects, save_dir, save_name)

# %%
# Example usage setup
sub, example_elec = 'D0065', 'RASF14'
electrode_index = 14 # right now just manually get this from the grid plot BUT make this real after CNS 4/9
roi = 'lpfc'
save_name = 'sigElectrodesPerSubjectROI_D0065_RASF14'
sig_electrodes_for_sub = sig_electrodes_per_subject_roi.get(sub)

# Load in significant effects structure
significant_effects_structure_file_path = os.path.join(save_dir, f'congruency_switchType_ANOVAwithinElectrodes_significantEffectsStructure_{roi}.txt')
with open(significant_effects_structure_file_path, 'r') as file:
    significant_effects_structure = json.load(file)

# Extract significant effects for the specific electrode and subject
sig_effects = significant_effects_structure.get(sub, {}).get(example_elec, {})

electrode_data = {}
for output_name in output_names:
    electrode_data[output_name] = concatenated_trialAvg_data[roi][output_name][electrode_index]
plot_single_electrode_data_interaction_effect_all_four_conditions(electrode_data, times, example_elec, sub, sig_effects, save_dir, save_name)

# %% [markdown]
# old way of looping without a function 4/1

# %% [markdown]
# dlpfc

# %%
electrodes_data = []
electrode_counter = 0
grid_size = 16  # Number of electrodes per grid
grid_num = 0
roi = 'dlpfc'
if 'Stimulus_c25_fixationCrossBase_1sec_mirror' in output_names:
    save_name = 'congruency_congruencyProportion' # i think this will be congruency x con prop if i load in c25?
elif 'Stimulus_s25_fixationCrossBase_1sec_mirror' in output_names:
    save_name = 'switchType_switchProportion' # i think if there's no c25, but there is s25, then i am doing switch x switch prop? 3/17
elif 'Stimulus_cr_fixationCrossBase_1sec_mirror' in output_names:
    save_name = 'congruency_switchType'

# load in significant effects structure
significant_effects_structure_file_path = os.path.join(save_dir, f'{save_name}_ANOVAwithinElectrodes_significantEffectsStructure_{roi}.txt')
with open(significant_effects_structure_file_path, 'r') as file:
    significant_effects_structure = json.load(file)

# DUDE MAKE THE SIG ELECTRODES PER SUBJECT INTO A DICTIONARY. Bad code is bad.
for sub in subjects:
    if sub in sig_electrodes_per_subject_roi[roi]:
        for electrode in sig_electrodes_per_subject_roi[roi][sub]:
            
            electrode_data = {}
            for output_name in output_names:
                electrode_data[output_name] = concatenated_trialAvg_data[roi][output_name][electrode_counter]

            electrodes_data.append((electrode_data, sub, electrode))
            electrode_counter += 1
            if len(electrodes_data) == grid_size:
                plot_electrodes_grid_roi(electrodes_data, significant_effects_structure, grid_num, roi, output_names, times, save_dir, save_name, plotting_parameters)
                electrodes_data = []  # Reset for the next grid
                grid_num += 1

# Plot remaining electrodes in the last grid
if electrodes_data:
    plot_electrodes_grid_roi(electrodes_data, significant_effects_structure, grid_num, roi, output_names, times, save_dir, save_name, plotting_parameters)

# %% [markdown]
# acc

# %%
electrodes_data = []
electrode_counter = 0
grid_size = 16  # Number of electrodes per grid
grid_num = 0
roi = 'acc'
if 'Stimulus_c25_fixationCrossBase_1sec_mirror' in output_names:
    save_name = 'congruency_congruencyProportion' # i think this will be congruency x con prop if i load in c25?
elif 'Stimulus_s25_fixationCrossBase_1sec_mirror' in output_names:
    save_name = 'switchType_switchProportion' # i think if there's no c25, but there is s25, then i am doing switch x switch prop? 3/17
elif 'Stimulus_cr_fixationCrossBase_1sec_mirror' in output_names:
    save_name = 'congruency_switchType'

# load in significant effects structure
significant_effects_structure_file_path = os.path.join(save_dir, f'{save_name}_ANOVAwithinElectrodes_significantEffectsStructure_{roi}.txt')
with open(significant_effects_structure_file_path, 'r') as file:
    significant_effects_structure = json.load(file)

# DUDE MAKE THE SIG ELECTRODES PER SUBJECT INTO A DICTIONARY. Bad code is bad.
for sub in subjects:
    if sub in sig_electrodes_per_subject_roi[roi]:
        for electrode in sig_electrodes_per_subject_roi[roi][sub]:
            
            electrode_data = {}
            for output_name in output_names:
                electrode_data[output_name] = concatenated_trialAvg_data[roi][output_name][electrode_counter]

            electrodes_data.append((electrode_data, sub, electrode))
            electrode_counter += 1
            if len(electrodes_data) == grid_size:
                plot_electrodes_grid_roi(electrodes_data, significant_effects_structure, grid_num, roi, output_names, times, save_dir, save_name, plotting_parameters)
                electrodes_data = []  # Reset for the next grid
                grid_num += 1

# Plot remaining electrodes in the last grid
if electrodes_data:
    plot_electrodes_grid_roi(electrodes_data, significant_effects_structure, grid_num, roi, output_names, times, save_dir, save_name, plotting_parameters)

# %% [markdown]
# parietal

# %%
electrodes_data = []
electrode_counter = 0
grid_size = 16  # Number of electrodes per grid
grid_num = 0
roi = 'parietal'
if 'Stimulus_c25_fixationCrossBase_1sec_mirror' in output_names:
    save_name = 'congruency_congruencyProportion' # i think this will be congruency x con prop if i load in c25?
elif 'Stimulus_s25_fixationCrossBase_1sec_mirror' in output_names:
    save_name = 'switchType_switchProportion' # i think if there's no c25, but there is s25, then i am doing switch x switch prop? 3/17
elif 'Stimulus_cr_fixationCrossBase_1sec_mirror' in output_names:
    save_name = 'congruency_switchType'

# load in significant effects structure
significant_effects_structure_file_path = os.path.join(save_dir, f'{save_name}_ANOVAwithinElectrodes_significantEffectsStructure_{roi}.txt')
with open(significant_effects_structure_file_path, 'r') as file:
    significant_effects_structure = json.load(file)

# DUDE MAKE THE SIG ELECTRODES PER SUBJECT INTO A DICTIONARY. Bad code is bad.
for sub in subjects:
    if sub in sig_electrodes_per_subject_roi[roi]:
        for electrode in sig_electrodes_per_subject_roi[roi][sub]:
            
            electrode_data = {}
            for output_name in output_names:
                electrode_data[output_name] = concatenated_trialAvg_data[roi][output_name][electrode_counter]

            electrodes_data.append((electrode_data, sub, electrode))
            electrode_counter += 1
            if len(electrodes_data) == grid_size:
                plot_electrodes_grid_roi(electrodes_data, significant_effects_structure, grid_num, roi, output_names, times, save_dir, save_name, plotting_parameters)
                electrodes_data = []  # Reset for the next grid
                grid_num += 1

# Plot remaining electrodes in the last grid
if electrodes_data:
    plot_electrodes_grid_roi(electrodes_data, significant_effects_structure, grid_num, roi, output_names, times, save_dir, save_name, plotting_parameters)

# %% [markdown]
# lpfc

# %%
electrodes_data = []
electrode_counter = 0
grid_size = 16  # Number of electrodes per grid
grid_num = 0
roi = 'lpfc'
if 'Stimulus_c25_fixationCrossBase_1sec_mirror' in output_names:
    save_name = 'congruency_congruencyProportion' # i think this will be congruency x con prop if i load in c25?
elif 'Stimulus_s25_fixationCrossBase_1sec_mirror' in output_names:
    save_name = 'switchType_switchProportion' # i think if there's no c25, but there is s25, then i am doing switch x switch prop? 3/17
elif 'Stimulus_cr_fixationCrossBase_1sec_mirror' in output_names:
    save_name = 'congruency_switchType'

# load in significant effects structure
significant_effects_structure_file_path = os.path.join(save_dir, f'{save_name}_ANOVAwithinElectrodes_significantEffectsStructure_{roi}.txt')
with open(significant_effects_structure_file_path, 'r') as file:
    significant_effects_structure = json.load(file)

for sub in subjects:
    if sub in sig_electrodes_per_subject_roi[roi]:
        for electrode in sig_electrodes_per_subject_roi[roi][sub]:
            
            electrode_data = {}
            for output_name in output_names:
                electrode_data[output_name] = concatenated_trialAvg_data[roi][output_name][electrode_counter]

            electrodes_data.append((electrode_data, sub, electrode))
            electrode_counter += 1
            if len(electrodes_data) == grid_size:
                plot_electrodes_grid_roi(electrodes_data, significant_effects_structure, grid_num, roi, output_names, times, save_dir, save_name, plotting_parameters)
                electrodes_data = []  # Reset for the next grid
                grid_num += 1

# Plot remaining electrodes in the last grid
if electrodes_data:
    plot_electrodes_grid_roi(electrodes_data, significant_effects_structure, grid_num, roi, output_names, times, save_dir, save_name, plotting_parameters)


