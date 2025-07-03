# TODO: Add /Accuracy1.0 to all of these if I want to exclude error trials. I think that's good, right..?
# This file contains the different conditions used in the experiment, split by what comparisons you want to make. stimulus_experiment_conditions and stimulus_conditions are most important i think.

#  congruency
stimulus_congruency_conditions = {
    "Stimulus_c": {
        "BIDS_events": ["Stimulus/c25.0/Accuracy1.0", "Stimulus/c75.0/Accuracy1.0"],
        "congruency": "c"
    },
    "Stimulus_i": {
        "BIDS_events": ["Stimulus/i25.0/Accuracy1.0", "Stimulus/i75.0/Accuracy1.0"],
        "congruency": "i"
    }
}

# switch type
stimulus_switch_type_conditions = {
    "Stimulus_r": {
        "BIDS_events": ["Stimulus/r25.0/Accuracy1.0", "Stimulus/r75.0/Accuracy1.0"],
        "switchType": "r"
    },
    "Stimulus_s": {        
        "BIDS_events": ["Stimulus/s25.0/Accuracy1.0", "Stimulus/s75.0/Accuracy1.0"],
        "switchType": "s"
    }
}

# ir vs is
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

# cr vs cs
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

# is vs cs
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

# ir vs cr
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

# main effect interaction effects (run this with the anova code. Ugh make everything more modular later.)

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

# block interaction contrasts for lwpc

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

# block interaction contrasts for lwps

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