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
# commenting these out for now, going to put them in black
"Stimulus_err": {
     "condition_parameter": "error",
     "color": "firebrick",
     "line_style": "-",
 },
 "Stimulus_corr": {
     "condition_parameter": "correct",
     "color": "seagreen",
     "line_style": "-", 
 },
 "Response_err": {
     "condition_parameter": "error",
     "color": "firebrick",
     "line_style": "-",
 },
 "Response_corr": {
     "condition_parameter": "correct",
     "color": "seagreen",
     "line_style": "-", 
 },
# "Stimulus_err": {
#    "condition_parameter": "error",
#    "color": "black",
#    "line_style": "--",
#},
# "Stimulus_corr": {
#   "condition_parameter": "correct",
#    "color": "black",
#    "line_style": "-", 
# },
"Stimulus_err_inc": {
    "condition_parameter": "err-inc",
    "color": "black",
    "line_style": "--",
},
"Stimulus_err_con": {
    "condition_parameter": "err-con",
    "color": "black",
    "line_style": "-", 
},
"Stimulus_err_iS": {
    "condition_parameter": "err-iS",
    "color": "black",
    "line_style": "-", 
},
"Stimulus_err_cR": {
    "condition_parameter": "err-cR",
    "color": "black",
    "line_style": "--", 
},
"Stimulus_err_iR": {
    "condition_parameter": "err-iR",
    "color": "darkred",
    "line_style": "-", 
},
"Stimulus_err_cS": {
    "condition_parameter": "err-cS",
    "color": "midnightblue",
    "line_style": "-", 
},
"Stimulus_inc_err": {
    "condition_parameter": "err-inc",
    "color": "black",
    "line_style": "-",
},
"Stimulus_switch_err": {
    "condition_parameter": "err-switch",
    "color": "black",
    "line_style": "-",
},
'Stimulus_i75-Stimulus_c75': {
    "condition_parameter": "i75-c75",
    "color": "orange",
    "line_style": "-"
},
'Stimulus_i25-Stimulus_c25': {
    "condition_parameter": "i25-c25",
    "color": "pink",
    "line_style": "-"
},
'Stimulus_s75-Stimulus_r75': {
    "condition_parameter": "s75-r75",
    "color": "purple",
    "line_style": "-"
},
'Stimulus_s25-Stimulus_r25': {
    "condition_parameter": "s25-r25",
    "color": "skyblue",
    "line_style": "-"
},

# === 16-condition entries (congruency × incongruentProportion × switchType × switchProportion) ===
# Color carries (congruency, incongruentProportion):
#   c25 -> pink,    c75 -> orange    (warm hues = congruency family)
#   i25 -> skyblue, i75 -> purple    (cool hues = incongruency family)
# Linestyle carries switchType:  r -> '-',  s -> '--'
# Linewidth carries switchProportion: 25 -> 1.5, 75 -> 2.5
"Stimulus_c25r25": {"condition_parameter": "c25r25", "color": "pink",    "line_style": "-",  "linewidth": 1.5},
"Stimulus_c25r75": {"condition_parameter": "c25r75", "color": "pink",    "line_style": "-",  "linewidth": 2.5},
"Stimulus_c25s25": {"condition_parameter": "c25s25", "color": "pink",    "line_style": "--", "linewidth": 1.5},
"Stimulus_c25s75": {"condition_parameter": "c25s75", "color": "pink",    "line_style": "--", "linewidth": 2.5},
"Stimulus_c75r25": {"condition_parameter": "c75r25", "color": "orange",  "line_style": "-",  "linewidth": 1.5},
"Stimulus_c75r75": {"condition_parameter": "c75r75", "color": "orange",  "line_style": "-",  "linewidth": 2.5},
"Stimulus_c75s25": {"condition_parameter": "c75s25", "color": "orange",  "line_style": "--", "linewidth": 1.5},
"Stimulus_c75s75": {"condition_parameter": "c75s75", "color": "orange",  "line_style": "--", "linewidth": 2.5},
"Stimulus_i25r25": {"condition_parameter": "i25r25", "color": "skyblue", "line_style": "-",  "linewidth": 1.5},
"Stimulus_i25r75": {"condition_parameter": "i25r75", "color": "skyblue", "line_style": "-",  "linewidth": 2.5},
"Stimulus_i25s25": {"condition_parameter": "i25s25", "color": "skyblue", "line_style": "--", "linewidth": 1.5},
"Stimulus_i25s75": {"condition_parameter": "i25s75", "color": "skyblue", "line_style": "--", "linewidth": 2.5},
"Stimulus_i75r25": {"condition_parameter": "i75r25", "color": "purple",  "line_style": "-",  "linewidth": 1.5},
"Stimulus_i75r75": {"condition_parameter": "i75r75", "color": "purple",  "line_style": "-",  "linewidth": 2.5},
"Stimulus_i75s25": {"condition_parameter": "i75s25", "color": "purple",  "line_style": "--", "linewidth": 1.5},
"Stimulus_i75s75": {"condition_parameter": "i75s75", "color": "purple",  "line_style": "--", "linewidth": 2.5},

}
