import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import sys
import os
import argparse

print(sys.path)
sys.path.append("C:/Users/jz421/Desktop/GlobalLocal/IEEG_Pipelines/") #need to do this cuz otherwise ieeg isn't added to path...

# Get the absolute path to the directory containing the current script
# For GlobalLocal/src/analysis/preproc/make_epoched_data.py, this is GlobalLocal/src/analysis/preproc
try:
    # This will work if running as a .py script
    current_file_path = os.path.abspath(__file__)
    current_script_dir = os.path.dirname(current_file_path)
except NameError:
    # This will be executed if __file__ is not defined (e.g., in a Jupyter Notebook)
    # os.getcwd() often gives the directory of the notebook,
    # or the directory from which the Jupyter server was started.
    current_script_dir = os.getcwd()

# Navigate up three levels to get to the 'GlobalLocal' directory
project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..', '..'))

# Add the 'GlobalLocal' directory to sys.path if it's not already there
if project_root not in sys.path:
    sys.path.insert(0, project_root) # insert at the beginning to prioritize it

from ieeg.navigate import channel_outlier_marker, trial_ieeg, crop_empty_data, \
    outliers_to_nan
from ieeg.io import raw_from_layout, get_data
from ieeg.timefreq.utils import crop_pad
from ieeg.timefreq import gamma
from ieeg.calc.scaling import rescale
import mne
import numpy as np
from ieeg.calc.reshape import make_data_same
from ieeg.calc.stats import time_perm_cluster, window_averaged_shuffle
from ieeg.viz.mri import gen_labels

# from utils import make_or_load_subjects_electrodes_to_ROIs_dict, load_acc_arrays, calculate_RTs, save_channels_to_file, save_sig_chans, \
#       load_sig_chans, channel_names_to_indices, filter_and_average_epochs, permutation_test, perform_permutation_test_across_electrodes, perform_permutation_test_within_electrodes, \
#       add_accuracy_to_epochs, load_mne_objects, create_subjects_mne_objects_dict, extract_significant_effects, convert_dataframe_to_serializable_format, \
#       perform_modular_anova, make_plotting_parameters, plot_significance

import matplotlib.pyplot as plt
from collections import OrderedDict, defaultdict
import json
# still need to test if the permutation test functions load in properly.
import pandas as pd
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

# from src.analysis.power.roi_analysis import blah_blah
from src.analysis.config import experiment_conditions, plotting_parameters
import src.analysis.utils.general_utils as utils # import utils functions one by one by name
from src.analysis.power.power_traces import make_multi_channel_evokeds_for_all_conditions_and_rois, plot_power_traces_for_all_rois

def main(subject_id):
    # All your logic for processing ONE subject goes here
    print(f"Processing subject: {subject_id}")
    config_dir = /cwork/etb28/GlobalLocal/src/analysis/config
    subjects_electrodestoROIs_dict = utils.make_or_load_subjects_electrodes_to_ROIs_dict(
        [subject_id], # Pass a list with just the one subject
        task='GlobalLocal', 
        LAB_root=None, 
        save_dir=config_dir, 
        filename=f'{subject_id}_electrodestoROIs_dict.json' 
    )
    print(f"Finished processing {subject_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject', required=True, help='Subject ID to process')
    args = parser.parse_args()
    main(args.subject)
