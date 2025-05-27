# ongoing refactoring of roi_analysis.ipynb. This will be the main function that imports all of the stats functions and plotting functions.
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import sys
print(sys.path)
sys.path.append("C:/Users/jz421/Desktop/GlobalLocal/IEEG_Pipelines/") #need to do this cuz otherwise ieeg isn't added to path...

# Get the absolute path to the directory containing the current script
# For GlobalLocal/src/analysis/power/roi_analysis_functions.py, this is GlobalLocal/src/analysis/power
current_script_dir = os.path.dirname(os.path.abspath(__file__))

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
import os
import numpy as np
from ieeg.calc.reshape import make_data_same
from ieeg.calc.stats import time_perm_cluster, window_averaged_shuffle
from ieeg.viz.mri import gen_labels

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






# i copied this main function from make_epoched_data.py, need to modify this for roi_analysis.py. CURRENTLY BROKEN 5/26/25.
def main(subjects=None, task='GlobalLocal', times=(-1, 1.5),
         within_base_times=(-1, 0), base_times_length=0.5, pad_length=0.5, LAB_root=None, channels=None, dec_factor=8, outliers=10, passband=(70,150)):
    """
    Main function to bandpass filter and compute time permutation cluster stats and task-significant electrodes for chosen subjects.
    """
    if subjects is None:
        subjects = ['D0057', 'D0059', 'D0063', 'D0065', 'D0069', 'D0071', 'D0077', 'D0090', 'D0094', 'D0100', 'D0102', 'D0103']

    for sub in subjects:
        bandpass_and_epoch_and_find_task_significant_electrodes(sub=sub, task=task, times=times,
                          within_base_times=within_base_times, base_times_length=base_times_length,
                          pad_length=pad_length, LAB_root=LAB_root, channels=channels,
                          dec_factor=dec_factor, outliers=outliers, passband=passband)
        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process subjects and plot bandpass-filtered data, compute time permutation cluster matrix of electrodes by time, and find task-significant electrodes.")
    parser.add_argument('--subjects', nargs='+', default=None, help='List of subjects to process. If not provided, all subjects will be processed.')
    parser.add_argument('--task', type=str, default='GlobalLocal', help='Task to process. Default is GlobalLocal.')
    parser.add_argument('--times', type=float, nargs=2, default=(-1, 1.5), help='Time window for event processing. Default is (-1, 1.5).')
    parser.add_argument('--within_base_times', type=float, nargs=2, default=(-1, 0), help='Time window for baseline processing. Default is (-1, 0).')
    parser.add_argument('--base_times_length', type=float, default=0.5, help='Length of the time intervals to randomly select within `within_base_times`. Default is 0.5.')
    parser.add_argument('--pad_length', type=float, default=0.5, help='Length to pad each time interval. Will be removed later. Default is 0.5.')
    parser.add_argument('--LAB_root', type=str, default=None, help='Root directory for the lab. Will be determined based on OS if not provided. Default is None.')
    parser.add_argument('--channels', type=str, default=None, help='Channels to plot and get stats for. Default is all channels.')
    parser.add_argument('--dec_factor', type=int, default=8, help='Decimation factor. Default is 8.')
    parser.add_argument('--outliers', type=int, default=10, help='How many standard deviations above the mean for a trial to be considered an outlier. Default is 10.')
    parser.add_argument('--passband', type=float, nargs=2, default=(70,150), help='Frequency range for the frequency band of interest. Default is (70, 150).')
    args=parser.parse_args()

    print("--------- PARSED ARGUMENTS ---------")
    print(f"args.passband: {args.passband} (type: {type(args.passband)})")
    print(f"args.subjects: {args.subjects} (type: {type(args.subjects)})")

    main(subjects=args.subjects, 
        task=args.task, 
        times=args.times, 
        within_base_times=args.within_base_times, 
        base_times_length=args.base_times_length, 
        pad_length=args.pad_length, 
        LAB_root=args.LAB_root, 
        channels=args.channels, 
        dec_factor=args.dec_factor, 
        outliers=args.outliers, 
        passband=args.passband)