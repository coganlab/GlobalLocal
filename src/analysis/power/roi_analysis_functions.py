import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import sys
print(sys.path)
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