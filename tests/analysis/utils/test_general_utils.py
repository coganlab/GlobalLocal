import pytest
import sys
import os

# Add the project root to the path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import mne
import json
import numpy as np
import pandas as pd
from ieeg.navigate import channel_outlier_marker, trial_ieeg, crop_empty_data, \
    outliers_to_nan
from ieeg.io import raw_from_layout, get_data
from ieeg.timefreq.utils import crop_pad
from ieeg.timefreq import gamma
from ieeg.calc.scaling import rescale
from ieeg.calc.stats import time_perm_cluster, window_averaged_shuffle, find_outliers
from ieeg.viz.mri import gen_labels
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from collections import OrderedDict, defaultdict
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from numpy.lib.stride_tricks import as_strided, sliding_window_view

from src.analysis.config import experiment_conditions

from src.analysis.utils.general_utils import (
    get_conditions_save_name
)

def test_get_conditions_save_name():
    name = get_conditions_save_name(
        experiment_conditions.stimulus_lwpc_conditions, experiment_conditions, 19
    )
    assert name == "stimulus_lwpc_conditions_19_subjects"

def test_unknown_conditions_raises():
    with pytest.raises(ValueError):
        get_conditions_save_name("not_a_real_condition", experiment_conditions, 5)