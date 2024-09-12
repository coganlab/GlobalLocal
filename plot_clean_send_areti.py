import sys
print(sys.path)
sys.path.append("C:/Users/jz421/Desktop/GlobalLocal/IEEG_Pipelines/") #need to do this cuz otherwise ieeg isn't added to path...

import mne
import os
from ieeg.mt_filter import line_filter
from ieeg.io import get_data, raw_from_layout, save_derivative, update
from ieeg import viz
from bids import BIDSLayout
from ieeg.viz.utils import figure_compare
from ieeg.navigate import trial_ieeg, channel_outlier_marker, crop_empty_data, outliers_to_nan
import pandas as pd
import mne
import os
from ieeg.io import save_derivative, raw_from_layout
from ieeg.mt_filter import line_filter
from ieeg.viz.utils import figure_compare
from bids import BIDSLayout

HOME = os.path.expanduser("~")
if 'SLURM_ARRAY_TASK_ID' in os.environ.keys():
    LAB_root = os.path.join(HOME, "workspace", "CoganLab")
    subject = int(os.environ['SLURM_ARRAY_TASK_ID'])
else:  # if not then set box directory
    LAB_root = os.path.join(HOME, "Box", "CoganLab")
    subject = 57


# %%
# Set up paths
# ------------
HOME = os.path.expanduser("~")

# get box directory depending on OS
if os.name == 'nt': # windows
    LAB_root = os.path.join(HOME, "Box", "CoganLab")
else: # mac
    LAB_root = os.path.join(HOME, "Library", "CloudStorage", "Box-Box", "CoganLab")


## Load Data
layout = get_data("GlobalLocal", LAB_root)
subjects = layout.get_subjects()
subjects.sort()
print(subjects)

#this is prob gonna fail at d0100 cuz eeg channels
subjects = ['D0110']
for subj in subjects:
    # Load the raw data without excluding any channels
    raw = raw_from_layout(layout, subject=subj, extension=".edf", desc=None, preload=True)

    # this is to exclude the eeg channels
    # List of channels you want to exclude (EEG channels if they have them) Gonna have to run this once, then grab the channels from the error message.
    # channels_to_exclude = ['T5', 'T6', 'FZ', 'CZ', 'PZ', 'FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', '02', 'F7', 'F8', 'T3', 'T4']
    # Drop the channels you want to exclude
    # raw.drop_channels(channels_to_exclude)

    # %%
    # Filter Data
    # -----------
    # A filter length of 700 ms does a good job of removing 60Hz line noise, while
    # a Filter length of 20000 ms does a good job of removing the harmonics (120Hz,
    # 180Hz, 240Hz)

    # can delete picks parameter if not excluding the eeg channels
    line_filter(raw,
                mt_bandwidth=10.,
                n_jobs=6,
                filter_length='700ms',
                verbose=10,
                freqs=[60, 120, 180],
                notch_widths=20,
                copy=False)


    # filter again to get rid of harmonics
    line_filter(raw,
                mt_bandwidth=10.,
                n_jobs=6,
                filter_length='20000ms',
                verbose=10,
                freqs=[60],
                notch_widths=20,
                copy=False)

    # save the data
    save_derivative(raw, layout, "clean", True)

    channel_outlier_marker(raw, 3, 2, save=True) #uhh try this again

    # this is just for plotting unfiltered vs filtered for each subject. Not necessary.
    actual_raw = raw_from_layout(layout, subject=subj, extension=".edf", desc=None, preload=True)

    figure_compare([actual_raw, raw],
                labels=["Un", ""],
                avg=True,
                n_jobs=6,
                verbose=10,
                proj=True,
                fmax=250)