# %% [markdown]
# 
# # Example line noise filtering script
# 
# Filters the 60Hz line noise from the data, as well as the harmonics. Includes
# environment checks for SLURM jobs for convenience
# 

# %%
import sys
print(sys.path)
sys.path.append("C:/Users/jz421/Desktop/GlobalLocal/IEEG_Pipelines/") #need to do this cuz otherwise ieeg isn't added to path...

import mne
import os
from ieeg.mt_filter import line_filter
from ieeg.io import get_data, raw_from_layout, save_derivative, update
from ieeg import viz
from bids import BIDSLayout
# from ieeg.viz.utils import figure_compare
from ieeg.navigate import trial_ieeg, channel_outlier_marker, crop_empty_data, outliers_to_nan
import pandas as pd


# %%
HOME = os.path.expanduser("~")
if 'SLURM_ARRAY_TASK_ID' in os.environ.keys():
    LAB_root = os.path.join(HOME, "workspace", "CoganLab")
    subject = int(os.environ['SLURM_ARRAY_TASK_ID'])
else:  # if not then set box directory
    LAB_root = os.path.join(HOME, "Box", "CoganLab")
    subject = 57

# %% [markdown]
# ### try new code from ieeg examples

# %%
"""
Line noise filtering script
===================================

Filters the 60Hz line noise from the data, as well as the harmonics. Includes
environment checks for SLURM jobs for convenience
"""

import mne
import os
from ieeg.io import save_derivative, raw_from_layout
from ieeg.mt_filter import line_filter
# from ieeg.viz.utils import figure_compare
from bids import BIDSLayout

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
subjects = ['D0121']
for subj in subjects:
    # Load the raw data without excluding any channels
    raw = raw_from_layout(layout, subject=subj, extension=".edf", desc=None, preload=True)

    # this is to exclude the eeg channels
    # List of channels you want to exclude. Gonna have to run this once, then grab the channels from the error message.
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

    # # %%
    # # plot the data before and after filtering
    # figure_compare([raw, filt],
    #             labels=["Un", ""],
    #             avg=True,
    #             n_jobs=6,
    #             verbose=10,
    #             proj=True,
    #             fmax=250)

    # filter again to get rid of harmonics
    line_filter(raw,
                mt_bandwidth=10.,
                n_jobs=6,
                filter_length='20000ms',
                verbose=10,
                freqs=[60],
                notch_widths=20,
                copy=False)

    # plot raw vs filt in a separate notebook later, don't want to load in two datasets into memory
    # # plot the data before and after filtering again
    # figure_compare([raw, filt],
    #                labels=["Un", ""],
    #                avg=True,
    #                n_jobs=6,
    #                verbose=10,
    #                proj=True,
    #                fmax=250)

    # save the data
    save_derivative(raw, layout, "clean", True)

    channel_outlier_marker(raw, 3, 2, save=True) #uhh try this again

    # filt.info['bads'] += channel_outlier_marker(filt, 3, 2, save=True)

# %%
# this is for testing, directly load the raw and do channel outlier marker on it
HOME = os.path.expanduser("~")

if os.name == 'nt':  # windows
    LAB_root = os.path.join(HOME, "Box", "CoganLab")
else:  # mac
    LAB_root = os.path.join(HOME, "Library", "CloudStorage", "Box-Box",
                            "CoganLab")

task='GlobalLocal'
subj = 'D0117'

layout = get_data(task, root=LAB_root)
raw = raw_from_layout(layout, subject=subj, extension=".edf", desc=None, preload=True)

channel_outlier_marker(raw, 3, 2, save=True) #uhh try this again

# %%
from ieeg.viz.ensemble import figure_compare

HOME = os.path.expanduser("~")
task='GlobalLocal'

if os.name == 'nt':  # windows
    LAB_root = os.path.join(HOME, "Box", "CoganLab")
else:  # mac
    LAB_root = os.path.join(HOME, "Library", "CloudStorage", "Box-Box",
                            "CoganLab")
    
layout = get_data(task, root=LAB_root)
actual_raw = raw_from_layout(layout, subject=subj, extension=".edf", desc=None, preload=True)

figure_compare([actual_raw, raw],
               labels=["Un", ""],
               avg=True,
               n_jobs=6,
               verbose=10,
               proj=True,
               fmax=250)

# %% [markdown]
# delete the below cell after it runs (02/20)

# %%
filenames = layout.get(return_type='filename', suffix='channels', extension='tsv', subject=subj)

# %%
bads = ['LTPS8', 'FP1', 'C4', 'T4', 'O2', 'LTMM1', 'F7', 'C3']
raw.info['bads'] = bads
file_paths = [r'C:\Users\jz421\Box\CoganLab\BIDS-1.1_GlobalLocal\BIDS\derivatives\clean\sub-D0100\ieeg\sub-D0100_task-GlobalLocal_acq-01_run-01_desc-clean_channels.tsv', r'C:\Users\jz421\Box\CoganLab\BIDS-1.1_GlobalLocal\BIDS\derivatives\clean\sub-D0100\ieeg\sub-D0100_task-GlobalLocal_acq-01_run-02_desc-clean_channels.tsv', r'C:\Users\jz421\Box\CoganLab\BIDS-1.1_GlobalLocal\BIDS\derivatives\clean\sub-D0100\ieeg\sub-D0100_task-GlobalLocal_acq-01_run-03_desc-clean_channels.tsv', r'C:\Users\jz421\Box\CoganLab\BIDS-1.1_GlobalLocal\BIDS\derivatives\clean\sub-D0100\ieeg\sub-D0100_task-GlobalLocal_acq-01_run-04_desc-clean_channels.tsv']

for file_path in file_paths:
    update(file_path, bads, status='bad')

# %% [markdown]
# ### the code below is broken 1/25/24 cuz new ieeg updates i think

# %%
HOME = os.path.expanduser("~")

# get box directory depending on OS
if os.name == 'nt': # windows
    LAB_root = os.path.join(HOME, "Box", "CoganLab")
else: # mac
    LAB_root = os.path.join(HOME, "Library", "CloudStorage", "Box-Box", "CoganLab")

layout = get_data("GlobalLocal", root=LAB_root)


## Load Data
layout = get_data("GlobalLocal", LAB_root)
subjects = layout.get_subjects()
subjects.sort()
print(subjects)

for subj in subjects:
    raw = raw_from_layout(layout, subject=subj, extension=".edf", desc=None,
                      preload=True)

    ## filter data
    filt = line_filter(raw, mt_bandwidth=10., n_jobs=-1, copy=False, verbose=10,
                filter_length='700ms', freqs=[60], notch_widths=20)
    
    filt.info['bads'] += channel_outlier_marker(filt, 3, 2, save=True)


    # line_filter(raw, mt_bandwidth=10., n_jobs=-1, copy=False, verbose=10,
    #             filter_length='20s', freqs=[60, 120, 180, 240],
    #             notch_widths=20)

    save_derivative(filt, layout, "clean", overwrite=True)

    print("Data saved successfully.")



# # Save data to save_dir
# file_names = ["data_raw.fif", "data_filt.fif"]  # Specify the desired file names
# for d, file_name in zip(data, file_names):
#     file_path = os.path.join(save_dir, file_name)
#     d.save(file_path, overwrite=True)


# for subj in subjects:
#     # if subj != "D0022":
#     #     continue
#     # Load the data
#     raw = raw_from_layout(layout, subject=subj,
#                            extension='.edf', preload=True)
#     filt = line_filter(raw, mt_bandwidth=10., n_jobs=6,
#                    filter_length='700ms', verbose=10,
#                    freqs=[60], notch_widths=20)


#     data = [raw, filt]
#     viz.utils.figure_compare(data, ["Un", ""], avg=True, n_jobs=6,
#                    verbose=10, proj=True, fmax=250)

# %%
HOME = os.path.expanduser("~")

# get box directory depending on OS
if os.name == 'nt': # windows
    LAB_root = os.path.join(HOME, "Box", "CoganLab")
else: # mac
    LAB_root = os.path.join(HOME, "Library", "CloudStorage", "Box-Box", "CoganLab")

layout = get_data("GlobalLocal", root=LAB_root)


## Load Data
layout = get_data("GlobalLocal", LAB_root)
subjects = layout.get_subjects()
subjects.sort()
print(subjects)

subj = 'D0102'
raw = raw_from_layout(layout, subject=subj, extension=".edf", desc=None,
                    preload=True)

## filter data
filt = line_filter(raw, mt_bandwidth=10., n_jobs=-1, copy=False, verbose=10,
            filter_length='700ms', freqs=[60], notch_widths=20)

filt.info['bads'] += channel_outlier_marker(filt, 3, 2, save=True)


# line_filter(raw, mt_bandwidth=10., n_jobs=-1, copy=False, verbose=10,
#             filter_length='20s', freqs=[60, 120, 180, 240],
#             notch_widths=20)

save_derivative(filt, layout, "clean", overwrite=True)

print("Data saved successfully.")

# %% [markdown]
# ### just one subject for testing

# %%
# get box directory depending on OS
if os.name == 'nt': # windows
    LAB_root = os.path.join(HOME, "Box", "CoganLab")
else: # mac
    LAB_root = os.path.join(HOME, "Library", "CloudStorage", "Box-Box", "CoganLab")

layout = get_data("GlobalLocal", LAB_root)
subj = "D0057"
raw = raw_from_layout(layout, subject=subj, extension=".edf", desc=None,
                    preload=True)

## filter data
filt = line_filter(raw, mt_bandwidth=10., n_jobs=-1, copy=False, verbose=10,
            filter_length='700ms', freqs=[60], notch_widths=20)

# line_filter(raw, mt_bandwidth=10., n_jobs=-1, copy=False, verbose=10,
#             filter_length='20s', freqs=[60, 120, 180, 240],
#             notch_widths=20)

# do channel outlier marker with max_rounds of 2
filt.info['bads'] += channel_outlier_marker(filt, 3, 2, save=True)

# filt.drop_channels(filt.info['bads'])

# update(filt, layout, "bad")


# and then feed into preprocess.py and then feed into wavelet/HG/stats

# update in check_chans saves the bad channels
# save_derivative(filt, layout, "clean", overwrite=True)

print("Data saved successfully.")

# %% [markdown]
# save_derivative(filt, layout, "clean")
# 
# 

# %% [markdown]
# # trying to make pre-experiment baseline

# %%
# add experiment start event to the events.tsv file

HOME = os.path.expanduser("~")

# get box directory depending on OS
if os.name == 'nt': # windows
    LAB_root = os.path.join(HOME, "Box", "CoganLab")
else: # mac
    LAB_root = os.path.join(HOME, "Library", "CloudStorage", "Box-Box", "CoganLab")

layout = get_data("GlobalLocal", root=LAB_root)

# Step 1: Read the existing file
sub = 'D0071'  # Replace this with the actual subject ID

events_path = os.path.join(layout.root, 'derivatives', 'clean', f'sub-{sub}', 'ieeg', 
                           f'sub-{sub}_task-GlobalLocal_acq-01_run-01_desc-clean_events.tsv')

df = pd.read_csv(events_path, sep='\t')

# Step 2: Create a new DataFrame for the custom event
new_row = pd.DataFrame({
    'onset': [0],
    'duration': [0],
    'trial_type': ['experimentStart'],
    'value': [None],
    'sample': [None]
})

# Concatenate the new row with the existing DataFrame
df = pd.concat([new_row, df]).reset_index(drop=True)

# Step 3: Write the updated DataFrame back to the events.tsv file
df.to_csv(events_path, sep='\t', index=False)


# %%
print(filt.annotations[0])  # this will show you all annotations and their onset times

# %%
filt.plot()

# %%
raw_selection = raw.copy().crop(tmin=0, tmax=300) #grab the first x seconds of the dataset. Onset is like 1300 seconds for D0057's first Stimulus event so I think we're good even if we do like 500 for everyone.

filt_selection = filt.copy().crop(tmin=0, tmax=300)

# %%
## Copy cropped data
good = filt_selection.copy()

# good.drop_channels(good.info['bads'])
good.info['bads'] = channel_outlier_marker(good, 3, 2)
good.drop_channels(good.info['bads'])
good.load_data()

ch_type = filt.get_channel_types(only_data_chs=True)[0]
good.set_eeg_reference(ref_channels="average", ch_type=ch_type)

# Remove intermediates from mem
good.plot()

 # make stimulus baseline EpochsTFR
times=[-1,3] #this is for 0.5 sec of padding on each side
trials = trial_ieeg(good, "Stimulus", times, preload=True)
outliers_to_nan(trials, outliers=10)
base = wavelet_scaleogram(trials, n_jobs=-2, decim=int(good.info['sfreq'] / 100))
crop_pad(base, "0.5s")


# %% [markdown]
# test d0116 to show aaron

# %%
"""
Line noise filtering script
===================================

Filters the 60Hz line noise from the data, as well as the harmonics. Includes
environment checks for SLURM jobs for convenience
"""

import mne
import os
from ieeg.io import save_derivative, raw_from_layout
from ieeg.mt_filter import line_filter
# from ieeg.viz.utils import figure_compare
from bids import BIDSLayout

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
subjects = ['D0116']
for subj in subjects:
    # Load the raw data without excluding any channels
    raw = raw_from_layout(layout, subject=subj, extension=".edf", desc=None, preload=True)

    # this is to exclude the eeg channels
    # List of channels you want to exclude. Gonna have to run this once, then grab the channels from the error message.
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

    # # %%
    # # plot the data before and after filtering
    # figure_compare([raw, filt],
    #             labels=["Un", ""],
    #             avg=True,
    #             n_jobs=6,
    #             verbose=10,
    #             proj=True,
    #             fmax=250)

    # filter again to get rid of harmonics
    line_filter(raw,
                mt_bandwidth=10.,
                n_jobs=6,
                filter_length='20000ms',
                verbose=10,
                freqs=[60],
                notch_widths=20,
                copy=False)

    # plot raw vs filt in a separate notebook later, don't want to load in two datasets into memory
    # # plot the data before and after filtering again
    # figure_compare([raw, filt],
    #                labels=["Un", ""],
    #                avg=True,
    #                n_jobs=6,
    #                verbose=10,
    #                proj=True,
    #                fmax=250)

    # save the data
    save_derivative(raw, layout, "clean", True)

    channel_outlier_marker(raw, 3, 2, save=True) #uhh try this again

    # filt.info['bads'] += channel_outlier_marker(filt, 3, 2, save=True)


