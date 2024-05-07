import mne.time_frequency
import mne
from ieeg.io import get_data, raw_from_layout
from ieeg.navigate import trial_ieeg, channel_outlier_marker, crop_empty_data, outliers_to_nan
from ieeg.calc.scaling import rescale
import os
from ieeg.timefreq.utils import wavelet_scaleogram, crop_pad
import numpy as np

from utils import calculate_RTs

# ### 7/5 try to get wavelets for all stimulus 

# ### testing with one subject

HOME = os.path.expanduser("~")

# get box directory depending on OS
if os.name == 'nt': # windows
    LAB_root = os.path.join(HOME, "Box", "CoganLab")
else: # mac
    LAB_root = os.path.join(HOME, "Library", "CloudStorage", "Box-Box", "CoganLab")

layout = get_data("GlobalLocal", root=LAB_root)


layout = get_data("GlobalLocal", root=LAB_root)
subjects = layout.get(return_type="id", target="subject")

print(subjects)

sub='D0071' #why does this not work in the for loop???
# Load the data
filt = raw_from_layout(layout.derivatives['derivatives/clean'], subject=sub,
                        extension='.edf', desc='clean', preload=False) #get line-noise filtered data
print(filt)

## Crop raw data to minimize processing time
good = crop_empty_data(filt)

good.info['bads'] = channel_outlier_marker(good, 3, 2)
good.drop_channels(good.info['bads'])

good.load_data()

ch_type = filt.get_channel_types(only_data_chs=True)[0]
good.set_eeg_reference(ref_channels="average", ch_type=ch_type)

# Remove intermediates from mem
good.plot()

## epoching and trial outlier removal

save_dir = os.path.join(layout.root, 'derivatives', 'spec', 'wavelet', sub)
if not os.path.exists(save_dir):
        os.makedirs(save_dir)

good._data.shape

RTs, skipped = calculate_RTs(good)
avg_RT = np.median(RTs)
print(avg_RT)
# make stimulus baseline EpochsTFR
times=[-1.5, avg_RT+1.3+0.5] #this is for 0.5 sec of padding on each side
print("times: " + str(times))
trials = trial_ieeg(good, "Stimulus", times, preload=True)
print("base trials shape is: " + str(trials._data.shape))
outliers_to_nan(trials, outliers=10)
base = wavelet_scaleogram(trials, n_jobs=-2, decim=int(good.info['sfreq'] / 100))
crop_pad(base, "0.5s")

print("done with base")
#now do rescale with the concatenated baseline epochs
for event, t in zip(("Stimulus", "Response"),((-1, 2), (-1, 2))):
        times = [None, None]
        times[0] = t[0] - 0.5
        times[1] = t[1] + 0.5
        trials = trial_ieeg(good, event, times, preload=True)
        print(f"trials for {event} is: " + str(trials._data.shape))
        outliers_to_nan(trials, outliers=10)
        spec = wavelet_scaleogram(trials, n_jobs=-2, decim=int(good.info['sfreq'] / 100))
        crop_pad(spec, "0.5s")
        spec_a = rescale(spec, base, copy=True, mode='ratio').average(
                lambda x: np.nanmean(x, axis=0), copy=True)
        spec_a._data = np.log10(spec_a._data) * 20
        fnames = [os.path.relpath(f, layout.root) for f in good.filenames]
        spec_a.info['subject_info']['files'] = tuple(fnames)
        spec_a.info['bads'] = good.info['bads']
        filename = os.path.join(save_dir, f'{event}_fullTrialBase-tfr.h5')
        mne.time_frequency.write_tfrs(filename, spec_a, overwrite=True)
        spec_a.save(os.path.join(save_dir, f'{event}_fullTrialBase-avg.fif'), overwrite=True)

good.info