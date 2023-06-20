import mne.time_frequency
import mne
from ieeg.io import get_data, raw_from_layout
from ieeg.navigate import trial_ieeg, channel_outlier_marker, crop_empty_data, outliers_to_nan
from ieeg.calc.scaling import rescale
import os
from ieeg.timefreq.utils import wavelet_scaleogram, crop_pad
import numpy as np

# manually set this for now since mac is different from windows, make more robust later
# HOME = '/Users/jinjiang-macair/Library/CloudStorage'
# os.path.expanduser("~") SHOULD be '/Users/jinjiang-macair/'
HOME = os.path.expanduser("~")

# get box directory depending on OS
if os.name == 'nt': # windows
    LAB_root = os.path.join(HOME, "Box", "CoganLab")
else: # mac
    LAB_root = os.path.join(HOME, "Library", "CloudStorage", "Box-Box", "CoganLab")

layout = get_data("GlobalLocal", root=LAB_root)
subjects = layout.get(return_type="id", target="subject")

print(subjects)
for sub in subjects:
    # if sub != "D0022":
    #     continue
    # Load the data
    filt = raw_from_layout(layout, subject=sub,
                           extension='.edf', preload=False)
    print(filt)

    ## Crop raw data to minimize processing time
    good = crop_empty_data(filt)

    # good.drop_channels(good.info['bads'])
    good.info['bads'] = channel_outlier_marker(good, 3, 2)
    good.drop_channels(good.info['bads'])
    # good.info['bads'] += channel_outlier_marker(good, 4, 2)
    # good.drop_channels(good.info['bads'])
    good.load_data()

    ch_type = filt.get_channel_types(only_data_chs=True)[0]
    good.set_eeg_reference(ref_channels="average", ch_type=ch_type)

    # Remove intermediates from mem
    # good.plot()

    ## epoching and trial outlier removal

    save_dir = os.path.join(layout.root, 'derivatives', 'spec', 'wavelet', sub)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # This part will change when I fix the events
    for epoch, t, name in zip(
            ("Stimulus", "Stimulus",  "Response"),
            ((-0.5, 0), (0, 1), (-0.5, 0.5)),
            ("base", "stim", "resp")):
        times = [None, None]
        times[0] = t[0] - 0.5
        times[1] = t[1] + 0.5
        trials = trial_ieeg(good, epoch, times, preload=True)
        outliers_to_nan(trials, outliers=10)
        spec = wavelet_scaleogram(trials, n_jobs=-2, decim=int(
            good.info['sfreq'] / 100))
        crop_pad(spec, "0.5s")
        if name == "base":
            base = spec.copy()
            continue
        spec_a = rescale(spec, base, copy=True, mode='ratio').average(
            lambda x: np.nanmean(x, axis=0), copy=True)
        spec_a._data = np.log10(spec_a._data) * 20
        fnames = [os.path.relpath(f, layout.root) for f in good.filenames]
        spec_a.info['subject_info']['files'] = tuple(fnames)
        spec_a.info['bads'] = good.info['bads']
        filename = os.path.join(save_dir, f'{name}-tfr.h5')
        mne.time_frequency.write_tfrs(filename, spec_a, overwrite=True)
        spec_a.save(os.path.join(save_dir, f'{name}-avg.fif'), overwrite=True)


