import mne.time_frequency
import mne
from ieeg.io import get_data, raw_from_layout
from ieeg.navigate import trial_ieeg, channel_outlier_marker, crop_empty_data, outliers_to_nan
from ieeg.calc.scaling import rescale
import os
from ieeg.timefreq.utils import wavelet_scaleogram, crop_pad
import numpy as np
from misc_functions import calculate_RTs

def get_wavelet_baseline(inst: mne.io.BaseRaw, base_times: tuple[float, float]):
    inst = inst.copy()
    inst.load_data()
    ch_type = inst.get_channel_types(only_data_chs=True)[0]
    inst.set_eeg_reference(ref_channels="average", ch_type=ch_type)

    adjusted_base_times = [base_times[0] - 0.5, base_times[1] + 0.5]
    trials = trial_ieeg(inst, "Stimulus", adjusted_base_times, preload=True)
    outliers_to_nan(trials, outliers=10)
    base = wavelet_scaleogram(trials, n_jobs=-2, decim=int(inst.info['sfreq'] / 100))
    crop_pad(base, "0.5s")
    del inst
    return base