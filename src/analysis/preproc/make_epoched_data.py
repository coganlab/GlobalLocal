# %% [markdown]
# 
# # Example of High Gamma Filter
# 
# Below is a code sample for extracting high gamma power from a raw data file, followed by permutation cluster stats on that high gamma power data
# 

# %% [markdown]
# ### working version 12/1/23

# %% [markdown]

# %% [markdown]
# use window stats with perm testing (0 to 0.5, 0.5 to 1, 0 to 1 sec relative to stim onset)

# %%
import sys
import os
print(sys.path)
# sys.path.append("C:/Users/jz421/Desktop/GlobalLocal/IEEG_Pipelines/") #need to do this cuz otherwise ieeg isn't added to path...comment out when running on cluster, but uncomment when running on pc.

# Get the absolute path to the directory containing the current script
# For GlobalLocal/src/analysis/preproc/make_epoched_data.py, this is GlobalLocal/src/analysis/preproc
current_script_dir = os.path.dirname(os.path.abspath(__file__))

# Navigate up three levels to get to the 'GlobalLocal' directory
project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..', '..'))

# Add the 'GlobalLocal' directory to sys.path if it's not already there
if project_root not in sys.path:
    sys.path.insert(0, project_root) # insert at the beginning to prioritize it

import pandas as pd
import json
from statsmodels.stats.multitest import multipletests
from ieeg.navigate import channel_outlier_marker, trial_ieeg, crop_empty_data, \
    outliers_to_nan
from ieeg.io import raw_from_layout, get_data
from ieeg.timefreq.utils import crop_pad
from ieeg.timefreq import gamma
from ieeg.calc.scaling import rescale
import mne
import numpy as np
from ieeg.calc.stats import time_perm_cluster
from ieeg.calc.fast import mean_diff, ttest
from ieeg.viz.mri import gen_labels
import matplotlib.pyplot as plt
from mne.utils import fill_doc, verbose
import random
from contextlib import redirect_stdout

print(sys.path)
sys.path.append("C:/Users/jz421/Desktop/GlobalLocal/IEEG_Pipelines/") #need to do this cuz otherwise ieeg isn't added to path...
import pickle
from functools import partial
from src.analysis.utils.general_utils import calculate_RTs, save_channels_to_file, save_sig_chans, load_sig_chans, bad_channels_from_trial_mask, impute_trial_nans_by_channel_mean, get_default_LAB_root, crop_empty_data_fixed, resolve_lab_root
from src.analysis.power.block_diagnostics import max_abs_z_per_trial
from src.analysis.utils.epoch_metadata_utils import make_metadata_from_event_names, add_previous_trial_info
from src.analysis.preproc.epoch_helpers import trial_ieeg_rand_offset, shuffle_array

def bandpass_and_epoch_and_find_task_significant_electrodes(sub, task='GlobalLocal', times=(-1, 1.5),
                      within_base_times=(-1, 0), base_times_length=0.5, baseline_event="Stimulus", pad_length = 3, LAB_root=None, channels=None, dec_factor=8, 
                      outlier_policy='drop_and_impute', outliers=10, threshold_percent=2.0, max_abs_z=None, passband=(70,150), filter_method='filterbank_hilbert', method='fir', fir_design='firwin',
                      stat_func=ttest):
    """
    Bandpass the filtered data, epoch around Stimulus and Response onsets, and find electrodes with significantly different activity from baseline for a given subject.

    Parameters:
    - sub (str): The subject identifier.
    - task (str): The task identifier.
    - times (tuple, optional): A tuple indicating the start and end times for event processing. Defaults to (-1, 1.5).
    - times (tuple [float, float]): The time window to epoch around the event.
    - within_base_times (tuple [float, float]): The time window within which to randomly select intervals for each event, for baseline.
    - base_times_length (float): The length of the time intervals to randomly select within `within_base_times`. 
    - baseline_event (str): The event to use for baseline. Use "experimentStart" for beginning of experiment, or use "Stimulus" for pre-stimulus. Not sure if "experimentStart" actually exists, check how to make this. 
    - pad_length (float): The length to pad each time interval. Will be removed later.
    - LAB_root (str, optional): The root directory for the lab. Will be determined based on OS if not provided. Defaults to None.
    - channels (list of strings, optional): The channels to plot and get stats for. Default is all channels.
    - decimation_factor (int, optional): The factor by which to subsample the data. Default is 10, so should be 2048 Hz down to 204.8 Hz.
    - outlier_policy (str, optional): How to handle outliers. Either set to drop, nan, drop_and_nan, drop_and_impute, or ignore. Note that drop in this case refers to dropping channels with more % outliers (combined from voltage and zscore) than the threshold percentage, not to dropping the voltage outlier trials themselves. NaN will replace the voltage outlier trials with NaNs, and impute will replace them with the channel mean. The zscore outlier trials are all marked as NaN, regardless of policy.
    - outliers (int, optional): How many standard deviations above the mean for a trial to be considered an outlier. Default is 10.
    - threshold_percent (int | float, optional): Channels with a greater percent of outlier trials than this threshold will be removed from further analyses, if using an outlier policy that drops. A trial counts as an outlier for a channel if either rejection pass flagged it: the raw-voltage `outliers` pass or the `max_abs_z` pass. The list is decided on the Stimulus pass and reused for Response, so both events keep one channel set.
    - max_abs_z (float | None, optional): Reject a (trial, channel) trace whose baseline-normalized power exceeds this anywhere in the epoch. Catches artifacts the raw-voltage `outliers` pass cannot see, since power ratio is amplitude ratio squared. Rejections here also count toward `threshold_percent`. None disables. Default None.
    - passband (tuple, optional): The frequency range for the frequency band of interest. Default is (70, 150).
    - filter_method (str, optional): The filtering method to use for extracting your chosen passband. Currently can either use filterbank_hilbert (https://naplib-python.readthedocs.io/en/latest/references/preprocessing.html#naplib.preprocessing.filterbank_hilbert) or bandpass (which will apply a FIR bandpass filter).
    - method (str, optional): The bandpass method to use if you use bandpass as the filter_method. Default is to use fir but can use iir.
    - fir_design (str, optional): The fir design for the bandpass filter if you use bandpass as the filter_method. Default is firwin but can use firwin2 or other things.
    - stat_func (function, optional): The statistical function to use for significance testing. Default is ieeg.calc.fast.ttest, time_perm_cluster's own default.    
    
    This function will process the provided event for a given subject and task.
    Bandpassed and epoched data will be computed, and statistics will be calculated and plotted.
    The results will be saved to output files.
    """
    print("=" * 70)
    print(f'epoching data for subject: {sub}')

    # Determine LAB_root based on the operating system and environment
    LAB_root = resolve_lab_root(LAB_root)
    
    layout = get_data(task, root=LAB_root)
    filt = raw_from_layout(layout.derivatives['derivatives/clean'], subject=sub,
                        extension='.edf', desc='clean', preload=False)
    save_dir = os.path.join(layout.root, 'derivatives', 'freqFilt', 'figs', sub)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    print("Use my new crop empty data function that gets rid of the mne annotations extras dict")
    good = crop_empty_data_fixed(filt)
    # %%

    print(f"good channels before dropping bads: {len(good.ch_names)}")
    print(f"filt channels before dropping bads: {len(filt.ch_names)}")
    all_channels_before_marker = good.ch_names.copy()
    channels_to_drop = []

    good.info['bads'] = channel_outlier_marker(good, 3, 2)
    marker_bad_channels = good.info['bads'].copy()

    print("Bad channels in 'good':", good.info['bads'])

    filt.drop_channels(marker_bad_channels)  # this has to come first cuz if you drop from good first, then good.info['bads'] is just empty
    good.drop_channels(marker_bad_channels)

    print("Bad channels in 'good' after dropping once:", good.info['bads'])

    print(f"good channels after dropping bads: {len(good.ch_names)}")
    print(f"filt channels after dropping bads: {len(filt.ch_names)}")

    good.load_data()

    # If channels is None, use all channels
    if channels is not None:
        # Validate the provided channels
        invalid_channels = [ch for ch in channels if ch not in good.ch_names]
        if invalid_channels:
            raise ValueError(
                f"The following channels are not valid: {invalid_channels}")

        # Use only the specified channels
        good.pick_channels(channels)

    '''
    Keep only data channels (seeg/ecog/eeg/...), dropping any misc, ecg, stim,
    trigger, etc. contacts that came through the clean derivative.

    This has to happen before epoching because rescale (ieeg.calc.scaling) picks
    'data' internally on both of its arguments: it returns
    line.pick('data') and it picks the baseline *in place*. So with a single
    non-data channel present, HG_ev1 / HG_ev1_power keep every channel while
    HG_ev1_rescaled / HG_ev1_power_rescaled (and HG_base, silently, from the
    first rescale call onwards) keep only the data channels. That is the
    channel count mismatch the z-reject mask trips on, and it also puts
    HG_ev1._data and HG_base._data on different channel sets for
    time_perm_cluster. Picking once here gives the whole function one channel
    set.
    '''
    channels_before_data_pick = good.ch_names.copy()
    good.pick('data')
    non_data_channels = [ch for ch in channels_before_data_pick if ch not in good.ch_names]
    if non_data_channels:
        print(f"Dropped {len(non_data_channels)} non-data channels: {non_data_channels}")
    channels_after_data_pick = good.ch_names.copy()

    ch_type = filt.get_channel_types(only_data_chs=True)[0]
    good.set_eeg_reference(ref_channels="average", ch_type=ch_type)

    '''
    <<< Step 1: PROCESS the baseline ONCE, on the full channel set >>>

    The bad-channel decision used to be made here, before the baseline, from a
    throwaway Stimulus epoching scored by the raw-voltage outlier pass alone.
    It cannot be made here any more: a channel is now dropped on the union of
    that pass and the max_abs_z pass, and the z-scores do not exist until the
    epochs have been rescaled against this baseline. Deciding first and
    rescaling second is circular, which is what makes the ordering awkward.

    Nothing is lost by deferring the drop instead. rescale normalizes each
    channel against its own baseline and the z threshold is per
    (trial, channel), so which other channels are present changes neither
    number: dropping a channel before rescaling and dropping it after give
    identical data for every channel that survives. The drop therefore happens
    at the bottom of the event loop, applied to the epochs and to a per-event
    copy of this baseline -- which is also where the only two things that
    genuinely require a shared channel set happen, saving and
    time_perm_cluster. The copy is what keeps the Response pass working: the
    baseline is built once and shared, so it has to stay on the full channel
    set for whichever event runs second.
    '''
    base_trials = trial_ieeg_rand_offset(good, baseline_event, within_base_times, base_times_length, pad_length, preload=True)
    if outlier_policy in ('nan', 'drop_and_nan', 'drop_and_impute'):
        outliers_to_nan(base_trials, outliers=outliers)
    else:
        print('ignoring outliers in the baseline')
        
    if filter_method == 'filterbank_hilbert':
        HG_base = gamma.extract(base_trials, passband=passband, copy=False, n_jobs=1)
    elif filter_method == 'bandpass':
        HG_base = base_trials.copy().filter(passband[0], passband[1], method=method, fir_design=fir_design)
        HG_base.apply_hilbert(envelope=True)
    else:
        raise ValueError("Please choose filterbank_hilbert or bandpass as your filter method. Other filter methods are not yet supported.")
    
    pad_length_string = f"{pad_length}s"
    crop_pad(HG_base, pad_length_string)
    HG_base.decimate(dec_factor)
    
    # Square the data to get power from amplitude (FIXED: Now correctly placed)
    HG_base_power = HG_base.copy()
    HG_base_power._data = HG_base._data ** 2

    '''
    The baseline is deliberately NOT imputed, even under 'drop_and_impute'.

    ieeg.calc.scaling.rescale reduces it through dist(), which builds
    where=~isnan(mat) and hands that to np.mean/np.std, so NaN baseline trials
    are excluded from the per-channel mean and SD rather than poisoning them.
    Filling them instead would add samples sitting exactly at the mean, which
    shrinks the SD and inflates every z-score on that channel -- and z is what
    the max_abs_z pass thresholds and what HG_ev1_rescaled is reported in.
    Leaving them NaN gives an unbiased SD over a correctly reduced N.

    The event epochs are still imputed under that policy (below), because there
    the point is a NaN-free array for downstream consumers that are not NaN
    aware, not a normalizing statistic.
    '''

    if isinstance(stat_func, partial):
        base_func_name = stat_func.func.__name__
        # Create a descriptive name like "ttest_ind_equal_var_False"
        keywords_str = "_".join(f"{k}_{v}" for k, v in sorted(stat_func.keywords.items()))
        if keywords_str: # If there are keywords like equal_var
            stat_func_for_filename = f"{base_func_name}_{keywords_str}"
        else: # If partial was used without keywords (less likely here)
            stat_func_for_filename = base_func_name
    elif hasattr(stat_func, '__name__'): # For regular functions
        stat_func_for_filename = stat_func.__name__
    elif isinstance(stat_func, str): # If a string was somehow passed (e.g., from a less robust CLI)
        # Sanitize or use the string directly if it's simple.
        # For safety, you might want to ensure it's a valid filename component.
        stat_func_for_filename = stat_func.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
    else:
        stat_func_for_filename = "custom_stat_func" # Fallback
        
    # output_name_base = f"{base_times_length}sec_within{within_base_times[0]}-{within_base_times[1]}sec_base_decFactor_{dec_factor}_outliers_{outliers}_{outlier_policy}_thresh_perc_{threshold_percent}_{passband[0]}-{passband[1]}_Hz_padLength_{pad_length}s_{filter_method}_stat_func_{stat_func_for_filename}"
    
    # drop the baseline stuff from the name for now since it's pretty constant across iterations
    output_name_base = f"decFactor_{dec_factor}_outliers_{outliers}_{outlier_policy}_thresh_perc_{threshold_percent}_{passband[0]}-{passband[1]}_Hz_padLength_{pad_length}s_{filter_method}_stat_func_{stat_func_for_filename}"

    if max_abs_z is not None:
        output_name_base += f"_zmax_{max_abs_z:g}"
        
    for event in ["Stimulus", "Response"]:
        print(f"--- Processing Event: {event} ---")
        output_name_event = f'{event}_{times[0]}to{times[1]}sec_{output_name_base}'

        times_adj = [times[0] - pad_length, times[1] + pad_length]
        
        trials = trial_ieeg(good, event, times_adj, preload=True, reject_by_annotation=False)
        trials.metadata = make_metadata_from_event_names(trials) # add metadata so we can grab specific trial types later. Untested 2/5/26.
        trials.metadata = add_previous_trial_info(trials.metadata)

        '''
        First outlier pass, on raw voltage. Two things come out of it: the NaNs
        it writes into the data, for the policies that keep them, and
        `voltage_bad`, a (n_trial, n_channel) record of which traces it caught.

        The record is the part that is new. 'drop' is why it is needed: that
        policy prunes channels but deliberately keeps every trial, so it never
        writes these NaNs into the analysis data at all and has to score them
        on a throwaway copy. Reading the NaNs back off the epochs at decision
        time -- the obvious alternative -- would therefore count nothing under
        the policy that main() actually defaults to. Keeping the record for
        every policy means the drop rule below reads the same way regardless,
        and the QC json can report what each pass contributed separately.

        Imputation deliberately does NOT happen here. It waits until after the
        z pass, so both kinds of rejected trial are filled the same way, and so
        the fill lands in gamma-band units rather than in raw voltage: the
        per-timepoint mean across trials of a set of ~100 Hz oscillations is
        close to zero, so a trial imputed before gamma.extract comes out with
        almost no high gamma power and then reads as a large negative z.
        '''
        if outlier_policy == 'drop':
            # Cheaper than the second `trial_ieeg` call this replaces: same
            # size, no re-read from disk, and it is freed here rather than
            # staying alive across both events and the gamma extraction.
            voltage_qc = trials.copy()
            outliers_to_nan(voltage_qc, outliers=outliers)
            voltage_bad = np.isnan(voltage_qc._data).any(axis=-1)
            del voltage_qc
        elif outlier_policy in ('nan', 'drop_and_nan', 'drop_and_impute'):
            outliers_to_nan(trials, outliers=outliers)
            voltage_bad = np.isnan(trials._data).any(axis=-1)
        else:
            print('ignoring outliers')
            voltage_bad = np.zeros((len(trials), len(trials.ch_names)), dtype=bool)

        print(f" [voltage-reject] {voltage_bad.sum()} (trial, channel) pairs over "
              f"{outliers} SD across {len(np.unique(np.nonzero(voltage_bad)[0]))} trials")

        # Now extract gamma and proceed with analysis
        if filter_method == 'filterbank_hilbert':
            HG_ev1 = gamma.extract(trials, passband=passband, copy=True, n_jobs=1)
        elif filter_method == 'bandpass':
            HG_ev1 = trials.copy().filter(passband[0], passband[1], method=method, fir_design=fir_design)
            HG_ev1.apply_hilbert(envelope=True) # get real signal only - if you set envelope=False, you can grab the complex signal and then do np.angle(HG_ev1._data) to get the phase. But make sure to do that before decimation cuz that will mess up the phase calculation.
        else:
            raise ValueError("Please choose filterbank_hilbert or bandpass as your filter method. Other filter methods are not yet supported.")
        
        crop_pad(HG_ev1, pad_length_string)
        HG_ev1.decimate(dec_factor)
        # Square the data to get power from amplitude
        HG_ev1_power = HG_ev1.copy()
        HG_ev1_power._data = HG_ev1._data ** 2 # Square amplitude to get power

        # get the rescaled amplitude
        HG_ev1_rescaled = rescale(HG_ev1, HG_base, copy=True, mode='zscore')

        # get the rescaled power
        HG_ev1_power_rescaled = rescale(HG_ev1_power, HG_base_power, copy=True, mode='zscore')

        '''
        Second outlier pass, in normalized units. The first pass runs on raw
        voltage before gamma.extract, so it cannot see how large an excursion
        becomes after squaring: power ratio is amplitude ratio squared, and 
        since baseline power is roughly exponential (sigma ~ mu), z is close to
        the raw power ratio. A 155x amplitude excursion reads as ~24000 z.
        Being per channel, the voltage pass also NaNs such a trial on whichever
        contact it is largest while missing it on neighbours (D0121 trial 367:
        NaN'd on LFMI8/9, left at 331 z on LFMI5).
        
        Absolute threshold, not another N-SD rule: rescale already normalizes
        per channel, so z is comparable across trials, and an SD recomputed across trials would itself be inflated by the outliers being removed --
        that is what let a 278 z trial through the first pass.
        
        One mask from the rescaled power, applied to every derived object, so the amplitude and power views agree on which trials exist.
        '''
        # Everything below indexes these five objects with one (trial, channel)
        # mask, so they have to be on the same channel set first. This is the
        # check that caught rescale picking 'data' in place on the baseline.
        for name, obj in (("trials", trials), ("HG_ev1", HG_ev1),
                          ("HG_ev1_power", HG_ev1_power), ("HG_ev1_rescaled", HG_ev1_rescaled)):
            if obj.ch_names != HG_ev1_power_rescaled.ch_names:
                missing = [ch for ch in obj.ch_names if ch not in HG_ev1_power_rescaled.ch_names]
                extra = [ch for ch in HG_ev1_power_rescaled.ch_names if ch not in obj.ch_names]
                raise RuntimeError(
                    f"channel set mismatch between {name} ({len(obj.ch_names)} channels) and "
                    f"HG_ev1_power_rescaled ({len(HG_ev1_power_rescaled.ch_names)} channels). "
                    f"Only in {name}: {missing}. Only in HG_ev1_power_rescaled: {extra}.")

        if max_abs_z is not None:
            # Shared with the threshold-choosing diagnostics in
            # src/analysis/vis/trial_z_distribution_vis.py, so the distribution
            # you read a value off is the one this line acts on.
            z_bad = max_abs_z_per_trial(HG_ev1_power_rescaled.get_data()) > max_abs_z
            if z_bad.any():
                print(f" [z-reject] {z_bad.sum()} (trial, channel) pairs over "
                      f"|z|>{max_abs_z} across {len(np.unique(np.nonzero(z_bad)[0]))} trials")

            for obj in (HG_ev1, HG_ev1_power, HG_ev1_rescaled, HG_ev1_power_rescaled):
                obj._data[z_bad] = np.nan
        else:
            z_bad = np.zeros_like(voltage_bad)

        '''
        <<< Channel drop, on both rejection passes at once >>>

        The whole point of doing it here rather than before the baseline: a
        trial the z pass threw out is as bad for the channel as one the voltage
        pass threw out, so it should count the same toward the
        threshold_percent budget. The union is safe to take because the two
        passes never flag the same pair twice -- max_abs_z_per_trial returns
        NaN for a pair the voltage pass already emptied, and NaN compares False
        against any threshold -- so `bad_trials.sum()` is a true count of
        distinct rejected pairs.

        '''
        bad_trials = voltage_bad | z_bad

        if outlier_policy in ('drop', 'drop_and_nan', 'drop_and_impute'):
            if event == "Stimulus":
                print(f"--- Identifying channels with more than {threshold_percent}% bad trials, "
                      f"counting the raw-voltage pass ({outliers} SD) and the z pass "
                      f"(|z| > {max_abs_z}) together (discuss this approach with Greg) ---")
                channels_to_drop = bad_channels_from_trial_mask(
                    bad_trials, HG_ev1_power_rescaled.ch_names, threshold_percent)

                bad_percent_by_channel = bad_trials.mean(axis=0) * 100
                qc_summary = {
                    "subject": sub,
                    "n_total_channels_before_marker": len(all_channels_before_marker),
                    "channels_before_marker": all_channels_before_marker,
                    "n_bad_by_channel_outlier_marker": len(marker_bad_channels),
                    "bad_by_channel_outlier_marker": marker_bad_channels,
                    "n_non_data_channels": len(non_data_channels),
                    "non_data_channels": non_data_channels,
                    "n_channels_after_data_pick": len(channels_after_data_pick),
                    "channels_after_data_pick": channels_after_data_pick,
                    "n_bad_by_trial_outliers": len(channels_to_drop),
                    "bad_by_trial_outliers": channels_to_drop,
                    # channels after both drops = channels after marker - trial-outlier drops
                    "n_channels_after_all_drops": len(good.ch_names) - len(channels_to_drop),
                    # what each rejection pass contributed, in (trial, channel) pairs
                    "threshold_percent": threshold_percent,
                    "outliers_sd": outliers,
                    "max_abs_z": max_abs_z,
                    "n_pairs_rejected_by_voltage": int(voltage_bad.sum()),
                    "n_pairs_rejected_by_max_abs_z": int(z_bad.sum()),
                    "n_pairs_rejected_total": int(bad_trials.sum()),
                    "percent_bad_trials_by_dropped_channel": {
                        ch: float(bad_percent_by_channel[HG_ev1_power_rescaled.ch_names.index(ch)])
                        for ch in channels_to_drop
                    },
                }

                qc_filepath = os.path.join(save_dir, f"{sub}_{output_name_event}_channel_qc_summary.json")
                with open(qc_filepath, "w") as f:
                    json.dump(qc_summary, f, indent=4)

                print(f"Saved QC summary for {sub} to {qc_filepath}")
            else:
                print(f"Reusing the {len(channels_to_drop)} channels dropped on the Stimulus pass "
                      f"so both events keep one channel set: {channels_to_drop}")

            for obj in (HG_ev1, HG_ev1_power, HG_ev1_rescaled, HG_ev1_power_rescaled):
                obj.drop_channels(channels_to_drop, on_missing='ignore')

        if outlier_policy == 'drop_and_impute':
            # The single imputation point, for both rejection passes at once:
            # the voltage NaNs rode through gamma.extract untouched and the
            # z NaNs were written just above, so one call per object fills them
            # with the same per-channel, per-timepoint mean.
            for obj in (HG_ev1, HG_ev1_power, HG_ev1_rescaled, HG_ev1_power_rescaled):
                impute_trial_nans_by_channel_mean(obj)

        # One channel set for everything this event saves and tests. The
        # baseline is built once and shared, so the drop goes on a copy --
        # HG_base itself has to stay whole for the event that runs second.
        HG_base_event = HG_base.copy()
        HG_base_event.drop_channels(channels_to_drop, on_missing='ignore')
        HG_base_power_event = HG_base_power.copy()
        HG_base_power_event.drop_channels(channels_to_drop, on_missing='ignore')

        # After dropping channels, update the channels list to match the data
        channels_after_dropping_bad_channels = HG_ev1.ch_names

        # get the evoke and evoke rescaled amplitude
        HG_ev1_evoke = HG_ev1.average(method=lambda x: np.nanmean(x, axis=0)) #axis=0 should be set for actually running this, the axis=2 is just for drift testing.
        HG_ev1_evoke_rescaled = HG_ev1_rescaled.average(method=lambda x: np.nanmean(x, axis=0))

        # get the evoke and evoke power rescaled amplitude
        HG_ev1_evoke_power = HG_ev1_power.average(method=lambda x: np.nanmean(x, axis=0)) #axis=0 should be set for actually running this, the axis=2 is just for drift testing.
        HG_ev1_evoke_power_rescaled = HG_ev1_power_rescaled.average(method=lambda x: np.nanmean(x, axis=0))

        # Save HG_ev1
        HG_ev1.save(f'{save_dir}/{sub}_{output_name_event}_HG_ev1-epo.fif', overwrite=True)
        HG_ev1.metadata.to_csv(f'{save_dir}/{sub}_{output_name_event}_HG_ev1_metadata.csv', index=False)
        
        HG_ev1_power.save(f'{save_dir}/{sub}_{output_name_event}_HG_ev1_power-epo.fif', overwrite=True)
        HG_ev1_power.metadata.to_csv(f'{save_dir}/{sub}_{output_name_event}_HG_ev1_power_metadata.csv', index=False)

        # Save HG_base (the shuffled version, on this event's channel set)
        HG_base_event.save(f'{save_dir}/{sub}_{output_name_event}_HG_base-epo.fif', overwrite=True)
        HG_base_power_event.save(f'{save_dir}/{sub}_{output_name_event}_HG_base_power-epo.fif', overwrite=True)

        # Save HG_ev1_rescaled
        HG_ev1_rescaled.save(f'{save_dir}/{sub}_{output_name_event}_HG_ev1_rescaled-epo.fif', overwrite=True)
        HG_ev1_rescaled.metadata.to_csv(f'{save_dir}/{sub}_{output_name_event}_HG_ev1_rescaled_metadata.csv', index=False)

        HG_ev1_power_rescaled.save(f'{save_dir}/{sub}_{output_name_event}_HG_ev1_power_rescaled-epo.fif', overwrite=True)
        HG_ev1_power_rescaled.metadata.to_csv(f'{save_dir}/{sub}_{output_name_event}_HG_ev1_power_rescaled_metadata.csv', index=False)

        # Save HG_ev1_evoke
        HG_ev1_evoke.save(f'{save_dir}/{sub}_{output_name_event}_HG_ev1_evoke-ave.fif', overwrite=True)
        HG_ev1_evoke_power.save(f'{save_dir}/{sub}_{output_name_event}_HG_ev1_evoke_power-ave.fif', overwrite=True)

        # Save HG_ev1_evoke_rescaled
        HG_ev1_evoke_rescaled.save(f'{save_dir}/{sub}_{output_name_event}_HG_ev1_evoke_rescaled-ave.fif', overwrite=True)
        HG_ev1_evoke_power_rescaled.save(f'{save_dir}/{sub}_{output_name_event}_HG_ev1_evoke_power_rescaled-ave.fif', overwrite=True)
        
        ###
        print(f"Shape of HG_ev1._data: {HG_ev1._data.shape}")
        print(f"Shape of HG_base_event._data: {HG_base_event._data.shape}")

        # time_perm_cluster compares these channel by channel, so they have to
        # line up (rescale picks the baseline in place, so this catches it if
        # a non-data channel ever slips through again).
        if HG_ev1.ch_names != HG_base_event.ch_names:
            raise RuntimeError(
                f"HG_ev1 ({len(HG_ev1.ch_names)} channels) and HG_base_event "
                f"({len(HG_base_event.ch_names)} channels) are on different channel sets; "
                "time_perm_cluster would compare mismatched channels.")

        # oh this changed and returns both the significant clusters matrix and the p values now
        mat, _ = time_perm_cluster(HG_ev1._data, HG_base_event._data, 0.05, n_jobs=6, ignore_adjacency=1, stat_func=stat_func) # TODO: rerun with HG_ev1_power and HG_base_power instead.

        #save channels with their indices 
        task_and_event = f"{task}_{event}"
        save_channels_to_file(channels_after_dropping_bad_channels, sub, task_and_event, save_dir)

        # save significant channels to a json
        save_sig_chans(f'{output_name_event}', mat, channels_after_dropping_bad_channels, sub, save_dir)
        
        # Assuming `mat` is your array and `save_dir` is the directory where you want to save it
        mat_save_path = os.path.join(save_dir, f'{output_name_event}_mat.npy')

        # Save the mat array
        np.save(mat_save_path, mat)
        
        # save dropped channels
        dropped_channels_filepath = os.path.join(save_dir, f'{sub}_{output_name_event}_dropped_channels.json')
        with open(dropped_channels_filepath, 'w') as f:
            json.dump({'dropped_channels': channels_to_drop}, f, indent=4)
        print(f"saved list of dropped channels to: {dropped_channels_filepath}")

# %%

def main(subjects=None, task='GlobalLocal', times=(-1, 1.5),
         within_base_times=(-1, 0), base_times_length=0.5, baseline_event="Stimulus", pad_length=3, LAB_root=None, channels=None, dec_factor=8, outlier_policy='drop', 
         outliers=10, threshold_percent=5.0, max_abs_z=30, passband=(70,150), filter_method='filterbank_hilbert', method='fir', fir_design='firwin',
         stat_func=ttest):
    """
    Main function to bandpass filter and compute time permutation cluster stats and task-significant electrodes for chosen subjects.
    """
    if subjects is None:
        subjects = ['D0057', 'D0059', 'D0063', 'D0065', 'D0069', 'D0071', 'D0077', 'D0090', 'D0094', 'D0100', 'D0102', 'D0103', 'D0107A', 'D0110', 'D0116', 
                    'D0117', 'D0121', 'D0130', 'D0133', 'D0134', 'D0137']

    for sub in subjects:
        bandpass_and_epoch_and_find_task_significant_electrodes(sub=sub, task=task, times=times,
                          within_base_times=within_base_times, base_times_length=base_times_length,
                          baseline_event=baseline_event, pad_length=pad_length, LAB_root=LAB_root, channels=channels,
                          dec_factor=dec_factor, outlier_policy=outlier_policy, outliers=outliers, 
                          threshold_percent=threshold_percent, max_abs_z=max_abs_z, passband=passband, filter_method=filter_method, 
                          method=method, fir_design=fir_design, stat_func=stat_func)
        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process subjects and plot bandpass-filtered data, compute time permutation cluster matrix of electrodes by time, and find task-significant electrodes.")
    parser.add_argument('--subjects', nargs='+', default=None, help='List of subjects to process. If not provided, all subjects will be processed.')
    parser.add_argument('--task', type=str, default='GlobalLocal', help='Task to process. Default is GlobalLocal.')
    parser.add_argument('--times', type=float, nargs=2, default=(-1, 1.5), help='Time window for event processing. Default is (-1, 1.5).')
    parser.add_argument('--within_base_times', type=float, nargs=2, default=(-1, 0), help='Time window for baseline processing. Default is (-1, 0).')
    parser.add_argument('--baseline_event', type=str, default='Stimulus', help='Event to use for baseline. Default is Stimulus.')
    parser.add_argument('--base_times_length', type=float, default=0.5, help='Length of the time intervals to randomly select within `within_base_times`. Default is 0.5.')
    parser.add_argument('--pad_length', type=float, default=3, help='Length to pad each time interval. Will be removed later. Default is 0.5.')
    parser.add_argument('--LAB_root', type=str, default=None, help='Root directory for the lab. Will be determined based on OS if not provided. Default is None.')
    parser.add_argument('--channels', type=str, default=None, help='Channels to plot and get stats for. Default is all channels.')
    parser.add_argument('--dec_factor', type=int, default=8, help='Decimation factor. Default is 8.')
    parser.add_argument('--outlier_policy', type=str, default='drop', help='How to handle outlier values. Options: drop, nan, drop_and_nan, drop_and_impute, ignore.')
    parser.add_argument('--outliers', type=int, default=10, help='How many standard deviations above the trial mean for a timepoint to be considered an outlier. Default is 10.')
    parser.add_argument('--max_abs_z', type=lambda v: None if str(v).lower() == 'none' else float(v), default=30, help='Reject a (trial, channel) trace whose baseline-normalized power exceeds this anywhere. Real HG z-scores top out near 15-20; measured artifacts were 43-24000. Try 30 as default for now. But can set to None to not drop any trials.')
    parser.add_argument('--threshold_percent', type=float, default=5.0, help='Channels with a greater percent of outlier trials than this threshold will be removed from further analyses, if using drop, drop_and_nan, and drop_and_impute outlier policies.')
    parser.add_argument('--passband', type=float, nargs=2, default=(70,150), help='Frequency range for the frequency band of interest. Default is (70, 150).')
    parser.add_argument('--filter_method', type=str, default='filterbank_hilbert', help='Which filtering method to use for extracting your chosen frequency range. Currently can use either filterbank_hilbert or bandpass.')
    parser.add_argument('--method', type=str, default='fir', help='The bandpass method to use if you use bandpass as the filter_method. Default is to use fir but can use iir.')
    parser.add_argument('--fir_design', type=str, default='firwin', help='The fir design for the bandpass filter if you use bandpass as the filter_method. Default is firwin but can use firwin2 or other things.')
    parser.add_argument('--stat_func', default=ttest, help="Statistical function to use for significance testing. Default is ieeg.calc.fast.ttest, which is time_perm_cluster's own default: the same Welch t statistic as scipy's ttest_ind(equal_var=False), but vectorized over NaNs instead of falling into scipy's per-slice nan_policy='omit' loop.")
    args=parser.parse_args()

    # Convert the string 'None' to a proper None object
    channels_arg = None if args.channels == 'None' else args.channels
    
    print("--------- PARSED ARGUMENTS ---------")
    print(f"args.subjects: {args.subjects}")
    print(f"args.task: {args.task}")
    print(f"args.times: {args.times}")
    print(f"args.within_base_times: {args.within_base_times}")
    print(f"args.baseline_event: {args.baseline_event}")
    print(f"args.base_times_length: {args.base_times_length}")
    print(f"args.pad_length: {args.pad_length}")
    print(f"args.LAB_root: {args.LAB_root}")
    print(f"args.channels: {args.channels}")
    print(f"args.dec_factor: {args.dec_factor}")
    print(f"args.outlier_policy: {args.outlier_policy}")
    print(f"args.outliers: {args.outliers}")
    print(f"args.max_abs_z: {args.max_abs_z}")
    print(f"args.threshold_percent: {args.threshold_percent}")
    print(f"args.passband: {args.passband}")
    print(f"args.filter_method: {args.filter_method}")
    print(f"args.method: {args.method}")
    print(f"args.fir_design: {args.fir_design}")
    
    main(subjects=args.subjects, 
        task=args.task, 
        times=args.times, 
        within_base_times=args.within_base_times, 
        base_times_length=args.base_times_length, 
        pad_length=args.pad_length, 
        LAB_root=args.LAB_root, 
        channels=channels_arg, 
        dec_factor=args.dec_factor, 
        outlier_policy=args.outlier_policy,
        outliers=args.outliers, 
        threshold_percent=args.threshold_percent,
        max_abs_z=args.max_abs_z,
        passband=args.passband,
        filter_method=args.filter_method,
        method=args.method,
        fir_design=args.fir_design,
        stat_func=args.stat_func)
