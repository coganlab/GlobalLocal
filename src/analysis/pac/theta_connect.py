#!/usr/bin/env python3
import os
import json
import numpy as np
import mne
import pandas as pd
from itertools import combinations
from mne_connectivity import spectral_connectivity_epochs
import argparse
import copy
import warnings

def find_roi_names(part, subj, roi_json):
    rois_dict = {
    'dlpfc': ["G_front_middle", "G_front_sup", "S_front_inf", "S_front_middle", "S_front_sup"],
    'acc': ["G_and_S_cingul-Ant", "G_and_S_cingul-Mid-Ant"],
    'parietal': ["G_parietal_sup", "S_intrapariet_and_P_trans", "G_pariet_inf-Angular", "G_pariet_inf-Supramar"],
    'lpfc': ["G_front_inf-Opercular", "G_front_inf-Orbital", "G_front_inf-Triangul", "G_front_middle", "G_front_sup", "Lat_Fis-ant-Horizont", "Lat_Fis-ant-Vertical", "S_circular_insula_ant", "S_circular_insula_sup", "S_front_inf", "S_front_middle", "S_front_sup"],
    'v1': ["G_oc-temp_med-Lingual", "S_calcarine", "G_cuneus"],
    'occ': ["G_cuneus", "G_and_S_occipital_inf", "G_occipital_middle", "G_occipital_sup", "G_oc-temp_lat-fusifor", "G_oc-temp_med-Lingual", "Pole_occipital", "S_calcarine", "S_oc_middle_and_Lunatus", "S_oc_sup_and_transversal", "S_occipital_ant"]
    }
    roi_list = rois_dict[part[0]]
    print(f"Using ROIs: {roi_list}")
    with open(roi_json, 'r') as f:
        roi_data = json.load(f)

    chs = []
    filt = roi_data.get(subj, {}).get('filtROI_dict', {})
    for roi in roi_list:
        for key, val in filt.items():
            if roi in key:
                chs.extend(val)
    chs = list(dict.fromkeys(chs))
    print(f"Found {len(chs)} ROI channels for {subj}")
    return chs

def load_epochs(subjects, bids_root, condition, deriv_dir='freqFilt/figs', epoch_suffix='full-epo', nch_example=3):
    epoch_dicts = {}
    summary_records = []

    for subj in subjects:
        print(f"Subject {subj}: loading epochs for condition {condition}...")
        fname = f"{subj}_{condition}_{epoch_suffix}.fif"
        eph_path = os.path.join(bids_root, 'derivatives', deriv_dir, subj, fname)
        if os.path.isfile(eph_path):
            try:
                epochs = mne.read_epochs(eph_path, preload=True)
                epoch_dicts[subj] = epochs
                ch_list = epochs.ch_names[:nch_example]
                ch_example_str = ','.join(ch_list)
                summary_records.append({
                    'subject': subj,
                    'file': eph_path,
                    'n_epochs': len(epochs),
                    'sfreq': epochs.info.get('sfreq'),
                    'n_channels': len(epochs.ch_names),
                    'tmin': epochs.tmin,
                    'tmax': epochs.tmax,
                    'channel_examples': ch_example_str
                })
                print(f"    Loaded {len(epochs)} epochs ({ch_example_str}...)")
            except Exception as e:
                print(f"    Error loading epochs for {subj}: {e}")
        else:
            print(f"  Epoch file not found: {eph_path}")

    df_summary = pd.DataFrame.from_records(summary_records)
    return epoch_dicts, df_summary

def make_windows(start, end, win_len):
    eps = 1e-9
    starts = np.arange(start, end - win_len + eps, win_len)
    windows = [(float(np.round(s, 6)), float(np.round(s + win_len, 6))) for s in starts]
    return windows

def _bh_fdr(pvals, alpha=0.05):
    """Benjamini-Hochberg FDR. pvals: 1D array. Returns boolean mask of rejected hypotheses."""
    p = np.asarray(pvals)
    n = p.size
    if n == 0:
        return np.array([], dtype=bool)
    sort_idx = np.argsort(p)
    sorted_p = p[sort_idx]
    thresh = (np.arange(1, n+1) / n) * alpha
    below = sorted_p <= thresh
    if not np.any(below):
        return np.zeros(n, dtype=bool)
    max_idx = np.where(below)[0].max()
    cutoff_p = sorted_p[max_idx]
    rej = p <= cutoff_p
    return rej

def compute_alltrial_coherence_and_permutation(epochs, chs, freqs, n_cycles, method, mode, cwt_n_cycles, fmin, fmax, n_jobs, n_perm=200, alpha=0.05, verbose=True):
    """
    Compute all-trial coherence for all channel pairs in 'chs' (present in epochs),
    then for each pair perform permutation by shuffling trial indices of the second channel
    to create a null distribution. Return dictionary with:
      - con_true_dict[(ch1,ch2)] = array(n_freqs,)
      - pvals_dict[(ch1,ch2)] = array(n_freqs,)
      - sig_mask_dict[(ch1,ch2)] = boolean array(n_freqs,) after FDR across all tests
    """
    present = set(epochs.ch_names)
    chs_present = [ch for ch in chs if ch in present]
    missing = [ch for ch in chs if ch not in present]
    if missing:
        print(f"Warning: {len(missing)} ROI channels missing and will be skipped: {missing[:10]}{'...' if len(missing)>10 else ''}")
    if len(chs_present) < 2:
        print("Not enough ROI channels present to compute pairwise coherence — skipping.")
        return {}, {}, {}

    epochs_roi = epochs.copy().pick_channels(chs_present)
    n_epochs = len(epochs_roi)
    if n_epochs < 2:
        print("Not enough epochs (need >=2) to compute all-trial coherence — skipping.")
        return {}, {}, {}

    # build index pairs
    idx_pairs = list(combinations(range(len(chs_present)), 2))
    u_inds, v_inds = zip(*idx_pairs)
    n_pairs = len(idx_pairs)

    # Prepare kwargs for spectral_connectivity_epochs
    base_kwargs = dict(
        data=epochs_roi,
        names=None,
        method=method,
        indices=(list(u_inds), list(v_inds)),
        sfreq=epochs_roi.info['sfreq'],
        fmin=fmin,
        fmax=fmax,
        faverage=False,
        n_jobs=1  # avoid nested parallelism here; whole script can be parallelized externally
    )
    if mode in ('cwt', 'cwt_morlet', 'morlet'):
        base_kwargs['mode'] = 'cwt_morlet'
        base_kwargs['cwt_freqs'] = freqs
        # choose cycles: prefer explicit cwt_n_cycles if given; else n_cycles
        base_kwargs['cwt_n_cycles'] = list(cwt_n_cycles) if cwt_n_cycles is not None else list(n_cycles)
    else:
        # use multitaper or default
        base_kwargs['mode'] = 'multitaper'

    # Compute true all-trial coherence for all pairs at once
    if verbose:
        print("Computing true all-trial connectivity for all pairs (single call)...")
    try:
        con_true = spectral_connectivity_epochs(**base_kwargs)
    except Exception as e:
        warnings.warn(f"spectral_connectivity_epochs failed for full-call: {e}")
        return {}, {}, {}

    data_true = np.array(con_true.get_data())
    # handle shapes: could be (n_pairs, n_freqs) or (n_pairs, n_freqs, n_times)
    if data_true.ndim == 3:
        # collapse time dimension by averaging (user is doing windows anyway)
        # Alternatively could keep time dimension; here we average across time inside the cropped window.
        data_true = np.mean(data_true, axis=-1)
    # now data_true shape is (n_pairs, n_freqs)
    n_freqs = data_true.shape[1]

    # containers
    con_true_dict = {}
    pvals_dict = {}
    all_pvals_flat = []
    keys_list = []

    # store true for each pair
    for p_idx, (i, j) in enumerate(idx_pairs):
        ch1, ch2 = chs_present[i], chs_present[j]
        con_true_dict[(ch1, ch2)] = data_true[p_idx].copy()
        keys_list.append((ch1, ch2))

    # Permutation test: for each pair, shuffle trial indices of channel j
    # We'll compute null distribution per pair (n_perm × n_freqs)
    rng = np.random.default_rng(42)
    print(f"Beginning permutations: n_pairs={n_pairs}, n_perm={n_perm} (this may take time)...")
    for p_idx, (i, j) in enumerate(idx_pairs):
        ch1, ch2 = chs_present[i], chs_present[j]
        if verbose:
            print(f"  Pair {p_idx+1}/{n_pairs}: {ch1} - {ch2}")

        # allocate perms
        perms = np.zeros((n_perm, n_freqs), dtype=float)

        # get original numpy data (n_epochs, n_channels, n_times)
        data = epochs_roi.get_data()  # shape (n_epochs, n_channels, n_times)
        # index of channel j in epochs_roi
        chj_idx = j

        for pi in range(n_perm):
            perm_idx = rng.permutation(n_epochs)
            # build permuted data: copy to avoid modifying original
            data_perm = data.copy()
            # permute channel j across epochs
            data_perm[:, chj_idx, :] = data[perm_idx, chj_idx, :]

            # create EpochsArray with same info
            try:
                epochs_perm = mne.EpochsArray(data_perm, info=epochs_roi.info, tmin=epochs_roi.tmin, verbose=False)
            except Exception as e:
                # fallback: try copying epochs_roi and replacing _data (hacky but sometimes works)
                epochs_perm = copy.deepcopy(epochs_roi)
                try:
                    epochs_perm._data = data_perm
                except Exception as e2:
                    raise RuntimeError(f"Failed to create permuted epochs: {e} / {e2}")

            # call spectral_connectivity_epochs for this pair only (indices=( [i], [j] ))
            kwargs = dict(
                data=epochs_perm,
                names=None,
                method=method,
                indices=([i], [j]),
                sfreq=epochs_perm.info['sfreq'],
                fmin=fmin,
                fmax=fmax,
                faverage=False,
                n_jobs=1
            )
            if mode in ('cwt', 'cwt_morlet', 'morlet'):
                kwargs['mode'] = 'cwt_morlet'
                kwargs['cwt_freqs'] = freqs
                kwargs['cwt_n_cycles'] = list(cwt_n_cycles) if cwt_n_cycles is not None else list(n_cycles)
            else:
                kwargs['mode'] = 'multitaper'
            try:
                con_p = spectral_connectivity_epochs(**kwargs)
                data_p = np.array(con_p.get_data())
                if data_p.ndim == 3:
                    data_p = np.mean(data_p, axis=-1)
                # data_p shape: (1, n_freqs)
                perms[pi, :] = data_p[0, :]
            except Exception as e:
                # on failure, fill with nan
                warnings.warn(f"    permutation {pi} failed for pair {ch1}-{ch2}: {e}")
                perms[pi, :] = np.nan

            if verbose and (pi % 50 == 0 and pi > 0):
                print(f"    done {pi}/{n_perm} perms")

        # compute p-values per frequency: right-tailed test (true greater than null)
        true_vals = con_true_dict[(ch1, ch2)]
        # handle NaNs in perms: count only valid
        pvals = np.empty(n_freqs, dtype=float)
        for fi in range(n_freqs):
            perm_vals = perms[:, fi]
            valid = np.isfinite(perm_vals)
            n_valid = int(np.sum(valid))
            if n_valid == 0:
                pvals[fi] = np.nan
                continue
            # count how many perm >= true
            cnt = np.sum(perm_vals[valid] >= true_vals[fi])
            pvals[fi] = (cnt + 1) / (n_valid + 1)  # add-one correction
        pvals_dict[(ch1, ch2)] = pvals
        all_pvals_flat.extend([p for p in pvals if (p is not None and not np.isnan(p))])
    # Multiple comparisons: FDR across all pair×freq tests (flattened)
    all_pvals_flat = np.array(all_pvals_flat)
    if all_pvals_flat.size == 0:
        print("No valid p-values computed.")
        sig_mask_dict = {k: np.zeros_like(v, dtype=bool) for k, v in con_true_dict.items()}
        return con_true_dict, pvals_dict, sig_mask_dict

    rej_flat = _bh_fdr(all_pvals_flat, alpha=alpha)
    # now we need to map rej_flat back to each (pair,freq)
    # We'll iterate in same order as we filled all_pvals_flat
    sig_mask_dict = {}
    idx_flat = 0
    for k in keys_list:
        pvals = pvals_dict[k]
        mask = np.zeros_like(pvals, dtype=bool)
        for fi in range(len(pvals)):
            p = pvals[fi]
            if np.isnan(p):
                mask[fi] = False
            else:
                if idx_flat < len(rej_flat):
                    mask[fi] = bool(rej_flat[idx_flat])
                    idx_flat += 1
                else:
                    mask[fi] = False
        sig_mask_dict[k] = mask

    return con_true_dict, pvals_dict, sig_mask_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute ROI all-trial coherence using spectral_connectivity_epochs + permutation test')
    parser.add_argument('--bids_root', type=str, required=True)
    parser.add_argument('--subjects', nargs='+', required=True)
    parser.add_argument('--roi_json', type=str, required=True)
    parser.add_argument('--part', nargs='+', required=True)
    parser.add_argument('--condition', type=str, required=True, help='e.g., stimulus_c')
    parser.add_argument('--tmin', type=float, required=True)
    parser.add_argument('--tmax', type=float, required=True)
    parser.add_argument('--stepsize', type=float, required=True, help='window length (s), contiguous non-overlapping windows')
    parser.add_argument('--fmin', type=float, default=3.0)
    parser.add_argument('--fmax', type=float, default=8.0)
    parser.add_argument('--n_freqs', type=int, default=25)
    parser.add_argument('--n_jobs', type=int, default=8)
    parser.add_argument('--n_perm', type=int, default=200, help='number of permutations per pair')
    parser.add_argument('--alpha', type=float, default=0.05, help='FDR alpha')
    parser.add_argument('--output_dir', type=str, default='.')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--method', type=str, default='coh', help='connectivity method, e.g., coh, plv, pli, imcoh, wpli')
    parser.add_argument('--mode', type=str, default='cwt', help="time-frequency mode: 'cwt' (Morlet) or 'multitaper'")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    out_dir = os.path.join(args.output_dir, 'coh_timewindow')
    os.makedirs(out_dir, exist_ok=True)

    # parse time_range
    try:
        tstart= args.tmin
        tend = args.tmax
        if tstart >= tend:
            raise ValueError("time_range start must be < end")
    except Exception as e:
        raise SystemExit(f"Failed to parse --time_range: {e}")

    windows = make_windows(tstart, tend, args.stepsize)
    if len(windows) == 0:
        raise SystemExit("No windows generated — check time_range and stepsize.")

    print(f"Generated {len(windows)} windows: {windows}")

    # Load epochs
    epoch_dicts, df_epochs = load_epochs(args.subjects, args.bids_root,
                                         condition=args.condition, deriv_dir='freqFilt/figs',
                                         epoch_suffix='full-epo')
    df_epochs.to_csv(os.path.join(args.output_dir, 'epoch_summary.csv'), index=False)
    print('Epoch summary saved.')

    # Frequency vector and cycle settings for cwt (Morlet)
    freqs = np.linspace(args.fmin, args.fmax, args.n_freqs)
    # If you prefer higher freq resolution set n_cycles = freqs (trade time resolution)
    n_cycles = freqs  # changed from freqs/2 to freqs for better frequency resolution

    all_records = []
    all_sig_pairs_records = []

    for subj in args.subjects:
        epochs = epoch_dicts.get(subj)
        if epochs is None:
            print(f"Skipping {subj}: no epochs loaded")
            continue
        print(f"Processing subject {subj}...")

        chs = find_roi_names(args.part, subj, args.roi_json)

        for (w0, w1) in windows:
            print(f"  Window {w0} to {w1} s — cropping epochs and computing all-trial coherence + permutations...")
            try:
                epochs_win = epochs.copy().crop(tmin=w0, tmax=w1)
            except Exception as e:
                print(f"    Error cropping epochs for {subj}, window {w0}-{w1}: {e}")
                continue

            if len(epochs_win) < 2:
                print(f"    Window {w0}-{w1}: less than 2 epochs after cropping — skipping.")
                continue

            con_true_dict, pvals_dict, sig_mask_dict = compute_alltrial_coherence_and_permutation(
                epochs_win, chs, freqs, n_cycles,
                method=args.method, mode=args.mode, cwt_n_cycles=n_cycles,
                fmin=args.fmin, fmax=args.fmax, n_jobs=args.n_jobs,
                n_perm=args.n_perm, alpha=args.alpha, verbose=True
            )

            # collect results and save
            for (ch1, ch2), true_vals in con_true_dict.items():
                pvals = pvals_dict.get((ch1, ch2), np.array([np.nan]*len(freqs)))
                sig_mask = sig_mask_dict.get((ch1, ch2), np.zeros(len(freqs), dtype=bool))
                mean_val = np.nanmean(true_vals)
                rec = {
                    'subject': subj,
                    'condition': args.condition,
                    'part': args.part[0],
                    'ch1': ch1,
                    'ch2': ch2,
                    'window_start': w0,
                    'window_end': w1,
                    'n_epochs_in_window': len(epochs_win),
                    'coh_mean': mean_val,
                    'freqs': freqs,
                    'coh_true': true_vals,
                    'pvals': pvals,
                    'sig_mask': sig_mask
                }
                all_records.append(rec)
                # if any frequency significant after FDR, record the pair as significant
                if np.any(sig_mask):
                    sig_freqs = freqs[sig_mask]
                    all_sig_pairs_records.append({
                        'subject': subj,
                        'part': args.part[0],
                        'condition': args.condition,
                        'window_start': w0,
                        'window_end': w1,
                        'ch1': ch1,
                        'ch2': ch2,
                        'n_epochs': len(epochs_win),
                        'sig_freqs': ",".join([f"{f:.2f}" for f in sig_freqs])
                    })
                    print(f"    SIGNIFICANT pair: {ch1}-{ch2} at freqs {sig_freqs}")

    # Save summary CSV and pickle
    if len(all_records) > 0:
        df_sum = pd.DataFrame([{
            'subject': r['subject'],
            'condition': r['condition'],
            'part': r['part'],
            'ch1': r['ch1'],
            'ch2': r['ch2'],
            'window_start': r['window_start'],
            'window_end': r['window_end'],
            'n_epochs_in_window': r['n_epochs_in_window'],
            'coh_mean': r['coh_mean']
        } for r in all_records])
        csv_path = os.path.join(out_dir, f'coherence_epochs_summary_{str(args.subjects)[2:-2]}_{args.part[0]}_{args.condition}.csv')
        df_sum.to_csv(csv_path, index=False)
        pkl_path = os.path.join(out_dir, f'coherence_epochs_full_{str(args.subjects)[2:-2]}_{args.part[0]}_{args.condition}.pkl')
        pd.to_pickle(all_records, pkl_path)
        print(f"Saved coherence summary to {csv_path}")
        print(f"Saved full per-trial data to {pkl_path}")
    else:
        print("No results to save (no records).")

    # Save significant pairs
    if len(all_sig_pairs_records) > 0:
        df_sig = pd.DataFrame(all_sig_pairs_records)
        sig_csv = os.path.join(out_dir, f'significant_pairs_{str(args.subjects)[2:-2]}_{args.part[0]}_{args.condition}.csv')
        df_sig.to_csv(sig_csv, index=False)
        print(f"Saved significant pairs to {sig_csv}")
    else:
        print("No significant pairs found.")

    print("Done.")
