import os
import json
import numpy as np
import mne
import pandas as pd
from itertools import combinations
from mne_connectivity import spectral_connectivity_epochs
import argparse

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

def compute_coherence_epochs(epochs, chs, freqs, n_cycles, method, mode, cwt_n_cycles, fmin, fmax, n_jobs, batch_size=1, verbose=True):
    
    present = set(epochs.ch_names)
    chs_present = [ch for ch in chs if ch in present]
    missing = [ch for ch in chs if ch not in present]
    if missing:
        print(f"Warning: {len(missing)} ROI channels missing and will be skipped: {missing[:10]}{'...' if len(missing)>10 else ''}")
    if len(chs_present) < 2:
        print("Not enough ROI channels present to compute pairwise coherence — skipping.")
        return {}, np.array([])

    epochs_roi = epochs.copy().pick_channels(chs_present)
    idx_pairs = list(combinations(range(len(chs_present)), 2))
    if len(idx_pairs) == 0:
        print("No channel pairs (idx_pairs empty).")
        return {}, np.array([])

    u_inds, v_inds = zip(*idx_pairs)

    n_epochs = len(epochs_roi)
    if n_epochs < 1:
        print("No epochs to process in this window.")
        return {}, np.array([])

    # We will collect data as (n_pairs, n_epochs, n_freqs)
    # To know n_freqs, perform a single call on the first epoch (or batch) to inspect shape
    # Prepare a helper function to call spectral_connectivity_epochs with safe kwargs
    def _call_spectral(ep):
        kwargs = dict(
            data=ep,
            names=None,
            method=method,
            indices=(list(u_inds), list(v_inds)),
            sfreq=epochs_roi.info['sfreq'],
            fmin=fmin,
            fmax=fmax,
            faverage=False,
            # Avoid nested parallelism for per-epoch calls:
            n_jobs=1
        )
        # mode-specific args
        if mode in ('cwt', 'cwt_morlet', 'morlet'):
            kwargs['cwt_freqs'] = freqs
            kwargs['cwt_n_cycles'] = list(cwt_n_cycles) if cwt_n_cycles is not None else list(n_cycles)
        # else: allow multitaper defaults (could pass mt_bandwidth if desired)

        return spectral_connectivity_epochs(**kwargs)

    # quick probe
    try:
        probe_con = _call_spectral(epochs_roi[0:1])
    except Exception as e:
        print("spectral_connectivity_epochs (probe) failed:", e)
        return {}, np.array([])

    # get freq vector from probe
    freqs_out = getattr(probe_con, 'freqs', None)
    if freqs_out is None:
        freqs_out = freqs
    # get per-call data shape and infer n_freqs
    probe_data = np.array(probe_con.get_data())
    if probe_data.ndim == 3:
        # (n_pairs, n_epochs_in_call, n_freqs) -> for single-epoch call n_epochs_in_call==1
        n_freqs = probe_data.shape[2]
    elif probe_data.ndim == 2:
        # (n_pairs, n_freqs)
        n_freqs = probe_data.shape[1]
    else:
        # unexpected
        n_freqs = probe_data.shape[-1]

    # allocate
    n_pairs = len(idx_pairs)
    coh_all = np.zeros((n_pairs, n_epochs, n_freqs), dtype=float)

    # Optionally process in small batches to reduce overhead. Default batch_size==1 => strict per-epoch
    ei = 0
    while ei < n_epochs:
        # batch indices [ei, ei+batch_size)
        bi = min(ei + batch_size, n_epochs)
        if batch_size == 1:
            # single epoch call
            ep = epochs_roi[ei:ei+1]
            try:
                con = _call_spectral(ep)
                data = np.array(con.get_data())
            except Exception as e:
                print(f"  spectral_connectivity_epochs failed for epoch {ei}: {e}")
                # fill with nan
                data = np.full((n_pairs, n_freqs), np.nan, dtype=float)
            # normalize data to shape (n_pairs, n_freqs)
            if data.ndim == 3:
                # (n_pairs, 1, n_freqs)
                data = data[:, 0, :]
            coh_all[:, ei, :] = data
            if verbose and (ei % 10 == 0 or ei == n_epochs-1):
                print(f"    processed epoch {ei+1}/{n_epochs}")
            ei += 1
        else:
            # batch processing
            ep_batch = epochs_roi[ei:bi]
            try:
                con = _call_spectral(ep_batch)
                data = np.array(con.get_data())  # possibly (n_pairs, n_epochs_in_batch, n_freqs) or (n_pairs, n_freqs)
            except Exception as e:
                print(f"  spectral_connectivity_epochs failed for epoch batch {ei}-{bi-1}: {e}")
                data = np.full((n_pairs, bi-ei, n_freqs), np.nan, dtype=float)

            # normalize
            if data.ndim == 2:
                # (n_pairs, n_freqs) -> likely averaged over batch; replicate to each epoch in batch
                data = np.tile(data[:, np.newaxis, :], (1, bi-ei, 1))
            elif data.ndim == 3:
                # (n_pairs, n_epochs_in_batch, n_freqs)
                pass
            else:
                # unexpected
                data = np.reshape(data, (n_pairs, bi-ei, n_freqs))

            coh_all[:, ei:bi, :] = data
            if verbose:
                print(f"    processed epochs {ei+1}-{bi} (batch_size={batch_size})")
            ei = bi

    # Build dict mapping channel pair to (n_epochs, n_freqs)
    coh_dict = {}
    for p_idx, (i, j) in enumerate(idx_pairs):
        ch1, ch2 = chs_present[i], chs_present[j]
        # transpose to (n_epochs, n_freqs) to match previous expectation
        coh_dict[(ch1, ch2)] = coh_all[p_idx].copy()

    return coh_dict, freqs_out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute ROI per-trial coherence using spectral_connectivity_epochs')
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
    parser.add_argument('--output_dir', type=str, default='.')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--method', type=str, default='coh', help='connectivity method, e.g., coh, plv, pli')
    parser.add_argument('--mode', type=str, default='cwt', help="time-frequency mode: 'cwt' (Morlet) or 'multitaper'")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    fig_dir = os.path.join(args.output_dir, 'coherence_figs')
    if args.plot:
        os.makedirs(fig_dir, exist_ok=True)

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
    # Use n_cycles per frequency (can be fractional) — here I reuse your previous heuristic
    n_cycles = freqs / 2.0

    all_records = []

    for subj in args.subjects:
        epochs = epoch_dicts.get(subj)
        if epochs is None:
            print(f"Skipping {subj}: no epochs loaded")
            continue
        print(f"Processing subject {subj}...")

        chs = find_roi_names(args.part, subj, args.roi_json)

        for (w0, w1) in windows:
            print(f"  Window {w0} to {w1} s — cropping epochs and computing per-trial coherence...")
            try:
                epochs_win = epochs.copy().crop(tmin=w0, tmax=w1)
            except Exception as e:
                print(f"    Error cropping epochs for {subj}, window {w0}-{w1}: {e}")
                continue

            # require at least 1 epoch; recommend >=2 for stability
            if len(epochs_win) < 1:
                print(f"    Window {w0}-{w1}: no epochs after cropping — skipping.")
                continue

            coh_dict, freqs_out = compute_coherence_epochs(
                epochs_win, chs, freqs, n_cycles,
                method=args.method, mode=args.mode, cwt_n_cycles=n_cycles,
                fmin=args.fmin, fmax=args.fmax, n_jobs=args.n_jobs
            )
            
            print("n_epochs_window:", len(epochs_win))
            # pick an arbitrary pair
            some_pair = next(iter(coh_dict.keys()))
            print("example pair shape (should be (n_epochs, n_freqs)):", coh_dict[some_pair].shape)


            # Save per-pair, per-window records
            for (ch1, ch2), coh_trials in coh_dict.items():
                # coh_trials expected shape: (n_epochs, n_freqs)
                try:
                    mean_val = np.nanmean(coh_trials)  # scalar summary across trials & freqs
                except Exception:
                    mean_val = float('nan')

                rec = {
                    'subject': subj,
                    'condition': args.condition,
                    'part': args.part[0],
                    'ch1': ch1,
                    'ch2': ch2,
                    'window_start': w0,
                    'window_end': w1,
                    'n_epochs_in_window': coh_trials.shape[0] if hasattr(coh_trials, 'shape') else None,
                    'coh_mean': mean_val,
                    'coh_trials': coh_trials,   # ndarray (n_epochs, n_freqs)
                    'freqs': freqs_out
                }
                all_records.append(rec)

    # Save summary CSV and pickle
    df_sum = pd.DataFrame(all_records)
    csv_path = os.path.join(args.output_dir, f'coh_timewindow/coherence_epochs_summary_{str(args.subjects)[2:-2]}_{args.part[0]}_{args.condition}.csv')
    if not df_sum.empty:
        df_sum[['subject', 'condition', 'part', 'ch1', 'ch2', 'window_start', 'window_end', 'n_epochs_in_window', 'coh_mean']].to_csv(csv_path, index=False)
    else:
        # write empty csv with headers
        pd.DataFrame(columns=['subject','condition','part','ch1','ch2','window_start','window_end','n_epochs_in_window','coh_mean']).to_csv(csv_path, index=False)
    pkl_path = os.path.join(args.output_dir, f'coh_timewindow/coherence_epochs_full_{str(args.subjects)[2:-2]}_{args.part[0]}_{args.condition}.pkl')
    pd.to_pickle(df_sum, pkl_path)

    print(f"Saved coherence summary to {csv_path}")
    print(f"Saved full per-trial data to {pkl_path}")

    print("Done.")
