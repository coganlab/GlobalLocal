import os
import json
import copy
import warnings
import argparse
from itertools import combinations

import numpy as np
import pandas as pd
import mne



def find_roi_names(part, subj, roi_json, epochs_ch_names):
    """Return bipolar channel names that overlap with ROI single-pole labels."""
    rois_dict = {
        'dlpfc': ["G_front_middle", "G_front_sup", "S_front_inf", "S_front_middle", "S_front_sup"],
        'acc': ["G_and_S_cingul-Ant", "G_and_S_cingul-Mid-Ant"],
        'parietal': ["G_pariet_sup", "S_intrapariet_and_P_trans", "G_pariet_inf-Angular", "G_pariet_inf-Supramar"],
        'lpfc': ["G_front_inf-Opercular", "G_front_inf-Orbital", "G_front_inf-Triangul", "G_front_middle", "G_front_sup", "Lat_Fis-ant-Horizont", "Lat_Fis-ant-Vertical", "S_circular_insula_ant", "S_circular_insula_sup", "S_front_inf", "S_front_middle", "S_front_sup"],
        'v1': ["G_oc-temp_med-Lingual", "S_calcarine", "G_cuneus"],
        'occ': ["G_cuneus", "G_and_S_occipital_inf", "G_occipital_middle", "G_occipital_sup", "G_oc-temp_lat-fusifor", "G_oc-temp_med-Lingual", "Pole_occipital", "S_calcarine", "S_oc_middle_and_Lunatus", "S_oc_sup_and_transversal", "S_occipital_ant"]
    }

    roi_list = rois_dict[part[0]]
    print(f"Using ROIs: {roi_list}")

    with open(roi_json, 'r') as f:
        roi_data = json.load(f)

    chs_single = []
    filt = roi_data.get(subj, {}).get('filtROI_dict', {})
    for roi in roi_list:
        for key, val in filt.items():
            if roi in key:
                chs_single.extend(val)

    chs_single = list(dict.fromkeys(chs_single))
    print(f"Found {len(chs_single)} single-pole ROI channels for {subj}")

    chs = [
        ch for ch in epochs_ch_names
        if '-' in ch and (ch.split('-')[0] in chs_single or ch.split('-')[1] in chs_single)
    ]
    chs = list(dict.fromkeys(chs))
    print(f"Filtered to {len(chs)} bipolar ROI channels for {subj}")
    print(chs[:3])
    return chs



def load_epochs(subjects, bids_root, condition, deriv_dir='freqFilt/figs', epoch_suffix='sig_HG_ev-epo', nch_example=3):
    """Load pre-epoched FIF files saved by your first pipeline."""
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
    """Non-overlapping contiguous windows."""
    eps = 1e-9
    starts = np.arange(start, end - win_len + eps, win_len)
    windows = [(float(np.round(s, 6)), float(np.round(s + win_len, 6))) for s in starts]
    return windows


def _bh_fdr(pvals, alpha=0.05):
    """Benjamini-Hochberg FDR correction (manual implementation)."""
    p = np.asarray(pvals)
    n = p.size
    if n == 0:
        return np.array([], dtype=bool)
    sort_idx = np.argsort(p)
    sorted_p = p[sort_idx]
    thresh = (np.arange(1, n + 1) / n) * alpha
    below = sorted_p <= thresh
    if not np.any(below):
        return np.zeros(n, dtype=bool)
    max_idx = np.where(below)[0].max()
    cutoff_p = sorted_p[max_idx]
    rej = p <= cutoff_p
    return rej


def sanitize_filename(name: str) -> str:
    return ''.join('_' if c in '<>:"/\\|?*' else c for c in name)



def compute_pearson_correlation_for_envelope_epochs(
    epochs,
    chs,
):
    """Compute Pearson correlation directly from gamma envelope epochs.

    The input epochs are expected to already contain gamma envelope values.

    Returns
    -------
    con_mat : ndarray, shape (n_epochs, n_channels, n_channels)
        Per-epoch correlation matrices.
    mean_mat : ndarray, shape (n_channels, n_channels)
        Mean correlation across epochs.
    chs_present : list[str]
        Channels actually used.
    """
    present = set(epochs.ch_names)
    chs_present = [ch for ch in chs if ch in present]
    missing = [ch for ch in chs if ch not in present]

    if missing:
        print(f"Warning: {len(missing)} ROI channels missing and will be skipped: {missing[:10]}{'...' if len(missing) > 10 else ''}")

    if len(chs_present) < 2:
        print("Not enough ROI channels present to compute Pearson correlation — skipping.")
        return None, None, chs_present

    epochs_roi = epochs.copy().pick(chs_present).load_data()
    data = epochs_roi.get_data()

    if data.ndim != 3:
        raise RuntimeError(
            f"Expected epochs data of shape (n_epochs, n_channels, n_times), got {data.shape}."
        )

    n_epochs, n_channels, _ = data.shape
    con_mat = np.full((n_epochs, n_channels, n_channels), np.nan, dtype=float)

    for ei in range(n_epochs):
        epoch_data = data[ei]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            con_mat[ei] = np.corrcoef(epoch_data)

    mean_mat = np.nanmean(con_mat, axis=0)
    return con_mat, mean_mat, chs_present


def main(
    subjects=None,
    bids_root=None,
    roi_json=None,
    part=None,
    condition=None,
    tmin=-1.0,
    tmax=1.5,
    stepsize=0.5,
    deriv_dir='freqFilt/figs',
    epoch_suffix='sig_HG_ev-epo',
    fmin=70.0,
    fmax=150.0,
    n_jobs=8,
    output_dir='.',
    orthogonalize=False,
    log=False,
    absolute=True,
):
    if subjects is None:
        subjects = ['D0137']
    if part is None:
        raise ValueError('--part is required')
    if roi_json is None:
        raise ValueError('--roi_json is required')
    if bids_root is None:
        raise ValueError('--bids_root is required')
    if condition is None:
        raise ValueError('--condition is required')

    os.makedirs(output_dir, exist_ok=True)
    out_dir = os.path.join(output_dir, 'envcorr_timewindow')
    os.makedirs(out_dir, exist_ok=True)

    epoch_dicts, df_epochs = load_epochs(
        subjects,
        bids_root,
        condition=condition,
        deriv_dir=deriv_dir,
        epoch_suffix=epoch_suffix,
    )
    df_epochs.to_csv(os.path.join(output_dir, 'epoch_summary.csv'), index=False)
    print('Epoch summary saved.')

    # Windows over time
    windows = make_windows(tmin, tmax, stepsize)
    if len(windows) == 0:
        raise SystemExit("No windows generated — check tmin/tmax and stepsize.")
    print(f"Generated {len(windows)} windows: {windows}")

    all_records = []
    all_sig_pairs_records = []

    for subj in subjects:
        epochs = epoch_dicts.get(subj)
        if epochs is None:
            print(f"Skipping {subj}: no epochs loaded")
            continue

        print(f"Processing subject {subj}...")
        chs = find_roi_names(part, subj, roi_json, epochs.ch_names)

        for (w0, w1) in windows:
            print(f"  Window {w0} to {w1} s — cropping epochs and computing Pearson correlation...")
            try:
                epochs_win = epochs.copy().crop(tmin=w0, tmax=w1)
            except Exception as e:
                print(f"    Error cropping epochs for {subj}, window {w0}-{w1}: {e}")
                continue

            if len(epochs_win) < 2:
                print(f"    Window {w0}-{w1}: less than 2 epochs after cropping — skipping.")
                continue

            try:
                con_mat, mean_mat, chs_present = compute_pearson_correlation_for_envelope_epochs(
                    epochs_win,
                    chs,
                )
            except Exception as e:
                print(f"    Pearson correlation failed for {subj}, window {w0}-{w1}: {e}")
                continue

            if con_mat is None:
                continue

            # Save pairwise summary from the mean matrix.
            idx_pairs = list(combinations(range(len(chs_present)), 2))
            for i, j in idx_pairs:
                ch1, ch2 = chs_present[i], chs_present[j]
                vals = con_mat[:, i, j]
                mean_val = float(np.nanmean(vals))
                std_val = float(np.nanstd(vals))
                rec = {
                    'subject': subj,
                    'condition': condition,
                    'part': part[0],
                    'ch1': ch1,
                    'ch2': ch2,
                    'window_start': w0,
                    'window_end': w1,
                    'n_epochs_in_window': len(epochs_win),
                    'envcorr_mean': mean_val,
                    'envcorr_std': std_val,
                    'fmin': fmin,
                    'fmax': fmax,
                    'orthogonalize': orthogonalize,
                    'log': log,
                    'absolute': absolute,
                }
                all_records.append(rec)

            # Store the full matrix for this subject-window.
            out_mat = {
                'subject': subj,
                'condition': condition,
                'part': part[0],
                'window_start': w0,
                'window_end': w1,
                'ch_names': chs_present,
                'conn_mean_matrix': mean_mat,
                'conn_epoch_matrices': con_mat,
            }
            mat_fname = os.path.join(
                out_dir,
                f"envcorr_{subj}_{sanitize_filename(part[0])}_{sanitize_filename(condition)}_{w0:.2f}_{w1:.2f}.pkl",
            )
            pd.to_pickle(out_mat, mat_fname)
            print(f"    Saved matrix pickle: {mat_fname}")

    # Save pairwise summary CSV / pickle
    if len(all_records) > 0:
        df_sum = pd.DataFrame(all_records)
        csv_path = os.path.join(
            out_dir,
            f"envcorr_summary_{str(subjects)[2:-2]}_{part[0]}_{condition}.csv",
        )
        df_sum.to_csv(csv_path, index=False)
        pkl_path = os.path.join(
            out_dir,
            f"envcorr_summary_{str(subjects)[2:-2]}_{part[0]}_{condition}.pkl",
        )
        pd.to_pickle(all_records, pkl_path)
        print(f"Saved summary to {csv_path}")
        print(f"Saved full summary pickle to {pkl_path}")
    else:
        print("No results to save (no records).")

    print("Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute Pearson correlation on precomputed gamma envelope epochs.')
    parser.add_argument('--bids_root', type=str, required=True)
    parser.add_argument('--subjects', nargs='+', required=True)
    parser.add_argument('--roi_json', type=str, required=True)
    parser.add_argument('--part', nargs='+', required=True)
    parser.add_argument('--condition', type=str, required=True, help='e.g., stimulus_c')
    parser.add_argument('--tmin', type=float, required=True)
    parser.add_argument('--tmax', type=float, required=True)
    parser.add_argument('--stepsize', type=float, required=True, help='window length (s), contiguous non-overlapping windows')
    parser.add_argument('--fmin', type=float, default=70.0, help='High-gamma lower bound (Hz)')
    parser.add_argument('--fmax', type=float, default=150.0, help='High-gamma upper bound (Hz)')
    parser.add_argument('--n_jobs', type=int, default=8)
    parser.add_argument('--output_dir', type=str, default='.')
    parser.add_argument('--deriv_dir', type=str, default='freqFilt/figs')
    parser.add_argument('--epoch_suffix', type=str, default='sig_HG_ev-epo')
    parser.add_argument('--orthogonalize', type=str, default='pairwise', choices=['pairwise', 'False'])
    parser.add_argument('--log', action='store_true')
    parser.add_argument('--absolute', action='store_true', default=True)
    args = parser.parse_args()

    orth = args.orthogonalize
    if orth == 'False':
        orth = False

    print("--------- PARSED ARGUMENTS ---------")
    print(f"args.subjects: {args.subjects}")
    print(f"args.part: {args.part}")
    print(f"args.condition: {args.condition}")
    print(f"args.fmin-fmax: {args.fmin}-{args.fmax}")

    main(
        subjects=args.subjects,
        bids_root=args.bids_root,
        roi_json=args.roi_json,
        part=args.part,
        condition=args.condition,
        tmin=args.tmin,
        tmax=args.tmax,
        stepsize=args.stepsize,
        deriv_dir=args.deriv_dir,
        epoch_suffix=args.epoch_suffix,
        fmin=args.fmin,
        fmax=args.fmax,
        n_jobs=args.n_jobs,
        output_dir=args.output_dir,
        orthogonalize=orth,
        log=args.log,
        absolute=args.absolute,
    )
