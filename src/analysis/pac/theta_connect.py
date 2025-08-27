import os
import json
import numpy as np
import mne
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from mne_connectivity import spectral_connectivity_time

def find_roi_names(part):
    rois_dict = {
    'dlpfc': ["G_front_middle", "G_front_sup", "S_front_inf", "S_front_middle", "S_front_sup"],
    'acc': ["G_and_S_cingul-Ant", "G_and_S_cingul-Mid-Ant"],
    'parietal': ["G_parietal_sup", "S_intrapariet_and_P_trans", "G_pariet_inf-Angular", "G_pariet_inf-Supramar"],
    'lpfc': ["G_front_inf-Opercular", "G_front_inf-Orbital", "G_front_inf-Triangul", "G_front_middle", "G_front_sup", "Lat_Fis-ant-Horizont", "Lat_Fis-ant-Vertical", "S_circular_insula_ant", "S_circular_insula_sup", "S_front_inf", "S_front_middle", "S_front_sup"],
    'v1': ["G_oc-temp_med-Lingual", "S_calcarine", "G_cuneus"],
    'occ': ["G_cuneus", "G_and_S_occipital_inf", "G_occipital_middle", "G_occipital_sup", "G_oc-temp_lat-fusifor", "G_oc-temp_med-Lingual", "Pole_occipital", "S_calcarine", "S_oc_middle_and_Lunatus", "S_oc_sup_and_transversal", "S_occipital_ant"]
    }
    roi_list=rois_dict[part[0]]#for now only one part at a time
    print(f"Using ROIs: {roi_list}")
    # Load ROI mapping
    with open(args.roi_json, 'r') as f:
        roi_data = json.load(f)

    # Gather channels from specified ROIs
    chs = []
    for roi in roi_list:
        for key in roi_data[subj]['filtROI_dict']:
            if roi in key:
                chs.extend(roi_data[subj]['filtROI_dict'][key])
    chs = list(dict.fromkeys(chs))  # de-duplicate
    print(f"Computing coherence for {subj}, {len(chs)} channels, {len(chs)*(len(chs)-1)//2} pairs")
    return chs


def load_epochs(subjects,
                bids_root,
                deriv_dir='freqFilt/figs',
                event_list=None,
                epoch_suffix='ev1-epo',
                nch_example=3):
    """
    Load epochs for each subject and event. Returns:
      - epoch_dicts: {event: {subject: Epochs}}
      - df_summary: DataFrame summarizing loaded epochs
    """
    if event_list is None:
        event_list = ['Stimulus', 'Response']

    epoch_dicts = {event: {} for event in event_list}
    summary_records = []

    for subj in subjects:
        print(f"Subject {subj}: loading epochs...")
        for event in event_list:
            fname = f"{subj}_{event}_{epoch_suffix}.fif"
            eph_path = os.path.join(bids_root, 'derivatives', deriv_dir, subj, fname)
            if os.path.isfile(eph_path):
                try:
                    epochs = mne.read_epochs(eph_path, preload=True)
                    epoch_dicts[event][subj] = epochs

                    ch_list = epochs.ch_names[:nch_example]
                    ch_example_str = ','.join(ch_list)

                    summary_records.append({
                        'subject': subj,
                        'event': event,
                        'n_epochs': len(epochs),
                        'sfreq': epochs.info.get('sfreq'),
                        'n_channels': len(epochs.ch_names),
                        'tmin': epochs.tmin,
                        'tmax': epochs.tmax,
                        'channel_examples': ch_example_str
                    })
                    print(f"    Loaded {len(epochs)} epochs for {event} ({ch_example_str}...)")
                except Exception as e:
                    print(f"    Error loading {event} for {subj}: {e}")
            else:
                print(f"  Epoch file not found for {event}: {eph_path}")

    df_summary = pd.DataFrame.from_records(summary_records)
    return epoch_dicts, df_summary


def compute_coherence_batch(epochs, chs, freqs, n_cycles, fmin, fmax, n_jobs):
    """
    Compute coherence for all pairs of channels in `chs` at once.
    Returns a dict mapping (ch1,ch2) -> mean coherence value.
    """
    # Pick only ROI channels to speed up computation
    epochs_roi = epochs.copy().pick_channels(chs)

    # Build all unique channel index pairs
    idx_pairs = list(combinations(range(len(chs)), 2))
    u_inds, v_inds = zip(*idx_pairs)

    # Batch compute coherence with average across epochs
    con_all = spectral_connectivity_time(
        epochs_roi,
        mode='multitaper',
        method='coh',
        indices=(list(u_inds), list(v_inds)),
        freqs=freqs,
        sfreq=epochs_roi.info['sfreq'],
        n_cycles=n_cycles,
        average=True,
        n_jobs=n_jobs,
        fmin=fmin,
        fmax=fmax
    )
    # Data: shape can be (n_pairs, n_freqs, n_times) or (n_pairs, n_freqs)
    data_all = con_all.get_data()
    # Flatten freq and time dims if present, then mean per pair
    reshaped = data_all.reshape(data_all.shape[0], -1)
    mean_vals = reshaped.mean(axis=1)

    # Build result dict
    coherence_dict = {}
    for (i, j), val in zip(idx_pairs, mean_vals):
        ch1, ch2 = chs[i], chs[j]
        coherence_dict[(ch1, ch2)] = val
    return coherence_dict


def plot_pair_coherence(freqs, spectrum, ch1, ch2, output_path):
    """
    Plot coherence spectrum for a single channel pair and save.
    """
    plt.figure(figsize=(8, 4))
    plt.plot(freqs, spectrum)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Coherence')
    plt.title(f'{ch1}→{ch2}')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved plot {ch1}->{ch2} to {output_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Compute ROI Coherence more efficiently')
    parser.add_argument('--bids_root', type=str, required=True)
    parser.add_argument('--subjects', nargs='+', required=True)
    parser.add_argument('--roi_json', type=str, required=True)
    parser.add_argument('--part', nargs='+', required=True)
    parser.add_argument('--event', type=str, required=True)
    parser.add_argument('--congruency', type=str, default='all')
    parser.add_argument('--fmin', type=float, default=3.0)
    parser.add_argument('--fmax', type=float, default=8.0)
    parser.add_argument('--n_freqs', type=int, default=25)
    parser.add_argument('--n_jobs', type=int, default=8,
                        help='Adjust based on CPU cores')
    parser.add_argument('--output_dir', type=str, default='.')
    parser.add_argument('--plot', action='store_true', help='Save per-pair plots')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    fig_dir = os.path.join(args.output_dir, 'coherence_figs')
    if args.plot:
        os.makedirs(fig_dir, exist_ok=True)


    # Load epochs
    epoch_dicts, df_epochs = load_epochs(args.subjects, args.bids_root,
                                         event_list=[args.event])
    df_epochs.to_csv(os.path.join(args.output_dir, 'epoch_summary.csv'), index=False)
    print('Epoch summary saved.')

    # Frequency setup
    freqs = np.linspace(args.fmin, args.fmax, args.n_freqs)
    n_cycles = freqs / 2

    # Main coherence computation
    all_records = []
    for subj in args.subjects:
        epochs = epoch_dicts.get(args.event, {}).get(subj)
        if epochs is None:
            print(f"Skipping {subj}: no {args.event} epochs")
            continue
        print(f"Processing subject {subj}...")
        chs = find_roi_names(args.part)
        coh_dict = compute_coherence_batch(
            epochs, chs, freqs, n_cycles,
            args.fmin, args.fmax, args.n_jobs
        )

        # Build summary and optional plotting
        for (ch1, ch2), mean_val in coh_dict.items():
            rec = {'subject': subj, 'event': args.event,
                   'ch1': ch1, 'ch2': ch2,
                   'coh_mean': mean_val}
            all_records.append(rec)
            if args.plot:
                # Detailed spectrum plotting skipped for speed
                pass

    # Save summary CSV
    df_sum = pd.DataFrame(all_records)
    csv_path = os.path.join(args.output_dir, f'coherence_{subj}_{args.part[0]}_{args.event}_summary.csv')
    df_sum.to_csv(csv_path, index=False)
    print(f"Saved coherence summary to {csv_path}")
