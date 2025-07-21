import os
import json
import numpy as np
import mne
import pandas as pd
import matplotlib.pyplot as plt
from mne_connectivity import spectral_connectivity_time
import matplotlib.ticker as ticker

def load_epochs(subjects,
                bids_root,
                deriv_dir='freqFilt/figs',
                event_list=None,
                epoch_suffix='ev1-epo',
                nch_example=3):
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


def compute_and_plot_connectivity(epoch_dicts,
                                  subject,
                                  event,
                                  ch_names,
                                  fmin=3.0,
                                  fmax=8.0,
                                  mode='multitaper',
                                  sfreq=None,
                                  output='connectivity.png'):
    if len(ch_names) != 2:
        raise ValueError(f"Please provide exactly two channel names, got {ch_names}")

    epochs = epoch_dicts.get(event, {}).get(subject)
    if epochs is None:
        raise KeyError(f"No epochs found for subject {subject}, event {event}")

    i = epochs.ch_names.index(ch_names[0])
    j = epochs.ch_names.index(ch_names[1])

    con_ij = spectral_connectivity_time(
        epochs,
        mode=mode,
        indices=([i], [j]),
        fmin=fmin,
        fmax=fmax,
        freqs=(fmin, fmax),
        sfreq=sfreq or epochs.info['sfreq']
    )
    con_ji = spectral_connectivity_time(
        epochs,
        mode=mode,
        indices=([j], [i]),
        fmin=fmin,
        fmax=fmax,
        freqs=(fmin, fmax),
        sfreq=sfreq or epochs.info['sfreq']
    )

    arr_ij = con_ij.get_data()
    arr_ji = con_ji.get_data()

    if arr_ij.ndim == 4:
        arr_ij = arr_ij.mean(axis=3)
    data_ij = arr_ij.mean(axis=0).squeeze()
    if data_ij.ndim > 1:
        data_ij = data_ij.mean(axis=0)

    if arr_ji.ndim == 4:
        arr_ji = arr_ji.mean(axis=3)
    data_ji = arr_ji.mean(axis=0).squeeze()
    if data_ji.ndim > 1:
        data_ji = data_ji.mean(axis=0)

    times = epochs.times

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(times, data_ij, label=f"{ch_names[0]}→{ch_names[1]}")
    ax.plot(times, data_ji, label=f"{ch_names[1]}→{ch_names[0]}")
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(f'Connectivity ({mode}, {fmin}-{fmax} Hz)')
    ax.set_title(f'Spectral Connectivity: {subject}, {event} ({ch_names[0]}–{ch_names[1]})')
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
    ax.legend()
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Load epochs, select two ROIs and compute connectivity')
    parser.add_argument('--bids_root', type=str, required=True, help='Path to BIDS root')
    parser.add_argument('--subjects', nargs='+', required=True, help='Subject IDs (e.g. D0057)')
    parser.add_argument('--roi_json', type=str, required=True, help='Path to JSON with channel-ROI mappings')
    parser.add_argument('--rois', nargs=2, required=True, help='Two ROI names to select channels for')
    parser.add_argument('--event', type=str, required=True, help='Event name to analyze')
    parser.add_argument('--channels', nargs=2, help='Two channel names (optional, not used currently)')
    parser.add_argument('--deriv_dir', type=str, default='freqFilt/figs', help='Derivatives dir')
    parser.add_argument('--epoch_suffix', type=str, default='ev1-epo', help='Epoch suffix')
    parser.add_argument('--nch_example', type=int, default=3, help='Number of channel examples in summary')
    parser.add_argument('--fmin', type=float, default=3.0, help='Low freq bound')
    parser.add_argument('--fmax', type=float, default=8.0, help='High freq bound')
    parser.add_argument('--mode', type=str, default='multitaper', help='Connectivity mode')
    parser.add_argument('--output', type=str, default='connectivity.png', help='Output figure name')
    args = parser.parse_args()

    # Load ROI mappings
    with open(args.roi_json, 'r') as f:
        roi_data = json.load(f)

    # Load epochs
    epoch_dicts, df_summary = load_epochs(
        args.subjects,
        bids_root=args.bids_root,
        deriv_dir=args.deriv_dir,
        epoch_suffix=args.epoch_suffix,
        nch_example=args.nch_example
    )
    print("Finished loading epochs. Summary:")

    # Prepare mapping of subjects to their two ROIs' first channels
    roi1, roi2 = args.rois
    channel_map = {}
    for subj in args.subjects:
        subj_mapping = roi_data.get(subj, {})
        filt_dict = subj_mapping.get('filtROI_dict', {})
        channels1 = filt_dict.get(roi1, [])
        channels2 = filt_dict.get(roi2, [])
        # Uncertainty: what to do if a region has no channels or fewer than 1?
        if not channels1 or not channels2:
            raise KeyError(f"Subject {subj}: ROI '{roi1}' or '{roi2}' has no channels")
        channel_map[subj] = [channels1[0], channels2[0]]

    # Update summary to include both ROI names and their first channels
    df_summary['roi_1'] = roi1
    df_summary['roi_2'] = roi2
    df_summary['ch_roi_1'] = df_summary['subject'].map(lambda s: channel_map[s][0])
    df_summary['ch_roi_2'] = df_summary['subject'].map(lambda s: channel_map[s][1])

    print(df_summary)
    df_summary.to_csv('epoch_summary.csv', index=False)
    print('Summary saved to epoch_summary.csv')

    for subj in args.subjects:
        ch_pair = channel_map[subj]
        print(f"Subject {subj} - using channels {{ch_pair[0]}} from ROI '{roi1}' and {{ch_pair[1]}} from ROI '{roi2}'")

        # Compute connectivity on those two channels
        compute_and_plot_connectivity(
            epoch_dicts,
            subject=subj,
            event=args.event,
            ch_names=ch_pair,
            fmin=args.fmin,
            fmax=args.fmax,
            mode=args.mode,
            sfreq=None,
            output=f"{subj}_{args.event}_{roi1}_{roi2}_connectivity.png"
        )
        print(f"Connectivity plot saved for subject {subj}, ROIs {roi1}, {roi2}")

    print('All connectivity computations done.')
