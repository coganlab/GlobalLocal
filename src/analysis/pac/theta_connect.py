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


def compute_and_plot_granger(epoch_dicts,
                              subject,
                              event,
                              ch_names,
                              fmin=3.0,
                              fmax=8.0,
                              n_freqs=25,
                              sfreq=None,
                              n_cycles=None,
                              output='granger_spectrum.png'):
    """
    Compute and plot Granger causality spectrum between two channels.
    """
    epochs = epoch_dicts.get(event, {}).get(subject)
    if epochs is None:
        raise KeyError(f"No epochs for {subject}, event {event}")

    # Get indices and data
    i, j = [epochs.ch_names.index(ch) for ch in ch_names]
    freqs = np.linspace(fmin, fmax, n_freqs)
    sf = sfreq or epochs.info['sfreq']
    if n_cycles is None:
        n_cycles = freqs / 2

    # Compute Granger causality spectrum
    con_ij = spectral_connectivity_time(
        epochs,
        mode='multitaper',
        method='gc',
        indices=([np.array([i])], [np.array([j])]),
        freqs=freqs,
        sfreq=sf,
        n_cycles=n_cycles,
        average=False
    )
    con_ji = spectral_connectivity_time(
        epochs,
        mode='multitaper',
        method='gc',
        indices=([np.array([j])], [np.array([i])]),
        freqs=freqs,
        sfreq=sf,
        n_cycles=n_cycles,
        average=False
    )

    # Extract and average (n_epochs, n_cons, n_freqs)
    spec_ij = con_ij.get_data().mean(axis=(0,1)).squeeze()
    spec_ji = con_ji.get_data().mean(axis=(0,1)).squeeze()

    # Plot Granger spectrum
    plt.figure(figsize=(8, 4))
    plt.plot(freqs, spec_ij, label=f"{ch_names[0]}→{ch_names[1]}")
    plt.plot(freqs, spec_ji, label=f"{ch_names[1]}→{ch_names[0]}")
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Granger Causality')
    plt.title(f'Granger Causality Spectrum: {subject}, {event} ({ch_names[0]}–{ch_names[1]})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output)
    plt.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Compute Granger causality spectrum for ROI pairs')
    parser.add_argument('--bids_root', type=str, required=True, help='Path to BIDS root')
    parser.add_argument('--subjects', nargs='+', required=True, help='Subject IDs')
    parser.add_argument('--roi_json', type=str, required=True, help='JSON mapping subjects to ROI channels')
    parser.add_argument('--rois', nargs=2, required=True, help='Two ROI names to analyze')
    parser.add_argument('--event', type=str, required=True, help='Event name (e.g., Stimulus)')
    parser.add_argument('--fmin', type=float, default=3.0, help='Low freq bound')
    parser.add_argument('--fmax', type=float, default=8.0, help='High freq bound')
    parser.add_argument('--n_freqs', type=int, default=25, help='Number of frequency points')
    parser.add_argument('--output_dir', type=str, default='.', help='Directory to save plots')
    args = parser.parse_args()

    # Load ROI mappings
    with open(args.roi_json, 'r') as f:
        roi_data = json.load(f)

    # Load epochs
    epoch_dicts, df_summary = load_epochs(
        args.subjects,
        bids_root=args.bids_root,
        deriv_dir='freqFilt/figs',
        epoch_suffix='ev1-epo',
        nch_example=3
    )
    df_summary.to_csv('epoch_summary.csv', index=False)
    print('Epoch summary saved to epoch_summary.csv')

    # Iterate subjects
    roi1, roi2 = args.rois
    for subj in args.subjects:
        ch1 = roi_data[subj]['filtROI_dict'][roi1][0]
        ch2 = roi_data[subj]['filtROI_dict'][roi2][0]
        out = os.path.join(args.output_dir, f"{subj}_{args.event}_granger_spectrum.png")
        print(f"Computing Granger causality for {subj}: {ch1}→{ch2}")
        compute_and_plot_granger(
            epoch_dicts,
            subj,
            args.event,
            [ch1, ch2],
            fmin=args.fmin,
            fmax=args.fmax,
            n_freqs=args.n_freqs,
            sfreq=None,
            n_cycles=None,
            output=out
        )
        print(f"Saved plot to {out}")

    print('All Granger causality computations done.')
