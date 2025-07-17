import os
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
    """
    Load Epochs for given subjects from a derivatives directory and return epoch dicts and summary DataFrame.

    Adds example channel names to the summary for quick reference.
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


def compute_and_plot_connectivity(epoch_dicts,
                                  subject,
                                  event,
                                  ch_names,
                                  fmin=3.0,
                                  fmax=8.0,
                                  mode='multitaper',
                                  sfreq=None,
                                  output='connectivity.png'):
    """
    Compute spectral connectivity over time between two channels and plot results.
    """
    if len(ch_names) != 2:
        raise ValueError(f"Please provide exactly two channel names, got {ch_names}")

    epochs = epoch_dicts.get(event, {}).get(subject)
    if epochs is None:
        raise KeyError(f"No epochs found for subject {subject}, event {event}")

    # Get channel indices
    i = epochs.ch_names.index(ch_names[0])
    j = epochs.ch_names.index(ch_names[1])

    # Compute connectivity for each direction
    con_ij = spectral_connectivity_time(
        epochs,
        mode=mode,
        indices=([i], [j]),
        fmin=fmin,
        fmax=fmax,
        sfreq=sfreq or epochs.info['sfreq'],
        freqs=(fmin, fmax)
    )
    con_ji = spectral_connectivity_time(
        epochs,
        mode=mode,
        indices=([j], [i]),
        fmin=fmin,
        fmax=fmax,
        sfreq=sfreq or epochs.info['sfreq'],
        freqs=(fmin, fmax)
    )

    # Extract data
    data_ij = con_ij.get_data().mean(axis=1).squeeze()  # (n_times,)
    data_ji = con_ji.get_data().mean(axis=1).squeeze()

    # Extract time vector
    try:
        times = con_ij.times
    except AttributeError:
        for key in ('time', 'times', 'epochs'):
            if key in con_ij.coords:
                times = con_ij.coords[key].values
                break
        else:
            raise AttributeError(f"Could not find time coordinate. Available coords: {list(con_ij.coords)}")

    # Plot on fresh figure
    fig, ax = plt.subplots(figsize=(12, 4))
    
    ax.plot(times, data_ij, label=f"{ch_names[0]}→{ch_names[1]}")
    ax.plot(times, data_ji, label=f"{ch_names[1]}→{ch_names[0]}")

    ax.set_xlabel('Time (s)')
    ax.set_ylabel(f'Connectivity ({mode}, {fmin}-{fmax} Hz)')
    ax.set_title(f'Spectral Connectivity: {subject}, {event} ({ch_names[0]}–{ch_names[1]})')
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
    

    #ax.legend(handles=[line_ij, line_ji])
    
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Load epochs and compute connectivity')
    parser.add_argument('--bids_root', type=str, required=True, help='Path to BIDS root')
    parser.add_argument('--subjects', nargs='+', required=True, help='Subject IDs')
    parser.add_argument('--channels', nargs=2, required=True, help='Two channel names')
    parser.add_argument('--event', type=str, required=True, help='Event name to analyze')
    parser.add_argument('--deriv_dir', type=str, default='freqFilt/figs', help='Derivatives dir')
    parser.add_argument('--epoch_suffix', type=str, default='ev1-epo', help='Epoch suffix')
    parser.add_argument('--nch_example', type=int, default=3, help='Number of channel examples in summary')
    parser.add_argument('--fmin', type=float, default=3.0, help='Low freq bound')
    parser.add_argument('--fmax', type=float, default=8.0, help='High freq bound')
    parser.add_argument('--mode', type=str, default='multitaper', help='Connectivity mode')
    parser.add_argument('--output', type=str, default='connectivity.png', help='Output figure name')
    args = parser.parse_args()

    subs = [s.replace('sub-', '') for s in args.subjects]
    epoch_dicts, df_summary = load_epochs(
        subs,
        bids_root=args.bids_root,
        deriv_dir=args.deriv_dir,
        epoch_suffix=args.epoch_suffix,
        nch_example=args.nch_example
    )
    print("Finished loading epochs. Summary:")
    print(df_summary)
    df_summary.to_csv('epoch_summary.csv', index=False)
    print('Summary saved to epoch_summary.csv')

    for subj in subs:
        compute_and_plot_connectivity(
            epoch_dicts,
            subject=subj,
            event=args.event,
            ch_names=args.channels,
            fmin=args.fmin,
            fmax=args.fmax,
            mode=args.mode,
            sfreq=None,
            output=f"{subj}_{args.event}_connectivity.png"
        )
    print('All connectivity computations done.')
