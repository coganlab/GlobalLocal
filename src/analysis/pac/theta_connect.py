import os
import mne


def load_epochs(subjects,
                bids_root,
                deriv_dir='freqFilt/figs',
                event_list=None,
                epoch_suffix='ev1-epo'):
    """
    Load Epochs for given subjects from a derivatives directory.

    Parameters
    ----------
    subjects : list of str
        List of subject IDs (without 'sub-' prefix).
    bids_root : str
        Path to the root of the BIDS dataset.
    deriv_dir : str
        Relative path under BIDS root to the derivatives folder containing epochs.
    event_list : list of str
        List of event names to load epochs for. Defaults to ['Stimulus', 'Response'].
    epoch_suffix : str
        Suffix in epoch filenames (default 'ev1-epo').

    Returns
    -------
    epochs_dict : dict
        Mapping (subject ID, event) -> MNE Epochs.
    """
    # Default events
    if event_list is None:
        event_list = ['Stimulus', 'Response']

    epochs_dict = {}

    for subj in subjects:
        print(f"Subject {subj}: loading epochs...")

        for event in event_list:
            # Build expected file path
            fname = f"{subj}_{event}_{epoch_suffix}.fif"
            eph_path = os.path.join(bids_root,
                                    'derivatives',
                                    deriv_dir,
                                    subj,
                                    fname)
            if os.path.isfile(eph_path):
                print(f"  Found epoch file for event '{event}': {eph_path}")
                try:
                    epochs = mne.read_epochs(eph_path, preload=True)
                    epochs_dict[(subj, event)] = epochs
                    # Print summary information
                    n_epochs = len(epochs)
                    sfreq = epochs.info.get('sfreq', 'Unknown')
                    n_channels = len(epochs.ch_names)
                    tmin, tmax = epochs.tmin, epochs.tmax
                    data_shape = epochs.get_data().shape  # (n_epochs, n_channels, n_times)
                    print(f"    Loaded epochs: {n_epochs} epochs")
                    print(f"    Sampling frequency: {sfreq} Hz")
                    print(f"    Number of channels: {n_channels}")
                    print(f"    Time range per epoch: {tmin} to {tmax} seconds")
                    print(f"    Data shape (epochs, channels, time points): {data_shape}")
                except Exception as e:
                    print(f"    Error loading epochs file: {e}")
            else:
                print(f"  Epoch file not found: {eph_path}")

    return epochs_dict


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Load epoch FIF files for BIDS subjects')
    parser.add_argument('--bids_root', type=str, required=True,
                        help='Path to BIDS dataset root')
    parser.add_argument('--subjects', nargs='+', required=True,
                        help='List of subject IDs (e.g., 0123 0456)')
    parser.add_argument('--deriv_dir', type=str, default='freqFilt/figs',
                        help='Relative derivatives directory under BIDS root')
    parser.add_argument('--epoch_suffix', type=str, default='ev1-epo',
                        help='Epoch filename suffix')
    args = parser.parse_args()

    subs = [s.replace('sub-', '') for s in args.subjects]
    epochs = load_epochs(
        subjects=subs,
        bids_root=args.bids_root,
        deriv_dir=args.deriv_dir,
        epoch_suffix=args.epoch_suffix
    )
    print('Finished loading epochs for all subjects and default events.')
