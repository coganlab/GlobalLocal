# here is where we load the data, correct the event onset, and save it to derivatives

"""Build bipolar-referenced iEEG derivatives from BIDS recordings.

Purpose:
- Load iEEG data from a BIDS layout.
- Group contacts by shank (name prefix), sort by contact index, and create adjacent bipolar pairs.
- Compute bipolar signals (A - B) and construct a new Raw object with bipolar channels.
- Derive bipolar channel coordinates as midpoints from the existing montage (if present).
- Preserve annotations and align absolute timing by reusing meas_date when available.
- Save the result to BIDS derivatives with desc="bipolar" using the project save utility.

Outputs:
- A bipolar-referenced iEEG recording saved as a BIDS derivative (plus associated metadata handled by the save utility).
"""
import argparse
from os import path
import pandas as pd

import mne
import numpy as np
from mne_bids import BIDSPath, read_raw_bids
from mne_bids.dig import _read_dig_bids
from mne_bids.path import _find_matching_sidecar
from ieeg.navigate import channel_outlier_marker
from ieeg.navigate import outliers_to_nan
from ieeg.calc.scaling import rescale
from ieeg.timefreq import gamma
from ieeg.io import raw_from_layout
from bids.layout import BIDSLayout
from pathlib import Path
from ieeg.navigate import trial_ieeg
from mne_bids import get_bids_path_from_fname
import re
import logging
import sys
import gc
from ieeg.io import save_derivative

# Simple logging: everything INFO and above to stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)
 

def preproc(
    bids_layout: BIDSLayout,
    subject: str,
    **kwargs
):

    # Load data
    raw = raw_from_layout(
        bids_layout.derivatives[('derivatives/clean')],
        subject=subject,
        extension='.edf',
        desc='clean',
        **kwargs
    )

    # record bad channels but don't drop yet - we need them for bipolar pairing
    logger.info(f"Loaded subject={subject} with {len(raw.ch_names)} channels @ {raw.info['sfreq']} Hz")
    bads = set(raw.info.get('bads', []))
    if bads:
        logger.info(f"Found {len(bads)} bad channels: {bads}")
    raw.load_data()

    # group electrodes by shank, order them by contact number
    pattern = re.compile(r'^(.+?)(\d+)$')
    shaft_groups = {}
    for ch_name in raw.ch_names:
        match = pattern.match(ch_name)
        if not match:
            continue
        prefix, contact_num = match.group(1), int(match.group(2))
        if prefix not in shaft_groups:
            shaft_groups[prefix] = []
        shaft_groups[prefix].append((ch_name, contact_num))
    for prefix in shaft_groups:
        shaft_groups[prefix].sort(key=lambda x: x[1])
    logger.info(f"Identified {len(shaft_groups)} shanks. Counts: " + 
                ", ".join([f"{k}:{len(v)}" for k, v in sorted(shaft_groups.items())]))

    pairs = []
    bipolar_names = []
    for prefix, ordered in shaft_groups.items():
        for i in range(len(ordered) - 1):
            ch1, n1 = ordered[i]
            ch2, n2 = ordered[i + 1]
            # skip if either channel is bad
            if ch1 in bads or ch2 in bads:
                continue
            pairs.append((ch1, ch2))
            bipolar_names.append(f"{subject}_{prefix}{n1}-{n2}")
    logger.info(f"Constructed {len(pairs)} bipolar pairs across all shanks")

    bipolar_coords = {}
    dig = raw.get_montage()
    if dig is not None:
        pos = dig.get_positions()
        ch_pos = pos.get('ch_pos', {})
        coord_map = {name: tuple(coord) for name, coord in ch_pos.items()}
        for (ch1, ch2), new_name in zip(pairs, bipolar_names):
            if ch1 in coord_map and ch2 in coord_map:
                x = (coord_map[ch1][0] + coord_map[ch2][0]) / 2.0
                y = (coord_map[ch1][1] + coord_map[ch2][1]) / 2.0
                z = (coord_map[ch1][2] + coord_map[ch2][2]) / 2.0
                bipolar_coords[new_name] = (x, y, z)
        logger.info(f"Montage available. Computed coordinates for {len(bipolar_coords)} bipolar channels")
    else:
        logger.info("No montage found on raw; bipolar coordinates will be empty")

    data_list = []
    for ch1, ch2 in pairs:
        d1 = raw.get_data(picks=[ch1])
        d2 = raw.get_data(picks=[ch2])
        data_list.append(d1 - d2)
    if data_list:
        data = np.vstack(data_list)
        info = mne.create_info(bipolar_names, raw.info['sfreq'], ch_types=['seeg'] * len(bipolar_names))
        raw_bip = mne.io.RawArray(data, info)
        if raw.info.get('meas_date', None) is not None:
            raw_bip.set_meas_date(raw.info['meas_date'])
        if bipolar_coords:
            # Use MRI (ACPC-aligned) coordinate frame for iEEG bipolar contacts
            montage = mne.channels.make_dig_montage(ch_pos=bipolar_coords, coord_frame='ras')
            raw_bip.set_montage(montage)
        if raw.annotations is not None:
            # Filter annotations to keep only one BAD boundary per run boundary
            # save_derivative uses 'BAD boundary' to split runs, but multiple BAD boundary
            # annotations at each boundary cause incorrect splitting
            from mne import Annotations
            ann = raw.annotations.copy()
            
            # Separate boundary and non-boundary annotations
            boundary_indices = [i for i, a in enumerate(ann) if a['description'] == 'BAD boundary']
            
            if len(boundary_indices) > 0:
                # Get boundary onsets and keep only the first one at each unique time (within 2s tolerance)
                boundary_onsets = [ann[i]['onset'] for i in boundary_indices]
                unique_boundary_onsets = []
                for onset in boundary_onsets:
                    if not unique_boundary_onsets or abs(onset - unique_boundary_onsets[-1]) > 2.0:
                        unique_boundary_onsets.append(onset)
                
                # Build new annotations: non-boundary + unique boundaries
                new_onsets = []
                new_durations = []
                new_descriptions = []
                
                for a in ann:
                    if a['description'] == 'BAD boundary':
                        # Only keep if this is a unique boundary
                        if a['onset'] in unique_boundary_onsets:
                            new_onsets.append(a['onset'])
                            new_durations.append(a['duration'])
                            new_descriptions.append(a['description'])
                            unique_boundary_onsets.remove(a['onset'])  # Don't add duplicates
                    else:
                        new_onsets.append(a['onset'])
                        new_durations.append(a['duration'])
                        new_descriptions.append(a['description'])
                
                new_ann = Annotations(onset=new_onsets, duration=new_durations, 
                                     description=new_descriptions, orig_time=ann.orig_time)
                logger.info(f"Filtered annotations: {len(ann)} -> {len(new_ann)} (removed {len(ann) - len(new_ann)} duplicate boundaries)")
                raw_bip.set_annotations(new_ann)
            else:
                raw_bip.set_annotations(ann)
        # keep source file reference so save_derivative can infer BIDS entities
        if getattr(raw, 'filenames', None):
            try:
                raw_bip._filenames = raw.filenames
            except Exception:
                pass
        raw = raw_bip
        logger.info(f"Created bipolar Raw: n_channels={len(bipolar_names)}, n_times={raw.n_times}, sfreq={raw.info['sfreq']}")

    # save raw as derivatives
    save_derivative(
        raw,
        bids_layout,
        "bipolar",
        True,
    )

def main(bids_root: str, subject: str, **kwargs):

    bids_layout = BIDSLayout(
        root=bids_root,
        derivatives=True,
    )
    preproc(
        bids_layout=bids_layout,
        subject=subject,
        **kwargs
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bids_root", default="/cwork/ns458/BIDS-1.4_Phoneme_sequencing/BIDS", type=str)
    parser.add_argument("--subject", type=str, default='D0022')

    _args = parser.parse_args()
    main(**vars(_args))
