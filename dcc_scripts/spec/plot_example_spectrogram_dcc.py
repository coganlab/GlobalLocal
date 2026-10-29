#!/usr/bin/env python
"""
Example-electrode spectrogram for Fig 2c: high-gamma power rising after
stimulus onset in one lPFC electrode.

Reads the rescaled (dB re: pre-stimulus baseline) TFRs written by
make_wavelets_dcc.py, averages the requested conditions, and saves the panel
as .png and .pdf to dcc_scripts/spec/figs/. The filename records the subject,
channel, conditions, display window and colour limits.

Light enough to run directly (no sbatch):
    python dcc_scripts/spec/plot_example_spectrogram_dcc.py
"""
import os
import sys

try:
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_script_dir = os.getcwd()

project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import matplotlib
matplotlib.use('Agg')  # headless backend for the cluster
import matplotlib.pyplot as plt
import mne

from ieeg.io import get_data
from src.analysis.spec.wavelet_functions import (load_wavelets, load_multitaper,
                                                 plot_channel_spectrogram)
from src.analysis.utils.general_utils import resolve_lab_root

# ============================================================================
# PARAMETERS
# ============================================================================
LAB_ROOT = None  # None -> resolve_lab_root default
TASK = 'GlobalLocal'

# D0094 LFPM12 (ctx_lh_S_front_inf; in both the lPFC and dlPFC ROIs) had the
# largest mean 70-150 Hz response, 0.1-0.8 s after onset, of the 418 lPFC
# channels across the 24 subjects, with a quiet baseline.
SUBJECT_ID = 'D0094'
CHANNEL = 'LFPM12'

# Stimulus_c + Stimulus_i together are all correct trials (stimulus_congruency_conditions).
CONDITION_NAMES = ['Stimulus_c', 'Stimulus_i']
SPEC_TYPE = 'multitaper'  # multitaper's linear 2 Hz bins resolve high gamma; wavelets have ~6 bins in 70-200 Hz

TIMES = (-0.5, 1.5)    # display window (s); trims wavelet edge artifacts
FREQ_RANGE = (None, 200.)  # Hz; None -> lowest computed frequency
HG_BAND = (70., 150.)  # marked with dotted lines
VLIM = (-3., 3.)       # dB; LFPM12's 70-150 Hz response peaks near +3 dB
YSCALE = 'linear'
FIGSIZE = (4, 3)       # inches
FONT_SIZE = 10

# Save figures to the figs folder inside dcc_scripts/spec/figs.
SAVE_DIR = os.path.join(project_root, "dcc_scripts", "spec", "figs")


def main():
    layout = get_data(TASK, root=resolve_lab_root(LAB_ROOT))
    load_func = {'wavelet': load_wavelets, 'multitaper': load_multitaper}[SPEC_TYPE]

    specs = [load_func(SUBJECT_ID, layout, cond, rescaled=True)
             for cond in CONDITION_NAMES]
    if CHANNEL in specs[0].info['bads']:
        raise ValueError(f"{CHANNEL} is marked bad for {SUBJECT_ID}")
    # Equal weights: each condition's mean counts once, regardless of trial count.
    spec = mne.time_frequency.combine_tfr(specs, weights='equal')
    spec.crop(*TIMES)

    with plt.rc_context({'font.size': FONT_SIZE}):
        fig = plot_channel_spectrogram(spec, CHANNEL, fmin=FREQ_RANGE[0],
                                       fmax=FREQ_RANGE[1], vlim=VLIM,
                                       yscale=YSCALE, band=HG_BAND)
        fig.set_size_inches(*FIGSIZE)
        fig.tight_layout()

    os.makedirs(SAVE_DIR, exist_ok=True)
    stem = (f"example_spectrogram_{SUBJECT_ID}_{CHANNEL}_"
            f"{'+'.join(CONDITION_NAMES)}_{SPEC_TYPE}_"
            f"{TIMES[0]}to{TIMES[1]}s_vlim{VLIM[1]}dB")
    for ext in ('png', 'pdf'):
        path = os.path.join(SAVE_DIR, f'{stem}.{ext}')
        fig.savefig(path, bbox_inches='tight', dpi=300)
        print("Saved figure:", path)
    plt.close(fig)


if __name__ == "__main__":
    main()
