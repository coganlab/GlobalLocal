#!/usr/bin/env python
"""Attribute a power-trace excursion to the electrodes that caused it.

Reads the per-condition evoked ``.npz`` files that ``power_traces_dcc.py``
already writes (``<save_dir>/<roi>/<conditions_save_name>_<condition>_<roi>_
evoked.npz``), so it needs no re-run and no access to the raw epochs.

For each condition it reports, over a chosen time window:

* the electrodes with the largest RMS deviation from the across-electrode mean;
* each one's single largest excursion, when it happened, and -- the number that
  actually matters -- how much of the plotted ROI mean that excursion accounts
  for (``excursion / n_electrodes``). If one electrode explains the whole
  visible feature, the feature is that electrode, not an effect;
* a cross-condition summary flagging electrodes that are extreme in some
  conditions but not others. In a blocked design a electrode that is extreme in
  exactly the conditions drawn from one block type is a block-locked artifact:
  condition and block are confounded, so it lands squarely on the interaction
  term you are trying to interpret.

Examples
--------
Diagnose the pre-stimulus period of an LWPC run::

    python diagnose_electrode_deviations.py \
        --save-dir figs/<epochs_root_file>/anova_within_roi \
        --roi lpfc \
        --conditions-save-name stimulus_lwpc_conditions_24_subjects \
        --window None 0.0

Whole epoch, more rows::

    python diagnose_electrode_deviations.py --save-dir ... --roi lpfc \
        --conditions-save-name ... --top 15
"""
import argparse
import os
import sys
from collections import defaultdict

import numpy as np

try:
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_script_dir = os.getcwd()
project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.analysis.power.plots import rank_electrode_deviations
# window_mask rather than the plotting module's private equivalent: there an
# empty selection returns None so the ranking can fall back to the full epoch,
# here an all-True mask is exactly the same fallback and needs no special case.
from src.analysis.power.block_diagnostics import window_mask


class _Evoked:
    """Minimal duck-type for ``rank_electrode_deviations``."""

    def __init__(self, data, times, ch_names):
        self.data = data
        self.times = times
        self.ch_names = list(ch_names)


def _parse_bound(text):
    """'None'/'' -> None, otherwise float. Lets --window take an open end."""
    if text is None or text.lower() in ('none', 'null', ''):
        return None
    return float(text)


def load_condition_evokeds(save_dir, roi, conditions_save_name):
    """Return {condition_name: _Evoked} for every saved condition in an ROI."""
    roi_dir = os.path.join(save_dir, roi)
    if not os.path.isdir(roi_dir):
        raise FileNotFoundError(f"No ROI directory at {roi_dir}")

    prefix = f"{conditions_save_name}_"
    suffix = f"_{roi}_evoked.npz"
    out = {}
    for fname in sorted(os.listdir(roi_dir)):
        if not (fname.startswith(prefix) and fname.endswith(suffix)):
            continue
        condition = fname[len(prefix):-len(suffix)]
        with np.load(os.path.join(roi_dir, fname), allow_pickle=True) as npz:
            out[condition] = _Evoked(np.asarray(npz['data'], dtype=float),
                                     np.asarray(npz['times'], dtype=float),
                                     npz['ch_names'])
    if not out:
        raise FileNotFoundError(
            f"No files matching {prefix}*{suffix} in {roi_dir}. Check "
            f"--conditions-save-name against the filenames actually there."
        )
    return out


def worst_excursion(evoked, channel_index, mask):
    """(time, value, |deviation from the across-electrode median|) of the peak.

    The reference is the median, not the mean: one electrode a hundred z-units
    off drags the mean toward itself and inflates every other electrode's
    deviation, which is exactly the situation this script exists to diagnose.
    """
    data = evoked.data
    with np.errstate(invalid='ignore'):
        reference = np.nanmedian(data, axis=0)
    delta = np.abs(data[channel_index] - reference)
    if mask is not None:
        delta = np.where(mask, delta, np.nan)
    if np.all(np.isnan(delta)):
        return None, np.nan, np.nan
    idx = int(np.nanargmax(delta))
    return evoked.times[idx], float(data[channel_index, idx]), float(delta[idx])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--save-dir', required=True,
                        help="Run directory holding the per-ROI subfolders.")
    parser.add_argument('--roi', required=True)
    parser.add_argument('--conditions-save-name', required=True,
                        help="e.g. stimulus_lwpc_conditions_24_subjects")
    parser.add_argument('--window', nargs=2, metavar=('TMIN', 'TMAX'),
                        default=('None', '0.0'),
                        help="Scoring window in seconds; either bound may be "
                             "'None' for the epoch edge. Default: None 0.0 "
                             "(epoch start through stimulus onset).")
    parser.add_argument('--top', type=int, default=8,
                        help="Rows per condition (default 8).")
    parser.add_argument('--min-share', type=float, default=0.02,
                        help="Summary keeps electrodes that account for at "
                             "least this fraction of the plotted ROI mean at "
                             "some sample, in some condition (default 0.02).")
    args = parser.parse_args(argv)

    window = (_parse_bound(args.window[0]), _parse_bound(args.window[1]))
    evokeds = load_condition_evokeds(args.save_dir, args.roi,
                                     args.conditions_save_name)

    lo = 'epoch start' if window[0] is None else f'{window[0]:g}s'
    hi = 'epoch end' if window[1] is None else f'{window[1]:g}s'
    print(f"\nROI {args.roi} — scoring window {lo} to {hi}")
    print(f"Run: {args.save_dir}\n")

    share_by_channel = defaultdict(dict)

    for condition, evoked in evokeds.items():
        n_ch = evoked.data.shape[0]
        mask = window_mask(evoked.times, window)
        ranking = rank_electrode_deviations(evoked, window)
        name_to_index = {n: i for i, n in enumerate(evoked.ch_names)}

        # rms: mean-referenced, matching the ranking in the plot's own report.
        # peak |dev| and share: median-referenced (see worst_excursion).
        print(f"[{condition}] {n_ch} electrodes")
        print(f"{'rank':>4}  {'electrode':<14}{'rms':>10}{'peak |dev|':>12}"
              f"{'at (s)':>9}{'value':>10}{'ROI-mean share':>16}")
        # Every channel gets a share, so the cross-condition table below can
        # compare a channel against itself rather than against a rank cutoff.
        for name, idx in name_to_index.items():
            _, _, dev = worst_excursion(evoked, idx, mask)
            share_by_channel[name][condition] = dev / n_ch

        for rank, (name, score) in enumerate(ranking[:args.top], start=1):
            t, value, dev = worst_excursion(evoked, name_to_index[name], mask)
            t_str = 'n/a' if t is None else f'{t:.3f}'
            print(f"{rank:>4}  {name:<14}{score:>10.4g}{dev:>12.4g}"
                  f"{t_str:>9}{value:>10.4g}{dev / n_ch:>16.4g}")
        print()

    # Cross-condition view: the same electrode, condition by condition. An
    # electrode that dominates the ROI mean in some conditions and not others
    # is the shape that matters here.
    condition_order = list(evokeds)
    material = {
        name: per_cond for name, per_cond in share_by_channel.items()
        if max(per_cond.values()) >= args.min_share
    }
    if not material:
        print(f"No electrode accounts for >= {args.min_share:g} of the ROI "
              f"mean at any sample in this window. The traces here are not "
              f"driven by a single electrode.")
        return 0

    print(f"Electrodes accounting for >= {args.min_share:g} of the plotted ROI "
          f"mean at some sample:")
    print("An electrode extreme in only the conditions drawn from one block "
          "type is a\nblock-locked artifact, not an effect — condition and "
          "block are confounded in a\nblocked design, so it lands on the "
          "interaction term.\n")
    header = f"{'electrode':<14}" + ''.join(f"{c[-12:]:>14}" for c in condition_order)
    print(header)
    for name in sorted(material, key=lambda n: -max(material[n].values())):
        row = ''.join(f"{material[name].get(c, float('nan')):>14.4g}"
                      for c in condition_order)
        print(f"{name:<14}{row}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
