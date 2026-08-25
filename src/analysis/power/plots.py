"""Plotting for power-trace analysis.

Per-ROI power traces (with SD/SEM/CI shading and significance bars), the 2-way
and 16-condition interaction plots, the shared plot-style/color helpers, and the
adapter that reshapes full-ANOVA cluster results into the shape the interaction
plots expect.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from typing import Union, List, Sequence

from src.analysis.utils.general_utils import _subdir
from src.analysis.config.plotting_parameters import plotting_parameters

DEFAULT_PLOT_STYLE = {
    # Toggles
    'show_title': True,
    'show_xlabel': True,
    'show_ylabel': True,
    'show_legend': True,

    # Labels
    'title': None,        # None = auto-generate from ROI name
    'x_label': 'Time (s)',
    'y_label': 'Power (z)',

    # Font sizes
    'title_font_size': 14,
    'axis_font_size': 12,
    'tick_font_size': 12,
    'legend_font_size': 10,

    # Tick customization
    'xticks': None,       # None = auto, or pass array
    'yticks': None,
    'xtick_labels': None, # Custom labels for xticks
    'ytick_labels': None,
    'xlim': None,
    'ylim': None,

    # Other
    'figsize': (12, 8),
    'text_color': '#002060',
    'sig_cluster_height': 0.3,
    
    # per electrode power traces overlaid in thin gray lines
    'show_electrode_traces': True,
    'electrode_trace_color': '#9e9e9e',
    'electrode_trace_alpha': 0.25,
    'electrode_trace_linewidth': 0.7,
    'label_outliers': True,
    'n_outlier_labels': 3,
}

def rank_electrode_deviations(evoked):
    '''
    Rank channels by their RMS deviation from the channel mean trace.
    
    RMS deviation uses the full displayed time course, so electrodes
    with a sustained offset and electrodes with a large transient excursion
    are both surfaced. NaN-only channels are retained at the bottom of the ranking.
    '''
    data = np.asarray(evoked.data, dtype=float)
    if data.ndim != 2:
        raise ValueError("evoked.data must have shape (channels, times)")
    
    with np.errstate(invalid='ignore'):
        mean_trace = np.nanmean(data, axis=0)
        scores = np.sqrt(np.nanmean((data - mean_trace) ** 2, axis=1))
    names = list(getattr(evoked, 'ch_names', []))
    
    if len(names) != data.shape[0]:
        names = [f"channel_{i}" for i in range(data.shape[0])]
    order = np.argsort(np.nan_to_num(scores, nan=-np.inf))[::-1]
    
    return [(names[i], float(scores[i])) for i in order]

def _write_electrode_deviation_report(path, roi, rankings):
    "Write a human-readable, condition-wise electrode outlier ranking"
    with open(path, 'w', encoding='utf-8') as report:
        report.write(f"Electrode deviation report for roi: {roi}\n")
        report.write(f"Metric: RMS deviation of each electrode trace from the "
                     "condition's across-electrode mean trace.\n")
        report.write("Larger values indicate greater deviation; this is a "
                     "diagnostic ranking, not a statistical outlier test. \n\n")
        for condition, ranking in rankings.items():
            report.write(f"[{condition}] ({len(ranking)} electrodes]]\n")
            report.write("rank\telectrode\trms_deviation\n")
            for rank, (electrode, score) in enumerate(ranking, start=1):
                report.write(f"{rank}\t{electrode}\t{score:.6g}\n")
            report.write("\n")
    
    
def plot_power_trace_for_roi(evks_dict, roi, condition_names, conditions_save_name,
                             plotting_parameters, significant_clusters=None,
                             window_size=None, sampling_rate=None, save_dir=None,
                             show_std=True, show_sem=False, show_ci=False, ci=0.95,
                             plot_style=None, save_name_suffix=None):
    """
    Custom plot with standard deviation or standard error shading.

    Since MNE's plot_compare_evokeds only supports confidence intervals,
    this function manually creates plots with SD or SEM shading.

    Parameters:
    -----------
    evks_dict : dict
        Dictionary with condition names as keys and evoked dictionaries as values
    roi : str
        ROI name
    condition_names : list
        List of condition names to plot
    conditions_save_name : str
        Name to use for saving the plot
    plotting_parameters : dict
        Dictionary with plotting parameters for the traces.
    save_dir : str
        Directory to save the plot
    show_std : bool
        Whether to show standard deviation shading
    show_sem : bool
        Whether to show standard error of mean shading
    plot_style : dict
        Dictionary with plot style parameters for the figure settings.
    save_name_suffix : str
        Suffix to add to the save name
    significant_clusters : array-like of bool
        A boolean array indicating which time windows are part of a
        statistically significant cluster. Shape: (n_windows,).
    Returns:
    --------
    fig : matplotlib figure
    """
    # Resolve plot style with defaults
    s = {**DEFAULT_PLOT_STYLE, **(plot_style or {})}
    figsize = s['figsize']
    sig_cluster_height = s['sig_cluster_height']
    fig, ax = plt.subplots(figsize=figsize)
    deviation_rankings = {}
    
    for condition_name in condition_names:
        evoked = evks_dict[condition_name][roi]
        if evoked is None or evoked.data.shape[0] == 0:
            continue

        # Get plotting parameters
        param_key = None

        # First try exact match
        if condition_name in plotting_parameters:
            param_key = condition_name
        else:
            # Then try to find the best match by looking for the longest matching key
            best_match_length = 0
            for key in plotting_parameters.keys():
                # Check if the key is a substring of condition_name or vice versa
                if (key in condition_name or condition_name in key):
                    # Prefer longer matches to avoid "Stimulus_c" matching when "Stimulus_c25" exists
                    if len(key) > best_match_length:
                        best_match_length = len(key)
                        param_key = key

        if param_key and param_key in plotting_parameters:
            params = plotting_parameters[param_key]
            color = params.get('color', 'black')
            linestyle = params.get('line_style', '-')
            label = params.get('condition_parameter', condition_name)
        else:
            # Default parameters if no match found
            color = 'black'
            linestyle = '-'
            label = condition_name

        # Get data
        times = evoked.times
        data = evoked.data
        n_channels = data.shape[0]

        # Calculate mean across channels
        mean_data = np.nanmean(data, axis=0)

        # draw these first so the condition mean and uncertainty stay visually 
        # dominant. Labels use the largest pointwise departure of each selected
        # trace, which puts the name next to the feature that drove the ranking.
        
        if s['show_electrode_traces']:
            ax.plot(times, data.T, color=s['electrode_trace_color'],
                    alpha=s['electrode_trace_alpha'],
                    linewidth=s['electrode_trace_linewidth'], zorder=1)
            
        ranking = rank_electrode_deviations(evoked)
        deviation_rankings[condition_name] = ranking
        if s['label_outliers'] and s['show_electrode_traces']:
            name_to_index = {name: idx for idx, name in enumerate(evoked.ch_names)}
            for electrode, _ in ranking[:max(0, int(s['n_oulier_labels']))]:
                channel_idx = name_to_index[electrode]
                delta = np.abs(data[channel_idx] - mean_data)
                if np.all(np.isnan(delta)):
                    continue
                time_idx = int(np.nanargmax(delta))
                ax.annotate(electrode,
                            (times[time_idx], data[channel_idx, time_idx]),
                            xytext=(3,3), textcoords='offset points', 
                            fontsize=max(6, s['legend_font_size'] - 2),
                            color='#555555', alpha=0.9, clip_on=True)
            
        # Plot mean
        ax.plot(times, mean_data, color=color, linestyle=linestyle,
                linewidth=2.5, label=label)

        # Add shading
        if show_std:
            std_data = np.std(data, axis=0)
            ax.fill_between(times, mean_data - std_data, mean_data + std_data,
                           alpha=0.3, color=color, linewidth=0)
        elif show_sem:
            sem_data = np.std(data, axis=0) / np.sqrt(n_channels)
            ax.fill_between(times, mean_data - sem_data, mean_data + sem_data,
                           alpha=0.3, color=color, linewidth=0)
        elif show_ci:
            ci_data = np.percentile(data, [100 * (1 - ci), 100 * ci], axis=0)
            ax.fill_between(times, ci_data[0], ci_data[1],
                           alpha=0.3, color=color, linewidth=0)

    # Find contiguous significant clusters
    def find_clusters(significant_clusters: Union[np.ndarray, List[bool], Sequence[bool]]):
        """Helper to find start and end indices of contiguous True blocks."""
        clusters = []
        in_cluster = False
        for idx, val in enumerate(list(significant_clusters)):
            if val and not in_cluster:
                # Start of a new cluster
                start_idx = idx
                in_cluster = True
            elif not val and in_cluster:
                # End of the cluster
                end_idx = idx - 1
                clusters.append((start_idx, end_idx))
                in_cluster = False
        # Handle the case where the last value is in a cluster
        if in_cluster:
            end_idx = len(list(significant_clusters)) - 1
            clusters.append((start_idx, end_idx))
        return clusters

    # logging.debug(f"--- For ROI: {roi} --- significant_clusters is: {significant_clusters}")

    if significant_clusters is not None:

        # logging.debug(f"    -> Not None. Trying to find and plot clusters for {roi}.")

        # The indices below index `times`, which is at SAMPLE resolution, so the
        # mask must be sample-resolution too. A window-level mask (one entry per
        # sliding window) silently draws the bar compressed into the first few
        # hundred ms of the epoch instead of erroring. Catch it here.
        n_mask = len(list(significant_clusters))
        if n_mask != len(times):
            raise ValueError(
                f"significant_clusters has {n_mask} entries but the time axis has "
                f"{len(times)} samples. Pass a sample-level mask (e.g. "
                f"`sample_mask` from the ANOVA cluster results), not a "
                f"window-level one — indices are used to index `times` directly."
            )

        clusters = find_clusters(significant_clusters)

        # # Determine y position for the bars
        # max_y = np.max(mean_true_accuracy + se_true_accuracy)
        # min_y = np.min(mean_shuffle_accuracy - se_shuffle_accuracy)
        # sig_cluster_height = max_y + 0.02  # Adjust as needed
        # plt.ylim([min_y, sig_cluster_height + 0.05])  # Adjust ylim to accommodate the bars

        # Plot horizontal bars and asterisks for significant clusters
        for cluster in clusters:
            start_idx, end_idx = cluster

            if window_size is None:
                window_size = 0 # set to zero for point-wise analysis

            if window_size is None or window_size == 0:
                # Point-wise analysis: Bar spans the centers of the first/last points
                start_time = times[start_idx]
                end_time = times[end_idx]
            else:
                # Windowed analysis: Bar spans the outer edges of the first/last windows
                window_duration = window_size / sampling_rate
                start_time = times[start_idx] - (window_duration / 2)
                end_time = times[end_idx] + (window_duration / 2)

            plt.hlines(y=sig_cluster_height, xmin=start_time, xmax=end_time, color='black', linewidth=8)
            # Place an asterisk at the center of the bar
            center_time = (start_time + end_time) / 2
            plt.text(center_time, sig_cluster_height + 0.01, '*', ha='center', va='bottom', fontsize=25)

    # Customize plot
    text_color = s['text_color']

    if s['show_xlabel']:
        ax.set_xlabel(s['x_label'], fontsize=s['axis_font_size'], color=text_color)
    if s['show_ylabel']:
        ax.set_ylabel(s['y_label'], fontsize=s['axis_font_size'], color=text_color)

    ax.axhline(y=0, color='black', linestyle=':', alpha=0.5)
    ax.axvline(x=0, color='black', linestyle=':', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', colors=text_color, labelsize=s['tick_font_size'])

    if s['xticks'] is not None:
        ax.set_xticks(s['xticks'])
    if s['yticks'] is not None:
        ax.set_yticks(s['yticks'])

    if s['show_title']:
        title = s['title'] if s['title'] else f'{roi.upper()}'
        ax.set_title(title, fontsize=s['title_font_size'], fontweight='bold', color=text_color)

    if s['ylim']:
        ax.set_ylim(s['ylim'])
    if s['xlim']:
        ax.set_xlim(s['xlim'])

    if s['show_legend']:
        ax.legend(loc='best', framealpha=0.95, fontsize=s.get('legend_font_size', 10))

    plt.tight_layout()

    # Save if directory provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        error_type = 'std' if show_std else 'sem' if show_sem else 'ci' if show_ci else 'no_error'
        base = f'{roi}_{conditions_save_name}_{save_name_suffix}_{error_type}_shading'
        
        for ext in ('.pdf', '.png'):
            filepath = os.path.join(save_dir, base + ext)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved plot to: {filepath}")

        report_path = os.path.join(save_dir, base + '_electrode_deviations.txt')
        _write_electrode_deviation_report(report_path, roi, deviation_rankings)
        print(f"Saved electrode deviation report to: {report_path}")
            
    plt.close()
    return fig

def plot_power_traces_for_all_rois(evks_dict_elecs, rois, condition_names, conditions_save_name,
                                   plotting_parameters, window_size=None, sampling_rate=None,
                                   significant_clusters=None, save_dir=None, error_type='std',
                                   plot_style=None, save_name_suffix=None):
    """
    Plot power traces for each ROI comparing the specified conditions

    Parameters:
    -----------
    evks_dict_elecs : dict
        Evoked objects for electrodes
    rois : list
        List of ROI names
    condition_names : list
        List of condition names
    conditions_save_name : str
        Name to use for saving the plot
    plotting_parameters : dict
        Plotting parameters dictionary (see config/plotting_parameters.py for details)
    save_dir : str
        Directory to save plots
    error_type : str
        Type of error to show: 'std', 'sem', 'ci', or 'none'
    plot_style : dict
        Dictionary with plot style parameters for the figure settings.
    save_name_suffix : str
        Suffix to add to the save name
    significant_clusters : array-like of bool
        A boolean array indicating which time windows are part of a
        statistically significant cluster. Shape: (n_windows,).
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for roi in rois:
        clusters_for_this_roi = None
        if significant_clusters is not None:
            # Look up the specific mask for this ROI
            clusters_for_this_roi = significant_clusters.get(roi, None)

        # Plot all electrodes
        plot_power_trace_for_roi(
            evks_dict_elecs, roi, condition_names, conditions_save_name,
            plotting_parameters, window_size=window_size, sampling_rate=sampling_rate,
            significant_clusters=clusters_for_this_roi, save_dir=_subdir(save_dir, roi),
            show_std=(error_type == 'std'),
            show_sem=(error_type == 'sem'),
            show_ci=(error_type == 'ci'),
            plot_style=plot_style, save_name_suffix=save_name_suffix
        )

    if save_dir:
        print(f"\nAll plots saved to: {save_dir}")
    plt.close()


def apply_plot_style(ax, roi, style=None):
    """Apply styling to an axis from a style dict, with defaults."""
    s = {**DEFAULT_PLOT_STYLE, **(style or {})}

    if s['show_title']:
        title = s['title'] if s['title'] else f"{roi.upper()}"
        ax.set_title(title, fontsize=s['title_font_size'],
                     fontweight='bold', color=s['text_color'])

    if s['show_xlabel']:
        ax.set_xlabel(s['x_label'], fontsize=s['axis_font_size'], color=s['text_color'])
    else:
        ax.set_xlabel('')

    if s['show_ylabel']:
        ax.set_ylabel(s['y_label'], fontsize=s['axis_font_size'], color=s['text_color'])
    else:
        ax.set_ylabel('')

    # Ticks
    if s['xticks'] is not None:
        ax.set_xticks(s['xticks'])
    if s['yticks'] is not None:
        ax.set_yticks(s['yticks'])
    if s['xtick_labels'] is not None:
        ax.set_xticklabels(s['xtick_labels'])
    if s['ytick_labels'] is not None:
        ax.set_yticklabels(s['ytick_labels'])

    ax.tick_params(axis='both', colors=s['text_color'], labelsize=s['tick_font_size'])

    if s['xlim']:
        ax.set_xlim(s['xlim'])
    if s['ylim']:
        ax.set_ylim(s['ylim'])

    if s['show_legend']:
        ax.legend(loc='best', framealpha=0.95, fontsize=s['legend_font_size'])

    # Standard cleanup
    ax.axhline(y=0, color='black', linestyle=':', alpha=0.5)
    ax.axvline(x=0, color='black', linestyle=':', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# =============================================================================
# Two-way interaction cluster correction across time
#
# We compute, per electrode per timepoint, the 2x2 interaction contrast
#   ((A1B1 - A1B2) - (A2B1 - A2B2))
# averaging across whatever other factors exist in the conditions_obj. We then
# use a one-sample sign-flip cluster permutation test across electrodes
# (mne.stats.permutation_cluster_1samp_test). This is mathematically equivalent
# to an F-test on the interaction term in a 2x2 ANOVA across electrodes
# (t**2 == F when df_num == 1), so the resulting cluster mask is equivalent to a
# F-statistic cluster correction.
# =============================================================================

def _get_subcell_condition_names(conditions_obj, factor1, factor2, level1, level2):
    """Find condition keys whose factor values match (level1, level2)."""
    return [
        k for k, v in conditions_obj.items()
        if v.get(factor1) == level1 and v.get(factor2) == level2
    ]

def _get_factor_levels(conditions_obj, factor):
    """Return the unique levels of 'factor' across the conditions, in stable order."""
    seen = []
    for v in conditions_obj.values():
        lvl = v.get(factor)
        if lvl is not None and lvl not in seen:
            seen.append(lvl)
    return seen

def compute_subcell_evoked_data(evks_dict, conditions_obj, factor1, factor2,
                                level1, level2, roi):
    """Average per-electrode evoked data across all condition keys that match (factor1=level1, factor2=level2).
    Other factors are collapsed by simmple average over the matching cells (equal weight per subcell).

    Returns
    -------
    arr : (n_electrodes, n_times) ndarray, or None if no matching condition has data
    """
    cond_keys = _get_subcell_condition_names(conditions_obj, factor1, factor2,
                                             level1, level2)
    arrays = []
    for k in cond_keys:
        evk = evks_dict.get(k, {}).get(roi)
        if evk is None or evk.data.shape[0] == 0:
            continue
        arrays.append(evk.data)
    if not arrays:
        return None
    return np.mean(np.stack(arrays, axis=0), axis=0)

# =============================================================================
# Plotting helpers
# =============================================================================

def _find_cluster_spans(mask):
    """Return list of (start_idx, end_idx) inclusive for contiguous True runs.""" # hmm pretty sure some version of this already exists somewhere
    spans = []
    in_run = False
    start = 0
    arr = np.asarray(mask).astype(bool)
    for i, val in enumerate(arr):
        if val and not in_run:
            start = i
            in_run = True
        elif not val and in_run:
            spans.append((start, i - 1))
            in_run = False
    if in_run:
        spans.append((start, len(arr) - 1))
    return spans

def _draw_cluster_bar(ax, times, mask, y, color='black', linewidth=6,
                      label=None, label_x=None, label_color=None,
                      label_fontsize=10):
    """Draw horizontal bar(s) wherever mask is True, at height y.""" # this code might be redundant, check plot_horizontal_bar from aaron or i'm sure i have other code that does this.
    spans = _find_cluster_spans(mask)
    for start_idx, end_idx in spans:
        ax.hlines(y=y, xmin=times[start_idx], xmax=times[end_idx],
                  color=color, linewidth=linewidth)
    if label is not None and label_x is not None:
        ax.text(label_x, y, label, ha='right', va='center',
                fontsize=label_fontsize,
                color=label_color if label_color is not None else color)

# --- Factor-pair → plotting_parameters key resolver ---
# The 4 levels of each 2-way factor pair map directly to a Stimulus_*** entry
# (e.g. (congruency='c', incongruentProportion='25%') -> 'Stimulus_c25').
# Each pair-resolver takes the factor *value* dict for a condition and returns
# the plotting_parameters key, or None if any factor is missing/unknown.

def _strip_pct(level):
    """Normalize '25%' / 25 / '25' -> '25' or '75', else None."""
    if level is None:
        return None
    s = str(level).rstrip('%')
    return s if s in {'25', '75'} else None

def _pp_key_incongruent_proportion(v):
    """(congruency, incongruentProportion) -> 'Stimulus_c25' / 'Stimulus_i75' / ...

    Now consistent: the stored incongruentProportion matches the suffix used
    in Stimulus_*** keys (both follow BIDS naming).
    """
    cong = v.get('congruency')
    prop = _strip_pct(v.get('incongruentProportion'))   # <-- no flip
    if cong in ('c', 'i') and prop in ('25', '75'):
        return f'Stimulus_{cong}{prop}'
    return None
def _pp_key_switch_proportion(v):
    """(switchType, switchProportion) -> 'Stimulus_s25' / 'Stimulus_r75' / ...

    switchProportion is NOT flipped -- both the stored value and the BIDS
    naming use the switch proportion directly.
    """
    st = v.get('switchType')
    prop = _strip_pct(v.get('switchProportion'))             # <-- no flip
    if st in ('s', 'r') and prop in ('25', '75'):
        return f'Stimulus_{st}{prop}'
    return None

def _pp_style(key, fallback=('gray', '-')):
    """Return (color, line_style) from plotting_parameters[key], with fallback."""
    if key is None or key not in plotting_parameters:
        return fallback
    p = plotting_parameters[key]
    return p.get('color', fallback[0]), p.get('line_style', fallback[1])

def _generate_16_condition_colors(condition_names, conditions_obj,
                                  factors=('congruency', 'incongruentProportion',
                                           'switchType', 'switchProportion')):
    """Style map for the 16-condition plot.

    Strategy: read directly from plotting_parameters[condition_name] when an
    entry exists. Otherwise compose a fallback style by looking up the
    (congruency, incongruentProportion) entry for color and the
    (switchType, switchProportion) entry for linestyle.
    """
    style_map = {}
    for name in condition_names:
        # Preferred path: direct entry exists in plotting_parameters.
        if name in plotting_parameters:
            p = plotting_parameters[name]
            style_map[name] = {
                'color':     p.get('color', 'gray'),
                'linestyle': p.get('line_style', '-'),
                'linewidth': p.get('linewidth', 1.8),
                'alpha':     p.get('alpha', 1.0),
            }
            continue

        # Fallback: compose from the two 2-way pair entries.
        v = conditions_obj.get(name, {})
        cong_key   = _pp_key_incongruent_proportion(v)
        switch_key = _pp_key_switch_proportion(v)
        color, _   = _pp_style(cong_key)
        _, ls      = _pp_style(switch_key)
        style_map[name] = {
            'color': color, 'linestyle': ls,
            'linewidth': 1.8, 'alpha': 1.0,
        }
    return style_map

def plot_2way_interaction_for_roi(
    evks_dict, roi, conditions_obj, factor1, factor2,
    interaction_mask, conditions_save_name, plotting_parameters=None,
    plot_style=None, save_dir=None, save_name_suffix=None, error_type='sem',
    interaction_label=None, factor_labels=None,
):
    """Plot the 4 sub-cell traces for a 2x2 interaction with a cluster bar overlay.

    Parameters
    ----------
    interaction_mask : (n_times,) bool array, or None
    """
    s = {**DEFAULT_PLOT_STYLE, **(plot_style or {})}
    fig, ax = plt.subplots(figsize=s['figsize'])

    # Figure out a time vector from any populated condition
    times = None
    for cname, roi_dict in evks_dict.items():
        evk = roi_dict.get(roi)
        if evk is not None and evk.data.shape[0] > 0:
            times = evk.times
            break
    if times is None:
        plt.close(fig)
        return None

    levels1 = _get_factor_levels(conditions_obj, factor1)
    levels2 = _get_factor_levels(conditions_obj, factor2)
    if len(levels1) != 2 or len(levels2) != 2:
        plt.close(fig)
        raise ValueError(
            f"Two-way plot expects 2 levels per factor; got "
            f"{factor1}={levels1}, {factor2}={levels2}"
        )

    # Resolve PP key per cell based on which factor pair this 2-way is.
    # If we recognize the pair, use direct PP lookups; otherwise fall back to a generic pair.
    pair = frozenset((factor1, factor2))
    if pair == frozenset(('congruency', 'incongruentProportion')):
        resolve = _pp_key_incongruent_proportion
    elif pair == frozenset(('switchType', 'switchProportion')):
        resolve = _pp_key_switch_proportion
    else:
        resolve = None

    for l1 in levels1:
        for l2 in levels2:
            data = compute_subcell_evoked_data(
                evks_dict, conditions_obj, factor1, factor2, l1, l2, roi
            )
            if data is None:
                continue
            mean_data = np.mean(data, axis=0)
            if resolve is not None:
                key = resolve({factor1: l1, factor2: l2})
                color, ls = _pp_style(key)
            else:
                # Cross-factor 2-way (e.g. congruency x switchProportion) -- no PP entry.
                color = ('#1f77b4' if l1 == levels1[0] else '#d62728')
                ls = '-' if l2 == levels2[0] else '--'
            label = f"{factor1}={l1}, {factor2}={l2}"
            ax.plot(times, mean_data, color=color, linestyle=ls,
                    linewidth=2.5, label=label)
            # ... (SEM shading block unchanged)
            n = data.shape[0]
            if error_type == 'sem' and n > 1:
                err = np.std(data, axis=0) / np.sqrt(n)
            elif error_type == 'std':
                err = np.std(data, axis=0)
            else:
                err = None
            if err is not None:
                ax.fill_between(times, mean_data - err, mean_data + err,
                                color=color, alpha=0.15, linewidth=0)

    # Overlay cluster bar at top of plot
    # === Sign-aware cluster bars: positive (red) above, negative (blue) below. ===
    if interaction_mask is not None:
        # interaction_mask may be a (n_times,) bool array (back-compat) or a dict
        # {'pos': bool[n_times], 'neg': bool[n_times]} from the sign-aware path.
        ylim = ax.get_ylim() if s.get('ylim') is None else s['ylim']
        span = ylim[1] - ylim[0]
        bar_y_top = s.get('sig_cluster_height', ylim[1] - span * 0.04)
        bar_y_step = span * 0.04

        if isinstance(interaction_mask, dict):
            bar_specs = [
                (interaction_mask.get('pos'),  '#D62728', 0, '+ direction'),
                (interaction_mask.get('neg'),  '#1F77B4', 1, '- direction'),
            ]
        else:
            bar_specs = [(interaction_mask, 'black', 0, None)]

        for mask, color, offset, label in bar_specs:
            if mask is None or not np.any(mask):
                continue
            y_here = bar_y_top - offset * bar_y_step
            _draw_cluster_bar(ax, times, mask, y=y_here, color=color, linewidth=8)
            center_idx = np.where(mask)[0]
            if len(center_idx) and label:
                ax.text(times[int(np.median(center_idx))], y_here + span * 0.005,
                        '*', ha='center', va='bottom', fontsize=18, color=color)

    apply_plot_style(ax, roi, plot_style)
    # Always show the 4-trace legend on per-interaction plots — the 4 cells are
    # otherwise indistinguishable. This overrides plot_style['show_legend']
    # because for these plots the legend is essential, not optional.
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc='lower right', framealpha=0.9,
                  fontsize=s.get('legend_font_size', 10))
    if interaction_label and s.get('show_title', True):
        ax.set_title(f"{roi.upper()} — {interaction_label}",
                     fontsize=s['title_font_size'], fontweight='bold',
                     color=s['text_color'])
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        suf = save_name_suffix or ''
        base = (f'{roi}_{conditions_save_name}_2way_{factor1}_x_{factor2}_'
                f'{suf}_{error_type}_shading')
        for ext in ('.pdf', '.png'):
            filepath = os.path.join(save_dir, base + ext)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved plot to: {filepath}")
    plt.close(fig)
    return fig

def plot_16_conditions_with_interaction_clusters_for_roi(
    evks_dict, roi, conditions_obj, condition_names, conditions_save_name,
    interaction_results,  # dict[interaction_name] -> result dict
    anova_interactions,
    plot_style=None, save_dir=None, save_name_suffix=None, error_type='sem',
):
    """Plot all 16 condition power traces with 4 stacked horizontal cluster bars
    (one per 2-way interaction) overlaid at the top of the panel.
    """
    s = {**DEFAULT_PLOT_STYLE, **(plot_style or {})}
    fig, ax = plt.subplots(figsize=s['figsize'])

    times = None
    colors = _generate_16_condition_colors(condition_names, conditions_obj)

    default_style = {'color': 'black', 'linestyle': '-', 'linewidth': 1.5, 'alpha': 1.0}
    for cname in condition_names:
        evk = evks_dict.get(cname, {}).get(roi)
        if evk is None or evk.data.shape[0] == 0:
            continue
        if times is None:
            times = evk.times
        data = evk.data
        mean_data = np.mean(data, axis=0)
        n = data.shape[0]
        style = colors.get(cname, default_style)
        ax.plot(times, mean_data,
                color=style['color'], linestyle=style['linestyle'],
                linewidth=style['linewidth'], alpha=style['alpha'],
                label=cname.replace('Stimulus_', ''))
        if error_type == 'sem' and n > 1:
            err = np.std(data, axis=0) / np.sqrt(n)
            ax.fill_between(times, mean_data - err, mean_data + err,
                            color=style['color'],
                            alpha=0.10 * style['alpha'], linewidth=0)

    if times is None:
        plt.close(fig)
        return None

    # === Per-interaction sign-aware bars: red for +, blue for -. Stack vertically. ===
    ylim = s.get('ylim') if s.get('ylim') is not None else ax.get_ylim()
    y_top = ylim[1]; y_bottom = ylim[0]
    span = y_top - y_bottom
    bar_band_top = y_top - span * 0.02
    bar_band_bottom = y_top - span * 0.20  # slightly wider to fit 2 bars per interaction
    n_inter = max(len(anova_interactions), 1)
    n_bar_slots = 2 * n_inter  # one slot per (interaction, sign)
    bar_ys = np.linspace(bar_band_top, bar_band_bottom, n_bar_slots)

    POS_COLOR = '#D62728'   # red:  contrast direction +
    NEG_COLOR = '#1F77B4'   # blue: contrast direction -

    res_for_roi = interaction_results.get(roi, {})
    bar_legend_handles = []
    bar_legend_labels = []
    slot = 0
    for inter in anova_interactions:
        info = res_for_roi.get(inter['name'])
        if info is None:
            slot += 2
            continue
        pos_mask = info.get('pos_sample_mask', info.get('pos_window_mask'))
        neg_mask = info.get('neg_sample_mask', info.get('neg_window_mask'))
        label_base = info.get('label', inter['name']) if isinstance(info, dict) else inter['name']

        for mask, color, sign_label in [(pos_mask, POS_COLOR, '+'),
                                        (neg_mask, NEG_COLOR, '-')]:
            if mask is not None and np.any(mask):
                _draw_cluster_bar(ax, times, mask, y=bar_ys[slot],
                                  color=color, linewidth=7,
                                  label=None, label_x=None)
                bar_legend_handles.append(Line2D([0], [0], color=color, linewidth=7))
                bar_legend_labels.append(f'{label_base} ({sign_label})')
            slot += 1

    apply_plot_style(ax, roi, plot_style)
    if s.get('ylim') is None:
        ax.set_ylim(y_bottom, y_top)

    # Two legends: the 16-condition trace legend (gated on plot_style.show_legend
    # because it's huge and often obscures the data) and the cluster-bar legend
    # (always shown — the bars are uninterpretable without it).
    trace_legend = None
    if s.get('show_legend', True):
        trace_legend = ax.legend(loc='lower left', framealpha=0.9, ncol=2,
                                 fontsize=max(6, s.get('legend_font_size', 10) - 2))
    if bar_legend_handles:
        if trace_legend is not None:
            ax.add_artist(trace_legend)  # keep both legends visible
        ax.legend(bar_legend_handles, bar_legend_labels,
                  loc='lower right', framealpha=0.9,
                  title='2-way interaction clusters',
                  fontsize=s.get('legend_font_size', 10))

    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        suf = save_name_suffix or ''
        base = (f'{roi}_{conditions_save_name}_16cond_with_interaction_clusters_'
                f'{suf}_{error_type}_shading')
        for ext in ('.pdf', '.png'):
            filepath = os.path.join(save_dir, base + ext)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved plot to: {filepath}")
    plt.close(fig)
    return fig

def plot_anova_interaction_results(
    evks_dict, conditions_obj, condition_names, conditions_save_name,
    rois, anova_interactions, interaction_results,
    plot_style=None, save_dir=None, save_name_suffix=None, error_type='sem',
):
    """Convenience wrapper: for each ROI, draw the 16-condition mega-plot AND
    one 4-trace plot per 2-way interaction.
    """
    for roi in rois:
        plot_16_conditions_with_interaction_clusters_for_roi(
            evks_dict, roi, conditions_obj, condition_names, conditions_save_name,
            interaction_results, anova_interactions,
            plot_style=plot_style, save_dir=_subdir(save_dir, roi),
            save_name_suffix=save_name_suffix, error_type=error_type,
        )
        for inter in anova_interactions:
            f1, f2 = inter['factors']
            info = interaction_results.get(roi, {}).get(inter['name'])
            if info is not None and ('pos_sample_mask' in info or 'pos_window_mask' in info):
                # Prefer sample-resolution masks if both shapes are available
                mask_arg = {
                    'pos': info.get('pos_sample_mask', info.get('pos_window_mask')),
                    'neg': info.get('neg_sample_mask', info.get('neg_window_mask')),
                }
            else:
                mask_arg = info['mask'] if info is not None else None
            plot_2way_interaction_for_roi(
                evks_dict, roi, conditions_obj, f1, f2, mask_arg,
                conditions_save_name,
                plot_style=plot_style, save_dir=_subdir(save_dir, roi),
                save_name_suffix=save_name_suffix, error_type=error_type,
                interaction_label=inter.get('label'),
            )

def anova_results_to_interaction_results_for_plotting(
    anova_cluster_results, anova_interactions
):
    """Adapt the full-ANOVA result dict into the shape my mega-plot expects.

    The mega-plot's interaction_results structure is:
        dict[roi][interaction_name] -> {'mask', 'label', 'factors', ...}
    where `mask` is the *sample-level* boolean array.
    """
    out = {}
    for roi, by_effect in anova_cluster_results.items():
        out[roi] = {}
        for inter in anova_interactions:
            f1, f2 = inter['factors']
            # statsmodels names the interaction term as 'C(f1):C(f2)' regardless
            # of which factor is listed first in the formula
            candidate_names = [f'C({f1}):C({f2})', f'C({f2}):C({f1})']
            info = None
            for nm in candidate_names:
                if nm in by_effect:
                    info = by_effect[nm]
                    break
            if info is None:
                continue
            out[roi][inter['name']] = {
                'mask': info['sample_mask'],
                # Carry the signed masks through. Without them the caller falls
                # back to the union `mask` and draws ONE black bar, so a positive
                # cluster abutting a negative one renders as a single continuous
                # span — which reads as one long effect rather than two opposite
                # ones. The plotting code already has a sign-aware branch; it was
                # just never reachable from this adapter.
                'pos_sample_mask': info.get('pos_sample_mask'),
                'neg_sample_mask': info.get('neg_sample_mask'),
                'signed_contrast': info.get('signed_contrast'),
                't_obs': info['observed_F'],   # F-trace, but field name kept for compat
                'cluster_p_values': np.array([]),  # not exposed by the percentile method
                'factors': (f1, f2),
                'levels': (_get_factor_levels({}, f1) or [], _get_factor_levels({}, f2) or []),
                'label': inter.get('label', inter['name']),
            }
    return out
