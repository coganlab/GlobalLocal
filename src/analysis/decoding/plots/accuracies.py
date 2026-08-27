"""Nature-style plotting of decoding accuracy time courses."""

import matplotlib
matplotlib.use('Agg')
import os
import numpy as np
from scipy.ndimage import label
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from typing import Union, List, Sequence
from typing import Dict, Optional, Union, List, Tuple

from ..accuracy_stats import find_contiguous_clusters
from ..plots.style import NATURE_STYLE, nature_style

def plot_accuracies_nature_style(
    time_points: np.ndarray,
    accuracies_dict: Dict[str, np.ndarray],
    significant_clusters: Optional[np.ndarray] = None,
    window_size: Optional[int] = None,
    step_size: Optional[int] = None,
    sampling_rate: float = 256,
    comparison_name: str = "",
    roi: str = "",
    save_dir: str = ".",
    timestamp: Optional[str] = None,
    p_thresh: float = 0.05,
    colors: Optional[Dict[str, str]] = None,
    linestyles: Optional[Dict[str, str]] = None,
    title: Optional[str] = None,
    ylabel: str = "Accuracy",
    ylim: Tuple[float, float] = (0.0, 1.0),  # CHANGED: Default Y-axis is now 0 to 1
    xlim: Tuple[float, float] = (-1.0, 1.5),
    single_column: bool = True,
    show_legend: bool = True,
    show_significance: bool = True,
    significance_y_position: Optional[float] = None,
    filename_suffix: str = "",
    return_fig: bool = False,
    show_chance_level: bool = True,
    chance_level: float = 0.5,
    samples_axis=0
):
    """
    Plot accuracies in Nature journal style.
    
    Follows Nature's guidelines:
    - Single column width: 89mm (3.5 inches)
    - Double column width: 183mm (7.2 inches)
    - Font: Arial or Helvetica
    - Font sizes: 7pt for labels, 6pt for tick labels
    - Minimal design with no unnecessary elements
    - High contrast colors suitable for print
    """
    
    # Apply Nature style settings
    with plt.rc_context(NATURE_STYLE):
        # Set figure size based on column width
        if single_column:
            fig_width = 89 / 25.4  # 89mm to inches
            fig_height = fig_width * 0.7  # Aspect ratio
        else:
            fig_width = 183 / 25.4 # 183mm to inches
            fig_height = fig_width * 0.4
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        # Define Nature-appropriate colors if not provided
        if colors is None:
            nature_colors = [
                '#0173B2',  # Blue
                '#DE8F05',  # Orange
                '#029E73',  # Green
                '#CC78BC',  # Light purple
                '#ECE133',  # Yellow
                '#56B4E9',  # Light blue
                '#F0E442',  # Light yellow
                '#949494',  # Gray
            ]
            colors = {}
            for i, label in enumerate(accuracies_dict.keys()):
                colors[label] = nature_colors[i % len(nature_colors)]
        
        # Plot each accuracy time series
        for i, (label, accuracies) in enumerate(accuracies_dict.items()):
            # Compute statistics
            if accuracies.ndim == 2:
                n_samples = accuracies.shape[samples_axis]
                mean_accuracy = np.mean(accuracies, axis=samples_axis)
                std_accuracy = np.std(accuracies, axis=samples_axis)
                sem_accuracy = std_accuracy / np.sqrt(n_samples)
            else:
                mean_accuracy = accuracies
                std_accuracy = np.zeros_like(accuracies)
                sem_accuracy = np.zeros_like(accuracies) # huhh? why is this zero???
                print(f"⚠️ Warning: Accuracy data for '{label}' is 1D. Cannot compute SEM; plotting without error bars.")

            # Get color and linestyle
            color = colors.get(label, '#0173B2') if colors else '#0173B2'
            linestyle = linestyles.get(label, '-') if linestyles else '-'
            
            # Plot mean line with higher contrast
            ax.plot(time_points, mean_accuracy, 
                    label=label, 
                    color=color, 
                    linestyle=linestyle,
                    linewidth=1,
                    zorder=3)
            
            # Plot SEM as shaded area
            ax.fill_between(
                time_points,
                mean_accuracy - std_accuracy,
                mean_accuracy + std_accuracy,
                alpha=0.25,  # Lighter shading for Nature style
                color=color,
                linewidth=0,
                zorder=1
            )
        
        # Add chance level line if requested
        if show_chance_level:
            ax.axhline(y=chance_level, 
                        color='#666666', 
                        linestyle='--', 
                        linewidth=0.5,
                        zorder=2,
                        label='Chance')
        # Add stimulus onset line
        ax.axvline(x=0, 
                    color='#666666', 
                    linestyle='-', 
                    linewidth=0.5,
                    alpha=0.5,
                    zorder=2)
        
        # Add significance markers
        if show_significance and significant_clusters is not None and np.any(significant_clusters):
            clusters = find_contiguous_clusters(significant_clusters)
            
            # CHANGED: Position significance bars at a fixed high point (e.g., 95% of the y-axis)
            if significance_y_position is None:
                y_range = ylim[1] - ylim[0]
                # Place the bar at 95% of the total y-axis height, well above the data
                significance_y_position = ylim[0] + y_range * 0.95
            
            for start_idx, end_idx in clusters:
                
                start_time = time_points[start_idx]
                end_time = time_points[end_idx]
                
                # Draw significance bar
                ax.plot([start_time, end_time], 
                        [significance_y_position, significance_y_position],
                        color='black', 
                        linewidth=1,
                        solid_capstyle='butt')
                
                # Add significance marker
                center_time = (start_time + end_time) / 2
                ax.text(center_time, 
                        significance_y_position + (ylim[1] - ylim[0]) * 0.02, # Position asterisk just above bar
                        '*', 
                        ha='center', 
                        va='bottom', 
                        fontsize=8,
                        fontweight='bold')
        
        # Set labels
        # changed to size 12 font
        # changed to response
        # ax.set_xlabel('Time from response onset (s)', fontsize=12)
        ax.set_xlabel('Time from stimulus onset (s)', fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        
        # Set axis limits
        ax.set_ylim(ylim)
        ax.set_xlim(xlim)
        
        # Configure ticks
        # changed to size 10 font
        ax.tick_params(axis='both', which='major', labelsize=10, width=0.5, length=2)
        
        # Set specific x-ticks for clarity
        # changed to 1.0
        x_ticks = np.arange(-1.0, 1.6, 0.5)
        ax.set_xticks(x_ticks)
        
        # CHANGED: Set specific y-ticks for consistency
        y_ticks = np.linspace(ylim[0], ylim[1], num=5) 
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f'{y:.2f}' for y in y_ticks])

        # Configure legend
        if show_legend:
            # The legend is placed in the upper right. With the new ylim, it should have enough space.
            legend = ax.legend(
                loc='upper right',
                fontsize=6,
                frameon=False,
                handlelength=1,
                handletextpad=0.5,
                borderpad=0.2,
                columnspacing=0.5
            )
            if len(accuracies_dict) > 1: # Only make lines thicker if there's more than just the chance line
                # Make legend lines thicker for visibility
                for line in legend.get_lines():
                    line.set_linewidth(1.5)

        # Add title only if specified
        if title:
            ax.set_title(title, fontsize=7, pad=3)
        
        # Remove top and right spines (Nature style)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Make remaining spines thinner
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)
        
        # Tight layout
        plt.tight_layout(pad=0.5)
        
        if return_fig:
            return fig
        else:
            # Create filename
            timestamp_str = f"{timestamp}_" if timestamp else ""
            
            filename_parts = [timestamp_str.rstrip('_')]
            if comparison_name:
                filename_parts.append(comparison_name)
            if roi:
                filename_parts.append(roi)
            if filename_suffix:
                filename_parts.append(filename_suffix)
            
            filename = "_".join(filter(None, filename_parts)) + ".pdf"  # PDF for publication
            filepath = os.path.join(save_dir, filename)
            
            # Ensure save directory exists
            os.makedirs(save_dir, exist_ok=True)
            
            # Save in multiple formats
            plt.savefig(filepath, format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.05)
            plt.savefig(filepath.replace('.pdf', '.png'), format='png', dpi=300, bbox_inches='tight', pad_inches=0.05)
            plt.savefig(filepath.replace('.pdf', '.eps'), format='eps', dpi=300, bbox_inches='tight', pad_inches=0.05)
            
            plt.close(fig)
            print(f"Saved Nature-style plot to: {filepath}")


# Convenience function to create multi-panel Nature figures
def create_multipanel_nature_figure(
    panels_data: List[Dict],
    panel_labels: List[str] = None,
    n_cols: int = 2,
    save_path: str = None,
    fig_title: str = None
):
    """
    Create a multi-panel figure in Nature style.
    
    Parameters
    ----------
    panels_data : List[Dict]
        List of dictionaries, each containing data for one panel.
        Each dict should have keys matching plot_accuracies_nature_style parameters.
    panel_labels : List[str], optional
        Labels for each panel (e.g., ['a', 'b', 'c', 'd']).
    n_cols : int
        Number of columns in the figure grid.
    save_path : str, optional
        Path to save the figure.
    fig_title : str, optional
        Overall figure title.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The complete multi-panel figure.
    """
    
    n_panels = len(panels_data)
    n_rows = (n_panels + n_cols - 1) // n_cols
    
    with plt.rc_context(NATURE_STYLE):
        # Full page width for Nature
        fig_width = 183 / 25.4  # 183mm to inches
        fig_height = fig_width * (n_rows / n_cols) * 0.7
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
        
        if n_panels == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, (panel_data, ax) in enumerate(zip(panels_data, axes)):
            plt.sca(ax)
            
            # Add panel label
            if panel_labels and i < len(panel_labels):
                ax.text(-0.15, 1.05, panel_labels[i], 
                       transform=ax.transAxes,
                       fontsize=8, fontweight='bold',
                       va='top', ha='right')
            
            # Plot the panel (simplified - you'd need to adapt the plotting code)
            # This is a placeholder for the actual plotting logic
            
        # Remove empty subplots
        for i in range(n_panels, len(axes)):
            fig.delaxes(axes[i])
        
        if fig_title:
            fig.suptitle(fig_title, fontsize=8, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
            plt.savefig(save_path.replace('.pdf', '.png'), format='png', dpi=300, bbox_inches='tight')
        
        return fig


def plot_true_vs_shuffle_accuracies(time_points, accuracies_true, accuracies_shuffle, significant_clusters,
                    window_size, step_size, sampling_rate, condition_comparison, roi, save_dir, timestamp=None, p_thresh=0.05, other_string_to_add=None):
    """
    Plot mean true and shuffled accuracies over time with significance.

    This function visualizes the average decoding accuracy from true labels and
    shuffled labels across different time windows. It highlights significant
    time clusters (where true accuracy is significantly higher than shuffled)
    with horizontal bars and asterisks. The plot is saved to a file.

    Parameters
    ----------
    time_points : array-like
        The center time points (in seconds) for each window.
    accuracies_true : numpy.ndarray
        Accuracies for true labels. Shape: (n_windows, n_repeats).
    accuracies_shuffle : numpy.ndarray
        Accuracies for shuffled labels. Shape: (n_windows, n_perm).
    significant_clusters : array-like of bool
        A boolean array indicating which time windows are part of a
        statistically significant cluster. Shape: (n_windows,).
    window_size_samples : int
        The size of the decoding window in samples.
    step_size_samples : int
        The step size of the decoding window in samples. (Not directly used in plot rendering logic beyond filename).
    sampling_rate : float
        The sampling rate of the data in Hz.
    condition_comparison : str
        A string describing the condition comparison (e.g., "TaskA_vs_TaskB").
        Used in the plot title and filename.
    roi : str
        The Region of Interest (ROI) being plotted. Used in the plot title
        and filename.
    save_dir : str
        The directory where the plot image will be saved.
    first_time_point_s : float, optional
        The time in seconds of the first sample of the epoch, used for x-axis limits
        if needed, though current xlim are fixed. Default is 0.
    timestamp : str
        Timestamp string for filenaming purposes
    p_thresh : float
        p-value threshold for determining significant clusters
    """
    n_repeats = accuracies_true.shape[1]
    n_perm = accuracies_shuffle.shape[1]

    # Compute mean and standard error
    mean_true_accuracy = np.mean(accuracies_true, axis=1)
    std_true_accuracy = np.std(accuracies_true, axis=1)
    se_true_accuracy = std_true_accuracy / np.sqrt(n_repeats)

    mean_shuffle_accuracy = np.mean(accuracies_shuffle, axis=1)
    std_shuffle_accuracy = np.std(accuracies_shuffle, axis=1)
    se_shuffle_accuracy = std_shuffle_accuracy / np.sqrt(n_perm)

    # Plotting
    plt.figure(figsize=(12, 8))
    plt.plot(time_points, mean_true_accuracy, label='True Accuracy', color='blue')
    plt.fill_between(
        time_points,
        mean_true_accuracy - se_true_accuracy,
        mean_true_accuracy + se_true_accuracy,
        alpha=0.2,
        color='blue'
    )

    plt.plot(time_points, mean_shuffle_accuracy, label='Shuffled Accuracy', color='red')
    plt.fill_between(
        time_points,
        mean_shuffle_accuracy - se_shuffle_accuracy,
        mean_shuffle_accuracy + se_shuffle_accuracy,
        alpha=0.2,
        color='red'
    )

    # Compute window duration
    window_duration = window_size / sampling_rate

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

    clusters = find_clusters(significant_clusters)

    # # Determine y position for the bars
    # max_y = np.max(mean_true_accuracy + se_true_accuracy)
    # min_y = np.min(mean_shuffle_accuracy - se_shuffle_accuracy)
    # y_bar = max_y + 0.02  # Adjust as needed
    # plt.ylim([min_y, y_bar + 0.05])  # Adjust ylim to accommodate the bars

    # Set y_bar to a fixed value within the y-axis limits
    y_bar = 0.95  # Fixed value near the top of the y-axis

    # Plot horizontal bars and asterisks for significant clusters
    for cluster in clusters:
        start_idx, end_idx = cluster
        start_time = time_points[start_idx] - (window_duration / 2)
        end_time = time_points[end_idx] + (window_duration / 2)
        plt.hlines(y=y_bar, xmin=start_time, xmax=end_time, color='black', linewidth=2)
        # Place an asterisk at the center of the bar
        center_time = (start_time + end_time) / 2
        plt.text(center_time, y_bar + 0.01, '*', ha='center', va='bottom', fontsize=14)

    # Set axis limits
    plt.ylim(0, 1)  # Y-axis limits
    plt.xlim(-1, 1.5)  # X-axis limits

    plt.xlabel('Time from Stim Onset (s)')
    plt.ylabel('Accuracy')
    plt.title(f'Decoding Accuracy over Time for {condition_comparison} in ROI {roi}')
    plt.legend()
    
    # CREATE TIMESTAMP PREFIX
    timestamp_str = f"{timestamp}_" if timestamp else ""

    # CREATE P THRESH PREFIX
    p_thresh_str = str(p_thresh)
    
    # ADD other string if provided
    other_str = f"_{other_string_to_add}" if other_string_to_add else ""
    
    # Construct the filename
    filename = (f"{timestamp_str}{condition_comparison}_ROI_{roi}_window{window_size}_step{step_size}_"
                f"{n_repeats}_repeats_{n_perm}_perm_{p_thresh_str}_p_thresh{other_str}.png") 
    filepath = os.path.join(save_dir, filename)

    # Ensure save_dir exists
    os.makedirs(save_dir, exist_ok=True)

    # Save and close the plot
    plt.savefig(filepath)
    plt.close()


# ---------------------------------------------------------------------------
# Shared layout helpers
# ---------------------------------------------------------------------------

def _mean_and_spread(accuracies, band, samples_axis=0):
    """Mean trace and the half-width of its shaded band.

    ``band`` is ``'std'`` (spread of the bootstrap samples, what these figures
    have always shaded), ``'sem'`` (uncertainty of the mean line itself) or
    ``None``.
    """
    accuracies = np.asarray(accuracies, dtype=float)
    if accuracies.ndim == 1:
        return accuracies, np.zeros_like(accuracies)
    mean = np.nanmean(accuracies, axis=samples_axis)
    if band is None:
        return mean, np.zeros_like(mean)
    std = np.nanstd(accuracies, axis=samples_axis)
    if band == 'std':
        return mean, std
    if band == 'sem':
        return mean, std / np.sqrt(max(accuracies.shape[samples_axis], 1))
    raise ValueError(f"band must be 'std', 'sem' or None, got {band!r}")


def _cluster_spans(mask, time_points, xlim=None):
    """Time spans of each contiguous significant cluster.

    Each span is padded by half a time step on both sides so it covers the
    windows it was computed from: without the padding a single-window cluster
    draws a zero-length line and vanishes from the figure. Spans are clipped
    to ``xlim`` so the padding can never make a bar overhang the panel.
    """
    if mask is None or not np.any(mask):
        return []
    time_points = np.asarray(time_points, dtype=float)
    half_step = float(np.median(np.diff(time_points))) / 2.0 if time_points.size > 1 else 0.0
    spans = []
    for start_idx, end_idx in find_contiguous_clusters(mask):
        start = time_points[start_idx] - half_step
        end = time_points[end_idx] + half_step
        if xlim is not None:
            start, end = max(start, xlim[0]), min(end, xlim[1])
        if end > start:
            spans.append((start, end))
    return spans


def _nice_ticks(lo, hi, target=5):
    """Round tick values spanning ``[lo, hi]``.

    ``linspace(0.3, 0.8, 5)`` gives 0.30, 0.42, 0.55, 0.68, 0.80 -- five
    labels nobody can read off a trace. This picks the roundest step landing
    near ``target`` ticks instead, so the same range gives 0.3 ... 0.8.
    """
    lo, hi = float(lo), float(hi)
    if hi <= lo:
        return np.array([lo])

    def ticks_for(step):
        first = np.ceil(lo / step - 1e-9) * step
        return np.round(np.arange(first, hi + step * 1e-6, step), 10)

    magnitude = 10.0 ** np.floor(np.log10(hi - lo))
    candidates = [m * magnitude for m in (0.1, 0.2, 0.25, 0.5, 1, 2, 2.5, 5)]
    # Closest to the requested tick count; on a tie take the rounder (larger) step.
    return ticks_for(min(candidates, key=lambda s: (abs(len(ticks_for(s)) - target), -s)))


def _tick_label(value):
    """Shortest faithful decimal for a tick (0.3 rather than 0.30)."""
    text = f'{value:.4f}'.rstrip('0').rstrip('.')
    return text if text not in ('', '-0') else '0'


def _xticks(xlim, step):
    """Ticks at multiples of ``step`` within ``xlim``."""
    if not step or step <= 0:
        return None
    first = np.ceil(xlim[0] / step - 1e-9) * step
    return np.round(np.arange(first, xlim[1] + step * 1e-6, step), 10)


def plot_accuracies_with_multiple_sig_clusters(
    time_points: np.ndarray,
    accuracies_dict: Dict[str, np.ndarray],
    significance_clusters_dict: Dict[str, Dict],
    window_size: Optional[int] = None,
    step_size: Optional[int] = None,
    sampling_rate: float = 256,
    comparison_name: str = "",
    roi: str = "",
    save_dir: str = ".",
    timestamp: Optional[str] = None,
    p_thresh: float = 0.05,
    colors: Optional[Dict[str, str]] = None,
    linestyles: Optional[Dict[str, str]] = None,
    title: Optional[str] = None,
    ylabel: str = "Accuracy",
    ylim: Tuple[float, float] = (0.0, 1.0),
    xlim: Optional[Tuple[float, float]] = None,
    single_column: bool = True,
    show_legend: bool = True,
    show_chance_level: bool = True,
    chance_level: float = 0.5,
    filename_suffix: str = "",
    return_fig: bool = False,
    samples_axis: int = 0,
    sig_bar_colors: Optional[Dict[str, str]] = None,
    sig_bar_labels: Optional[Dict[str, str]] = None,
    show_sig_legend: bool = False,
    sig_marker_style: Optional[str] = None,
    # Layout / styling
    xlabel: str = 'Time from stimulus onset (s)',
    base_fontsize: float = 9.0,
    figsize: Optional[Tuple[float, float]] = None,
    band: Optional[str] = 'std',
    band_alpha: float = 0.18,
    linewidth: float = 1.8,
    xtick_step: float = 0.5,
    n_yticks: int = 5,
    sig_band_frac: float = 0.16,
    contrast_bar_linewidth: float = 3.0,
    chance_bar_linewidth: float = 1.6,
    chance_bar_color: Optional[str] = None,
    legend_loc: str = 'lower left',
    sig_legend_loc: str = 'lower right',
    sig_legend_title: Optional[str] = None,
    chance_bar_label: str = '> chance',
    formats: Sequence[str] = ('png', 'pdf'),
    # Superseded by the automatic significance strip; accepted so existing
    # callers keep working, but no longer used to place the bars.
    sig_bar_height: Optional[float] = None,
    sig_bar_spacing: Optional[float] = None,
    sig_bar_base_position: Optional[float] = None,
):
    """Plot accuracy traces with significance bars in a strip above the data.

    Significance bars are stacked in a reserved strip at the top of the axes
    rather than dropped on top of the traces, so a bar can never land on the
    data and the caller no longer has to hand-tune a y position per figure.
    Rows run top to bottom in the order the entries appear, except that all
    entries marked ``kind='chance'`` are pushed below the between-condition
    contrasts -- the tests against chance sit under the test that compares the
    two traces.

    Parameters
    ----------
    time_points : np.ndarray
        Time-window centres, in seconds.
    accuracies_dict : dict
        ``{legend label: accuracies}``, each ``(n_samples, n_time)`` or
        ``(n_time,)``. Use short display labels here, not internal comparison
        keys: whatever is passed is what the legend shows.
    significance_clusters_dict : dict
        ``{name: info}``. ``info`` is a boolean mask, or a dict with

        ``clusters``   boolean mask over ``time_points`` -- required
        ``label``      legend label; omit to draw the bar without a legend entry
        ``color``      bar colour, defaults to the matching trace's colour
        ``linestyle``  ``'-'`` / ``'--'`` -- use one of each for the two
                       directions of a contrast, so they read apart in print
        ``kind``       ``'contrast'`` (default) or ``'chance'``
        ``marker``     text drawn above the bar; usually unnecessary now that
                       the bars are in a legend
    colors, linestyles : dict, optional
        Keyed by the labels of ``accuracies_dict``.
    band : {'std', 'sem', None}
        What the shaded area around each trace shows. Default ``'std'``, which
        is what these figures have always drawn.
    sig_band_frac : float
        Fraction of the axes height reserved above the data for the
        significance strip. The data keeps exactly ``ylim``; the axes are
        extended upward, and traces and error bands are clipped at ``ylim[1]``
        so nothing bleeds into the strip.
    chance_bar_color : str, optional
        Colour for every against-chance bar. ``None`` (default) draws each in
        its own trace's colour, which ties the bar to its line without
        spending a legend entry per bar; pass e.g. ``'black'`` to subordinate
        them all to one neutral colour instead, accepting that the bars then
        no longer say which trace they belong to.
    sig_marker_style : str, optional
        Text (e.g. ``'*'``) drawn above each bar. Defaults to ``None``: with a
        significance legend the asterisks were redundant ink.
    sig_bar_height, sig_bar_spacing, sig_bar_base_position
        Superseded by the automatic strip layout. Accepted and ignored.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Only when ``return_fig`` is True; otherwise the figure is saved to
        ``save_dir`` in every entry of ``formats`` and closed.
    """
    time_points = np.asarray(time_points, dtype=float)
    n_time = time_points.size
    colors = colors or {}
    linestyles = linestyles or {}

    if figsize is None:
        width = (89 if single_column else 183) / 25.4
        figsize = (width, width * (0.8 if single_column else 0.45))

    # --- normalise the significance entries ------------------------------
    sig_entries = []
    for name, info in (significance_clusters_dict or {}).items():
        info = dict(info) if isinstance(info, dict) else {'clusters': info}
        mask = info.get('clusters')
        if mask is None:
            continue
        mask = np.asarray(mask).ravel().astype(bool)
        if mask.size != n_time:
            raise ValueError(
                f"significance mask '{name}' has {mask.size} points but there "
                f"are {n_time} time points"
            )
        if not mask.any():
            continue
        label = info.get('label', name)
        if sig_bar_labels and name in sig_bar_labels:
            label = sig_bar_labels[name]
        color = info.get('color')
        if sig_bar_colors and name in sig_bar_colors:
            color = sig_bar_colors[name]
        sig_entries.append({
            'clusters': mask,
            'label': label,
            'color': color or 'black',
            'linestyle': info.get('linestyle', '-'),
            'kind': info.get('kind', 'contrast'),
            'marker': info.get('marker', sig_marker_style),
        })

    # Contrast bars on top, against-chance bars underneath them.
    contrast_entries = [e for e in sig_entries if e['kind'] != 'chance']
    chance_entries = [e for e in sig_entries if e['kind'] == 'chance']
    # The two directions of one contrast are mutually exclusive in time, so
    # they share the top row and read as one test. Two conditions can both
    # beat chance at the same moment, so those get a row each.
    rows = ([{'entries': contrast_entries, 'kind': 'contrast'}] if contrast_entries else [])
    rows += [{'entries': [e], 'kind': 'chance'} for e in chance_entries]

    with plt.rc_context(nature_style(base_fontsize)):
        fig, ax = plt.subplots(figsize=figsize)

        # --- traces ------------------------------------------------------
        data_artists = []
        for i, (label, accuracies) in enumerate(accuracies_dict.items()):
            mean, spread = _mean_and_spread(accuracies, band, samples_axis)
            if np.asarray(accuracies).ndim == 1:
                print(f"⚠️ Warning: Accuracy data for '{label}' is 1D. Plotting without an error band.")
            color = colors.get(label, f'C{i}')
            data_artists += ax.plot(
                time_points, mean,
                label=label,
                color=color,
                linestyle=linestyles.get(label, '-'),
                linewidth=linewidth,
                solid_capstyle='round',
                zorder=3,
            )
            if np.any(spread > 0):
                data_artists.append(ax.fill_between(
                    time_points, mean - spread, mean + spread,
                    color=color, alpha=band_alpha, linewidth=0, zorder=1,
                ))

        if show_chance_level:
            data_artists.append(ax.axhline(
                y=chance_level, color='#8C8C8C', linestyle='--',
                linewidth=0.8, zorder=2, label='Chance',
            ))
        data_artists.append(ax.axvline(
            x=0, color='#BFBFBF', linestyle='-', linewidth=0.8, zorder=0,
        ))

        # --- limits, with a clean strip reserved on top for the bars ------
        data_lo, data_hi = float(ylim[0]), float(ylim[1])
        band_frac = sig_band_frac if rows else 0.0
        axes_hi = data_lo + (data_hi - data_lo) / (1.0 - band_frac) if band_frac else data_hi
        ax.set_ylim(data_lo, axes_hi)

        if xlim is None:
            xlim = (float(time_points.min()), float(time_points.max()))
        ax.set_xlim(xlim)

        if band_frac:
            clip = Rectangle(
                (xlim[0], data_lo), xlim[1] - xlim[0], data_hi - data_lo,
                transform=ax.transData,
            )
            for artist in data_artists:
                artist.set_clip_path(clip)

        # --- significance strip ------------------------------------------
        chance_bar_colors = _draw_sig_strip(
            ax, time_points, xlim, rows, band_frac,
            contrast_bar_linewidth=contrast_bar_linewidth,
            chance_bar_linewidth=chance_bar_linewidth,
            chance_bar_color=chance_bar_color,
            base_fontsize=base_fontsize,
        )

        # --- ticks: over the data range only, never into the strip --------
        yticks = _nice_ticks(data_lo, data_hi, n_yticks)
        ax.set_yticks(yticks)
        ax.set_yticklabels([_tick_label(y) for y in yticks])
        xticks = _xticks(xlim, xtick_step)
        if xticks is not None:
            ax.set_xticks(xticks)
        ax.spines['left'].set_bounds(data_lo, data_hi)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title:
            # Clear the top significance row, which sits just inside the axes.
            ax.set_title(title, pad=0.6 * base_fontsize if rows else 4)

        # --- legends ------------------------------------------------------
        main_legend = ax.legend(loc=legend_loc) if show_legend else None
        if show_sig_legend:
            handles, labels, handler_map = _sig_legend_entries(
                contrast_entries, chance_bar_colors, chance_bar_label,
                contrast_bar_linewidth, chance_bar_linewidth,
            )
            if handles:
                # A composite handle is split into one section per colour, so
                # it needs extra width or the sections read as a dashed line
                # instead of as one bar per condition.
                n_sections = max(
                    (len(h) for h in handles if isinstance(h, tuple)), default=1)
                sig_legend = ax.legend(
                    handles, labels, loc=sig_legend_loc,
                    handler_map=handler_map, title=sig_legend_title,
                    handlelength=max(1.4, 1.1 * n_sections),
                    handletextpad=0.8,
                )
                if sig_legend_title:
                    sig_legend.get_title().set_fontsize(base_fontsize - 1.5)
                if main_legend is not None:
                    # ax.legend() drops the previous legend; put it back.
                    ax.add_artist(main_legend)

        fig.tight_layout(pad=0.4)

        if return_fig:
            return fig

        filename = "_".join(filter(None, [
            f"{timestamp}" if timestamp else "",
            comparison_name, roi, filename_suffix,
        ]))
        os.makedirs(save_dir, exist_ok=True)
        stem = os.path.join(save_dir, filename)
        for fmt in formats:
            fig.savefig(f'{stem}.{fmt}', format=fmt, bbox_inches='tight', pad_inches=0.05)
        plt.close(fig)
        print(f"Saved decoding accuracy plot to: {stem}.{formats[0]}")


def _draw_sig_strip(ax, time_points, xlim, rows, band_frac, *,
                    contrast_bar_linewidth, chance_bar_linewidth,
                    chance_bar_color, base_fontsize):
    """Draw the significance rows; return the colours used for chance bars."""
    if not rows:
        return []

    # x in data coordinates, y in axes fraction: the strip stays put whatever
    # the data limits are.
    transform = ax.get_xaxis_transform()
    top, bottom = 1.0 - 0.02, 1.0 - band_frac + 0.015
    row_ys = ([(top + bottom) / 2.0] if len(rows) == 1
              else list(np.linspace(top, bottom, len(rows))))

    chance_colors = []
    for row, y in zip(rows, row_ys):
        is_contrast = row['kind'] != 'chance'
        lw = contrast_bar_linewidth if is_contrast else chance_bar_linewidth
        for entry in row['entries']:
            color = entry['color']
            if not is_contrast:
                # Against-chance bars default to their trace's colour, which
                # is what links each bar to its line. One neutral colour for
                # all of them subordinates them to the contrast bars instead,
                # at the cost of no longer saying which trace each belongs to.
                color = chance_bar_color or color
                chance_colors.append(color)
            for start, end in _cluster_spans(entry['clusters'], time_points, xlim):
                ax.plot(
                    [start, end], [y, y],
                    transform=transform,
                    color=color,
                    linestyle=entry['linestyle'],
                    linewidth=lw,
                    solid_capstyle='butt',
                    dash_capstyle='butt',
                    clip_on=False,
                    zorder=5,
                )
                if entry['marker']:
                    ax.text(
                        (start + end) / 2.0, y + 0.015, entry['marker'],
                        transform=transform, ha='center', va='bottom',
                        fontsize=base_fontsize - 1, color=color,
                        clip_on=False,
                    )
    return chance_colors


def _sig_legend_entries(contrast_entries, chance_bar_colors, chance_bar_label,
                        contrast_bar_linewidth, chance_bar_linewidth):
    """One legend entry per contrast direction, plus one for all chance bars.

    The against-chance bars share a single entry: each is drawn in its own
    trace's colour, so the trace legend already says which bar belongs to
    which condition and a per-trace entry would only repeat it. When several
    colours are in play the handle stacks one short line per colour, so the
    entry still looks like what is on the figure.
    """
    handles, labels, handler_map = [], [], {}

    for entry in contrast_entries:
        if not entry['label']:
            continue
        handles.append(Line2D(
            [0], [0], color=entry['color'], linestyle=entry['linestyle'],
            linewidth=contrast_bar_linewidth, solid_capstyle='butt',
        ))
        labels.append(entry['label'])

    if chance_bar_colors and chance_bar_label:
        parts = tuple(
            Line2D([0], [0], color=c, linewidth=chance_bar_linewidth,
                   solid_capstyle='butt')
            for c in dict.fromkeys(chance_bar_colors)
        )
        handle = parts[0] if len(parts) == 1 else parts
        if len(parts) > 1:
            handler_map[handle] = HandlerTuple(ndivide=None, pad=0)
        handles.append(handle)
        labels.append(chance_bar_label)

    return handles, labels, handler_map
