"""Plot helpers for the F-traces saved by power_traces_dcc.py."""
from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt


def _load_F_trace_files(save_dir, roi, conditions_save_name):
    """Yield (effect_name, npz_data) pairs for one ROI."""
    p = Path(save_dir) / 'anova_F_traces'
    for f in sorted(p.glob(f'{conditions_save_name}_{roi}_*.npz')):
        # strip prefix + .npz; what's left is the safe-effect name
        m = re.match(rf'{re.escape(conditions_save_name)}_{re.escape(roi)}_(.+)\.npz', f.name)
        if not m:
            continue
        yield m.group(1), np.load(f)


def plot_anova_F_traces_for_roi(save_dir, roi, conditions_save_name,
                                window_centers=None, effects=None,
                                figsize_per_panel=(6, 3), save_path=None):
    """One panel per effect: observed F + null 50-95 envelope + cluster bar."""
    items = list(_load_F_trace_files(save_dir, roi, conditions_save_name))
    if effects:
        items = [(e, d) for (e, d) in items if e in effects]
    if not items:
        print(f"No F-traces found for {roi}.")
        return None

    n = len(items)
    fig, axes = plt.subplots(n, 1, figsize=(figsize_per_panel[0], figsize_per_panel[1] * n))
    if n == 1:
        axes = [axes]

    for ax, (eff, data) in zip(axes, items):
        obs = data['observed_F']
        null = data['null_F']
        wmask = data['window_mask']
        xs = window_centers if window_centers is not None else np.arange(len(obs))
        ax.plot(xs, obs, color='black', lw=2, label='observed F')
        ax.fill_between(xs,
                        np.percentile(null, 50, axis=0),
                        np.percentile(null, 95, axis=0),
                        color='gray', alpha=0.25, label='null 50-95th')
        ax.plot(xs, np.percentile(null, 95, axis=0), color='gray', lw=1, ls='--')
        if wmask.any():
            ymax = obs.max() * 1.1
            ax.fill_between(xs, ymax - 0.05, ymax,
                            where=wmask, color='red', alpha=0.7)
        ax.set_title(f'{roi} — {eff}')
        ax.set_xlabel('Time' if window_centers is not None else 'Window idx')
        ax.set_ylabel('F')
    axes[0].legend(loc='upper right', fontsize=9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    return fig

def plot_per_electrode_F_traces(within_run_dir, roi, effect,
                                channels_per_page=20, grid_shape=None,
                                save_dir=None, sample_window_centers=None):
    """One subplot per electrode, paginated. Mirrors plot_mask_pages from wavelet_functions.py."""
    from pathlib import Path
    p = Path(within_run_dir) / roi
    if not p.exists():
        print(f"No within-electrode results at {p}")
        return []

    # Walk subjects/electrodes
    elec_files = sorted(p.glob('*/*.npz'))
    if not elec_files:
        return []
    pages = []
    for start in range(0, len(elec_files), channels_per_page):
        chunk = elec_files[start:start + channels_per_page]
        n_ch = len(chunk)
        if grid_shape is None:
            n_rows = max(1, int(np.floor(np.sqrt(n_ch))))
            n_cols = int(np.ceil(n_ch / n_rows))
        else:
            n_rows, n_cols = grid_shape
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 2.5 * n_rows))
        axes = np.atleast_1d(axes).flatten()
        for i, fpath in enumerate(chunk):
            data = np.load(fpath)
            effect_names = list(data['effect_names'])
            if effect not in effect_names:
                continue
            ei = effect_names.index(effect)
            obs = data['observed_F'][ei]
            null = data['null_F'][:, ei, :] if data['null_F'].ndim == 3 else None
            wmask = data['window_mask'][ei]
            xs = sample_window_centers if sample_window_centers is not None else np.arange(len(obs))

            ax = axes[i]
            ax.plot(xs, obs, color='black', lw=1.3)
            if null is not None:
                ax.fill_between(xs, np.percentile(null, 95, axis=0),
                                np.percentile(null, 50, axis=0),
                                color='gray', alpha=0.2)
            if wmask.any():
                ymax = obs.max() * 1.05
                ax.fill_between(xs, ymax - 0.05, ymax, where=wmask, color='red', alpha=0.7)
            elec_label = f"{fpath.parent.name}/{fpath.stem}"
            ax.set_title(elec_label, fontsize=7)
            ax.tick_params(labelsize=6)
        for j in range(n_ch, len(axes)):
            axes[j].axis('off')
        fig.suptitle(f"{roi} — {effect}  (page {start // channels_per_page + 1})", y=1.0)
        fig.tight_layout()
        pages.append(fig)
        if save_dir:
            fig.savefig(Path(save_dir) / f'{roi}_{effect}_page{start // channels_per_page + 1}.png',
                        dpi=150, bbox_inches='tight')
    return pages

def plot_per_electrode_power_traces(subjects_mne_objects, rois, condition_names,
                                    electrodes_per_subject_roi,
                                    plotting_parameters,
                                    channels_per_page=20, grid_shape=None,
                                    save_dir=None, error='sem'):
    """For each ROI, plot a grid of single-electrode power traces (one subplot per electrode).

    Re-uses the existing per-electrode evoked structure rather than re-computing.
    """
    from pathlib import Path
    pages_by_roi = {}
    for roi in rois:
        flat = []  # list of (sub, elec, evk_dict_for_this_elec)
        for sub, elecs in electrodes_per_subject_roi.get(roi, {}).items():
            for elec in elecs:
                # Build a tiny one-channel evk_dict per condition for this electrode.
                per_cond = {}
                for cname in condition_names:
                    mne_obj = subjects_mne_objects[sub][cname]
                    if mne_obj is None or elec not in mne_obj.ch_names:
                        per_cond[cname] = None
                        continue
                    # Average across trials for this one electrode
                    per_cond[cname] = mne_obj.copy().pick([elec]).average()
                flat.append((sub, elec, per_cond))

        pages = []
        for start in range(0, len(flat), channels_per_page):
            chunk = flat[start:start + channels_per_page]
            n = len(chunk)
            if grid_shape is None:
                n_rows = max(1, int(np.floor(np.sqrt(n))))
                n_cols = int(np.ceil(n / n_rows))
            else:
                n_rows, n_cols = grid_shape
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 2.5 * n_rows))
            axes = np.atleast_1d(axes).flatten()
            for i, (sub, elec, per_cond) in enumerate(chunk):
                ax = axes[i]
                for cname, evk in per_cond.items():
                    if evk is None:
                        continue
                    params = plotting_parameters.get(cname, {})
                    color = params.get('color', 'gray')
                    ls = params.get('line_style', '-')
                    times = evk.times
                    data = evk.data[0]   # single-channel
                    ax.plot(times, data, color=color, linestyle=ls, lw=1.0)
                ax.set_title(f'{sub}/{elec}', fontsize=7)
                ax.axhline(0, color='black', lw=0.4, ls=':')
                ax.axvline(0, color='black', lw=0.4, ls=':')
                ax.tick_params(labelsize=6)
            for j in range(n, len(axes)):
                axes[j].axis('off')
            fig.suptitle(f'{roi} — page {start // channels_per_page + 1}', y=1.0)
            fig.tight_layout()
            pages.append(fig)
            if save_dir:
                fig.savefig(Path(save_dir) / f'{roi}_powertraces_page{start // channels_per_page + 1}.png',
                            dpi=150, bbox_inches='tight')
        pages_by_roi[roi] = pages
    return pages_by_roi