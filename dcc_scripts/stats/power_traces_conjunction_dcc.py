#!/usr/bin/env python
"""
DCC core for the conjunction driven by `power_traces` electrode definitions.

This is the A1/A2 layer again, but with the ELECTRODE DEFINITION swapped: instead
of fitting one ANOVA on window-mean HG (`sfs.per_electrode_anova_labels`, run by
`stability_flexibility_anova_conjunction_dcc.py`), the S/F flags are read back
from a within-electrode WINDOWED ANOVA + cluster-correction run
(`src/analysis/power/windowed_anova.py`, launched by
`dcc_scripts/power/run_power_traces_dcc.py` with `ANOVA_UNIT='electrode'`).

  1. `ptc.electrode_labels` reads the run's `summary.csv`, pivots the four
     interactions (CPC/SPS/CPS/SPC) onto one row per electrode, and applies the
     across-electrode correction the 2x2 needs (BH by default).
  2. `ptc.run_power_traces_conjunction` runs the count battery on those labels:
     CMH, within-subject permutation null on the joint count, shared-vs-distinct,
     the two cross-interaction specificity controls, and a threshold sweep.
  3. OPTIONALLY (`--run-continuous`) the continuous CONFOUND CONTROL for those
     counts: `ptc.continuous_confound_control` re-estimates each electrode's two
     sensitivities on DISJOINT trial halves, residualised on responsiveness, over
     the SAME stretch of the epoch the windowed-ANOVA run tiled, under all three
     effect measures. This is the only step that needs the epoched data, and it
     is a control on the counts -- not a second headline (guide §5.1).

Driven by `run_power_traces_conjunction_dcc.py` (wrapped by
`sbatch_power_traces_conjunction_dcc.sh`). Not run directly on the cluster; call
`main(args)` with a populated argument namespace.

Nothing here re-fits an ANOVA: step 1-2 are pure reads of a finished
`power_traces` run, so they are cheap and can be re-run at several alphas /
corrections without touching the epochs.
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import sys
import os
import json

# ---------------------------------------------------------------------------
# PATH SETUP (mirrors the other dcc_scripts entrypoints)
# ---------------------------------------------------------------------------
try:
    current_file_path = os.path.abspath(__file__)
    current_script_dir = os.path.dirname(current_file_path)
except NameError:
    current_script_dir = os.getcwd()

project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

if os.path.exists("/hpc/home"):
    USER = os.environ.get('USER')
    sys.path.append(f"/hpc/home/{USER}/coganlab/{USER}/GlobalLocal/IEEG_Pipelines/")

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')          # headless / cluster
import matplotlib.pyplot as plt

from src.analysis.stats import power_traces_conjunction as ptc
# NOTE: `general_utils` (and through it `mne`) is imported lazily inside
# `run_continuous_control`. The count test is a pure read of a finished run, so
# it must stay runnable in an environment with no epoched-data stack at all.

STAB, FLEX = "#2c7fb8", "#d95f0e"


# ---------------------------------------------------------------------------
# run-directory resolution
# ---------------------------------------------------------------------------
def resolve_runs(args):
    """`{group: run_dir}` for `ptc.electrode_labels`, from the args namespace.

    Two shapes, matching how `power_traces` was actually launched:

      * ONE run whose ANOVA held all four factors (`stimulus_experiment_conditions`,
        `anova_factors=['congruency','incongruentProportion','switchType',
        'switchProportion']`) -- pass `args.run_dir`, and every interaction is read
        from that single `summary.csv`.
      * FOUR separate two-factor runs (`stimulus_lwpc_conditions`,
        `stimulus_lwps_conditions`, `stimulus_congruency_by_switch_proportion_conditions`,
        `stimulus_switch_type_by_incongruent_proportion_conditions`) -- pass
        `args.run_dirs` as a `{group: path}` dict.

    The four-run shape is allowed to be partial: only CPC and SPS are required
    (the S x F table needs those two), the cross controls are reported when present.
    """
    if getattr(args, 'run_dirs', None):
        runs = {g: p for g, p in args.run_dirs.items() if p}
    elif getattr(args, 'run_dir', None):
        runs = args.run_dir
    else:
        raise ValueError(
            "no power_traces run given: set PT_RUN to a single 4-factor "
            "within-electrode ANOVA run directory, or PT_RUN_CPC / PT_RUN_SPS "
            "(+ optionally PT_RUN_CPS / PT_RUN_SPC) to four separate runs.")

    for group, path in ([('run', runs)] if isinstance(runs, str)
                        else sorted(runs.items())):
        if not os.path.isdir(path):
            raise FileNotFoundError(f"{group}: not a directory: {path}")
        if not os.path.exists(os.path.join(path, 'summary.csv')):
            raise FileNotFoundError(
                f"{group}: no summary.csv in {path}. Point at the run directory "
                "written by run_within_electrode_windowed_anova_cluster_correction "
                "(i.e. <SAVE_DIR>/anova_within_electrode/<conditions_save_name>), "
                "not its parent.")
    return runs


def rois_in_run(runs):
    """ROI names present in the run(s), from `summary.csv`'s `roi` column."""
    paths = [runs] if isinstance(runs, str) else list(runs.values())
    rois = set()
    for p in paths:
        rois |= set(pd.read_csv(os.path.join(p, 'summary.csv'))['roi'].unique())
    return sorted(str(r) for r in rois)


# ---------------------------------------------------------------------------
# synthetic run directories (path / plumbing validation, no cluster data)
# ---------------------------------------------------------------------------
def make_synthetic_runs(save_dir, n_subj=8, n_elec=25, overlap=0.6, base_rate=0.25,
                        roi='lpfc', seed=0, tmin=0.0, tmax=1.5, n_windows=30):
    """Write four fake within-electrode ANOVA runs to disk and return their paths.

    Plants a KNOWN degree of S/F co-selectivity so the whole read -> label ->
    conjunction path can be validated without a real run: `overlap` is the
    probability that an S electrode is also F, against a base rate `base_rate` of
    each selectivity. `overlap > base_rate` should come back as MH OR > 1;
    `overlap == base_rate` is the null and should come back ~1.

    The two cross interactions are drawn at the null, so the specificity controls
    should stay near-empty -- which is what makes the synthetic run a real check
    on the cross-control panel and not just on the plumbing.
    """
    rng = np.random.default_rng(seed)
    effects = {
        'CPC': ptc.anova_effect_name('congruency', 'incongruentProportion'),
        'SPS': ptc.anova_effect_name('switchType', 'switchProportion'),
        'CPS': ptc.anova_effect_name('congruency', 'switchProportion'),
        'SPC': ptc.anova_effect_name('switchType', 'incongruentProportion'),
    }
    rows = {g: [] for g in effects}
    for s in range(n_subj):
        for e in range(n_elec):
            sub, elec = f"S{s:02d}", f"e{e:02d}"
            is_s = rng.random() < base_rate
            # P(F) is base_rate either way, but conditioned on S it is `overlap`
            p_f = overlap if is_s else max(
                0.0, (base_rate - base_rate * overlap) / (1 - base_rate))
            flags = dict(CPC=is_s, SPS=rng.random() < p_f,
                         CPS=rng.random() < 0.02, SPC=rng.random() < 0.02)
            for g, eff in effects.items():
                sig = flags[g]
                # a graded p either side of the cutoff, so BH has something to rank
                p = float(rng.uniform(0.0001, 0.01) if sig
                          else rng.uniform(0.2, 1.0))
                ext = int(rng.integers(3, 9)) if sig else int(rng.integers(0, 2))
                sgn = int(rng.choice([-1, 1])) if sig else 0
                rows[g].append(dict(
                    subject=sub, electrode=elec, roi=roi, effect=eff,
                    cluster_idx=0 if sig else -1, sign=sgn, extent_windows=ext,
                    cluster_onset=0.1, cluster_offset=0.3,
                    cluster_p_value=p if sig else 1.0,
                    best_cluster_p=p, best_cluster_extent=ext,
                    best_cluster_sign=sgn,
                    peak_F=float(rng.uniform(1, 12)), peak_time=0.2))

    centers = np.linspace(tmin, tmax, n_windows)
    out = {}
    for g, rs in rows.items():
        run_dir = os.path.join(save_dir, 'synthetic_runs', g.lower())
        os.makedirs(run_dir, exist_ok=True)
        pd.DataFrame(rs).to_csv(os.path.join(run_dir, 'summary.csv'), index=False)
        with open(os.path.join(run_dir, 'run_config.json'), 'w') as f:
            json.dump(dict(window_centers=[float(c) for c in centers],
                           window_size=50, step_size=10, sampling_rate=256.0,
                           n_windows=int(n_windows),
                           analysis_tmin=float(tmin), analysis_tmax=float(tmax),
                           rois=[roi], subjects=[f"S{s:02d}" for s in range(n_subj)]),
                      f, indent=2)
        out[g] = run_dir
    return out


# ---------------------------------------------------------------------------
# serialization helpers
# ---------------------------------------------------------------------------
def _json_safe(o):
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def save_results(labels, res, save_dir):
    """Write the label table, the conjunction/null JSONs, and the raw nulls."""
    os.makedirs(save_dir, exist_ok=True)
    labels.to_csv(os.path.join(save_dir, 'labels.csv'), index=False)
    funnel = pd.DataFrame(labels.attrs.get('label_funnel', []))
    if not funnel.empty:
        funnel.to_csv(os.path.join(save_dir, 'label_funnel.csv'), index=False)
    res['cross_controls'].to_csv(
        os.path.join(save_dir, 'cross_controls.csv'), index=False)
    if 'sweep' in res:
        res['sweep'].to_csv(os.path.join(save_dir, 'threshold_sweep.csv'),
                            index=False)

    conj = res['cmh']
    conj_json = dict(
        mh_odds_ratio=conj['mh_odds_ratio'],
        cmh_pvalue=conj['cmh'].pvalue, cmh_statistic=conj['cmh'].statistic,
        homogeneity_pvalue=conj['homogeneity'].pvalue,
        homogeneity_statistic=conj['homogeneity'].statistic,
        pooled_table=conj['pooled_table'],
        pooled_fisher_or=conj['pooled_fisher_or'],
        pooled_fisher_p=conj['pooled_fisher_p'],
        labels=res['attrs'])
    if 'or_95ci' in conj:
        conj_json['mh_or_95ci'] = list(conj['or_95ci'])
    with open(os.path.join(save_dir, 'conjunction.json'), 'w') as f:
        json.dump(_json_safe(conj_json), f, indent=2)
    conj['per_subject'].to_csv(
        os.path.join(save_dir, 'conjunction_per_subject.csv'), index=False)

    null, bvd = res['joint_count_null'], res['both_vs_distinct']
    counts_json = dict(
        n_electrodes=bvd['n_electrodes'], n_subjects=bvd['n_subjects'],
        n_both=bvd['n_both'], n_stab_only=bvd['n_stab_only'],
        n_flex_only=bvd['n_flex_only'], n_neither=bvd['n_neither'],
        joint_count=dict(observed=null['observed'],
                         null_mean=float(np.mean(null['null'])),
                         null_std=float(np.std(null['null'])),
                         p_two_sided=null['p_two_sided'], z=null['z'],
                         n_perm=int(len(null['null']))),
        shared_minus_distinct=dict(
            observed=bvd['observed'], null_mean=bvd['null_mean'],
            null_sd=bvd['null_sd'], p_shared_greater=bvd['p_shared_greater'],
            p_two_sided=bvd['p_two_sided'], z=bvd['z'],
            n_perm=int(len(bvd['null'])),
            # documented in both_vs_distinct_test: with the marginals fixed by the
            # within-subject shuffle, D = 3*both - n_S - n_F, so this is the SAME
            # test as the joint count above -- reported, not double-counted.
            equivalent_to_joint_count=True))
    with open(os.path.join(save_dir, 'counts.json'), 'w') as f:
        json.dump(_json_safe(counts_json), f, indent=2)
    np.save(os.path.join(save_dir, 'joint_count_null.npy'), null['null'])
    np.save(os.path.join(save_dir, 'shared_minus_distinct_null.npy'), bvd['null'])


def save_continuous(control, save_dir):
    with open(os.path.join(save_dir, 'continuous_confound_control.json'), 'w') as f:
        json.dump(_json_safe(control), f, indent=2)


# ---------------------------------------------------------------------------
# threshold-sweep hygiene
# ---------------------------------------------------------------------------
def degenerate_sweep_rows(sweep, min_strata=3):
    """Boolean mask of sweep rows too thin to say anything about stability.

    `cmh_conjunction` now drops subject strata that carry no information about
    the S-F association and returns NaN when none are left, so an all-empty
    threshold no longer fabricates an odds ratio. What it cannot do is decide how
    many subjects are ENOUGH: a row resting on one or two informative strata has
    a finite OR that is nonetheless not evidence the verdict is threshold-stable.

    So flag three things: an undefined OR, fewer than `min_strata` informative
    subjects, and `n_both == 0` (an OR built entirely from the off-diagonal).
    These stay out of the plotted trend line and are named in `summary.txt`.
    """
    s = sweep
    thin = (s['n_informative_strata'] < min_strata
            if 'n_informative_strata' in s.columns
            else (s['n_S'] == 0) | (s['n_F'] == 0))
    return (~np.isfinite(s['mh_odds_ratio']) | thin | (s['n_both'] == 0)).to_numpy()


def _sweep_warning(sweep, min_strata=3):
    bad = degenerate_sweep_rows(sweep, min_strata=min_strata)
    if not bad.any():
        return []
    bits = []
    for _, r in sweep[bad].iterrows():
        why = []
        if not np.isfinite(r['mh_odds_ratio']):
            why.append("OR undefined")
        if r.get('n_informative_strata', np.inf) < min_strata:
            why.append(f"{int(r['n_informative_strata'])} informative subject(s)")
        if r['n_both'] == 0:
            why.append("n_both = 0")
        bits.append(f"q<{r['threshold']} ({'; '.join(why)})")
    return ["  WARNING: sweep row(s) " + ", ".join(bits) + " are too thin to "
            "support a threshold-stability claim; ignore them when reading the "
            "sweep. An undefined OR means no subject stratum had both an S and "
            "an F electrode at that cutoff."]


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def make_plots(labels, res, save_dir, roi_tag=''):
    lab = labels
    null, bvd, conj = res['joint_count_null'], res['both_vs_distinct'], res['cmh']

    fig, ax = plt.subplots(2, 3, figsize=(16, 9))

    # A: selectivity classes
    ax[0, 0].bar(["neither", "S only", "F only", "both"],
                 [bvd['n_neither'], bvd['n_stab_only'], bvd['n_flex_only'],
                  bvd['n_both']],
                 color=["#ccc", STAB, FLEX, "#31a354"])
    ax[0, 0].set(title=f"electrode selectivity classes{roi_tag}\n"
                       f"(power_traces cluster-corrected definition)",
                 ylabel="# electrodes")

    # B: pooled 2x2 + MH OR
    pooled = np.asarray(conj['pooled_table'])
    ax[0, 1].imshow(pooled, cmap="Blues")
    ax[0, 1].set_xticks([0, 1]); ax[0, 1].set_xticklabels(["F=1", "F=0"])
    ax[0, 1].set_yticks([0, 1]); ax[0, 1].set_yticklabels(["S=1", "S=0"])
    for (i, j), v in np.ndenumerate(pooled):
        ax[0, 1].text(j, i, int(v), ha="center", va="center",
                      color="white" if v > pooled.max() / 2 else "black", fontsize=13)
    ax[0, 1].set(title=f"pooled 2x2 (MH OR={conj['mh_odds_ratio']:.2f}, "
                       f"CMH p={conj['cmh'].pvalue:.3g})")

    # C: within-subject null on the joint count
    ax[0, 2].hist(null['null'], bins=40, color="#bbb")
    ax[0, 2].axvline(null['observed'], color="#d7191c", lw=2,
                     label=f"observed = {null['observed']}")
    ax[0, 2].legend()
    ax[0, 2].set(title=f"within-subject null on 'both'\n"
                       f"p={null['p_two_sided']:.3g}, z={null['z']:+.2f}",
                 xlabel="# 'both' electrodes", ylabel="# permutations")

    # D: shared - distinct (same test as C; plotted to make the identity visible)
    ax[1, 0].hist(bvd['null'], bins=40, color="#bbb")
    ax[1, 0].axvline(bvd['observed'], color="#d7191c", lw=2,
                     label=f"observed = {bvd['observed']}")
    ax[1, 0].legend()
    ax[1, 0].set(title=f"shared - distinct (monotone in 'both':\n"
                       f"same test as the panel above, p={bvd['p_two_sided']:.3g})",
                 xlabel="both - (S-only + F-only)", ylabel="# permutations")

    # E: threshold sweep. Degenerate rows (empty marginal / no 'both') are drawn
    # as hollow markers and kept OUT of the trend line -- see
    # `degenerate_sweep_rows` for why their OR is an artifact of shift_zeros.
    if 'sweep' in res and len(res['sweep']):
        sw = res['sweep']
        bad = degenerate_sweep_rows(sw)
        ok = sw[~bad]
        if len(ok):
            ax[1, 1].plot(ok['threshold'], ok['mh_odds_ratio'], 'o-',
                          color="#31a354", label="informative")
        if bad.any():
            ax[1, 1].plot(sw.loc[bad, 'threshold'], sw.loc[bad, 'mh_odds_ratio'],
                          'o', mfc='none', color="#999",
                          label="degenerate (empty cell)")
        ax[1, 1].axhline(1.0, ls='--', c='k', lw=1, label="OR = 1 (independence)")
        ax[1, 1].legend(fontsize=8)
        ax[1, 1].set(title="overlap OR vs selection threshold",
                     xlabel="q-value cutoff", ylabel="MH odds ratio")
    else:
        ax[1, 1].text(.5, .5, "no q-values\n(correction='cluster' / 'none')",
                      ha='center', va='center', transform=ax[1, 1].transAxes)
        ax[1, 1].set(title="overlap OR vs selection threshold")

    # F: cross-interaction specificity controls
    cc = res['cross_controls']
    if len(cc):
        colors = {'CPC': STAB, 'SPS': FLEX, 'CPS': "#8856a7", 'SPC': "#7fbf7b"}
        ax[1, 2].bar(cc['group'], 100 * cc['frac'],
                     color=[colors.get(g, "#999") for g in cc['group']])
        for xi, (n, fr) in enumerate(zip(cc['n_significant'], cc['frac'])):
            ax[1, 2].text(xi, 100 * fr, f"{int(n)}", ha='center', va='bottom')
    ax[1, 2].set(title="cross controls (CPS/SPC should be near-null)",
                 ylabel="% of electrodes selected")

    fig.tight_layout()
    fig_path = os.path.join(save_dir, 'power_traces_conjunction_summary.png')
    fig.savefig(fig_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"saved figure: {fig_path}")

    # a second, smaller figure: the graded evidence behind the flags
    if 'p_cpc' in lab.columns:
        fig2, ax2 = plt.subplots(1, 2, figsize=(11, 4.5))
        qx = 'q_cpc' if lab['q_cpc'].notna().any() else 'p_cpc'
        qy = 'q_sps' if lab['q_sps'].notna().any() else 'p_sps'
        ax2[0].scatter(lab[qx], lab[qy], c=lab['S'] + 2 * lab['F'],
                       cmap="viridis", s=20, alpha=.75)
        ax2[0].axvline(res['attrs'].get('alpha', .05), ls='--', c='k', lw=1)
        ax2[0].axhline(res['attrs'].get('alpha', .05), ls='--', c='k', lw=1)
        ax2[0].set(title="per-electrode evidence (dashed = alpha)",
                   xlabel=f"{qx} (stability / LWPC)",
                   ylabel=f"{qy} (flexibility / LWPS)",
                   xlim=(-.02, 1.02), ylim=(-.02, 1.02))
        ax2[1].scatter(lab['cpc_extent'], lab['sps_extent'], s=20, alpha=.6,
                       color="#444")
        ax2[1].set(title="cluster extent (windows) per electrode",
                   xlabel="CPC extent", ylabel="SPS extent")
        fig2.tight_layout()
        p2 = os.path.join(save_dir, 'power_traces_conjunction_evidence.png')
        fig2.savefig(p2, dpi=140, bbox_inches='tight')
        plt.close(fig2)
        print(f"saved figure: {p2}")


# ---------------------------------------------------------------------------
# text summary
# ---------------------------------------------------------------------------
def write_summary(labels, res, save_dir, meta, control=None, alpha=0.05):
    lines = [
        "=" * 72,
        "STABILITY vs FLEXIBILITY - CONJUNCTION ON power_traces ELECTRODES",
        "=" * 72,
    ]
    for key, val in meta.items():
        lines.append(f"{key:>26}: {val}")
    lines += ["-" * 72, ptc.summarize(res)]
    if 'sweep' in res and len(res['sweep']):
        lines += _sweep_warning(res['sweep'])
    if labels.attrs.get('n_dropped'):
        lines.append(f"  NOTE: {labels.attrs['n_dropped']} electrode(s) dropped -- "
                     "not present in every requested run (require_all=True). A 2x2 "
                     "built over inconsistent denominators is not interpretable.")
    funnel = pd.DataFrame(labels.attrs.get('label_funnel', []))
    if not funnel.empty:
        lines += [
            "  label funnel (where power_traces counts can drop):",
            funnel.to_string(index=False),
            "  Reading: run_raw_p_lt_alpha is the like-for-like raw cluster count "
            "from the power-traces run. final_flagged is the count actually used "
            "here after run alignment and the requested correction.",
        ]
    if control is not None:
        lines += ["-" * 72, ptc.summarize_confound_control(control, alpha=alpha)]
    else:
        lines += ["-" * 72,
                  "continuous confound control: NOT RUN. The count test cannot "
                  "correct for per-electrode detection power or shared trial "
                  "noise, and both bias it toward 'shared'. Re-run with "
                  "RUN_CONTINUOUS=1 before reporting a shared-core claim."]
    lines += [
        "=" * 72,
        "Reading: MH OR < 1 / overlap below null -> segregation (distinct "
        "populations); OR > 1 / overlap above null -> shared core; ~1 -> independent.",
        "Note 'shared - distinct' is a monotone function of the 'both' count under "
        "this null, so it is the SAME test as the joint-count permutation -- not a "
        "second, independent line of evidence.",
    ]
    txt = "\n".join(str(x) for x in lines)
    with open(os.path.join(save_dir, 'summary.txt'), 'w') as f:
        f.write(txt + "\n")
    print(txt)


# ---------------------------------------------------------------------------
# the continuous confound control (the only step that touches the epochs)
# ---------------------------------------------------------------------------
def run_continuous_control(args, labels, runs, save_dir):
    """Re-estimate the two sensitivities continuously over the run's own window.

    Three things make this a control on the counts rather than a separate result:
    the window comes from the run's `run_config.json` (same stretch of epoch), the
    electrode universe is restricted to the electrodes the labels were built on,
    and all three effect measures are run so the verdict cannot hinge on the
    scalarisation choice (guide §5.1).
    """
    from dcc_scripts.stats.stability_flexibility_segregation_dcc import (
        assemble_long_df, make_synthetic_df)
    from src.analysis.utils.general_utils import (
        resolve_lab_root, resolve_electrodes_to_keep)

    ref_run = runs if isinstance(runs, str) else runs['CPC']
    tmin, tmax = ptc.analysis_window_from_run(ref_run)
    print(ptc.describe_window_alignment(ref_run, tmin, tmax))
    print(f"continuous control window: [{tmin:.4f}, {tmax:.4f}] s "
          "(taken from the run, not from WINDOW_TMIN/TMAX)")

    if args.data_source == 'synthetic':
        df = make_synthetic_df(rho_true=getattr(args, 'synthetic_rho', 0.4),
                               effect_measure='cluster')
    else:
        LAB_root = resolve_lab_root(args.LAB_root)
        from src.analysis.utils.general_utils import load_HG_ev1_rescaled_per_subject
        subjects_epochs = load_HG_ev1_rescaled_per_subject(
            subjects=args.subjects, epochs_root_file=args.epochs_root_file,
            task=args.task, LAB_root=LAB_root, acc_trials_only=args.acc_trials_only)
        keep = resolve_electrodes_to_keep(args, LAB_root)
        # `assemble_long_df` needs time courses for peak_t / cluster; 'cohens_d'
        # is taken on the window mean of the same courses inside the control.
        df = assemble_long_df(subjects_epochs, tmin, tmax,
                              electrodes_to_keep=keep, effect_measure='cluster')

        # Restrict to the electrodes the labels are defined on. The long table
        # names electrodes '<subject>-<channel>'; the run's summary.csv keeps the
        # channel and the subject in separate columns.
        wanted = {f"{s}-{e}" for s, e in
                  zip(labels['subject'], labels['electrode'])}
        before = df['electrode'].nunique()
        df = df[df['electrode'].isin(wanted)].reset_index(drop=True)
        after = df['electrode'].nunique()
        print(f"continuous control electrodes: {after} of {before} in the long "
              f"table matched the {len(wanted)} labelled electrodes")
        if after == 0:
            raise RuntimeError(
                "no electrode in the assembled long table matched the labelled "
                "electrodes. Check that the power_traces run and this job were "
                "given the same subjects / ROI / epochs file.")

    control = ptc.continuous_confound_control(
        df, responsiveness=getattr(args, 'responsiveness', None),
        n_splits=args.n_splits, n_perm=args.n_perm_corr,
        min_elec=args.min_elec, alpha=args.alpha,
        effect_measures=tuple(args.effect_measures), seed=getattr(args, 'seed', 1))
    save_continuous(control, save_dir)
    return control


# ---------------------------------------------------------------------------
# one ROI's worth of work
# ---------------------------------------------------------------------------
def run_one_roi(args, runs, roi, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    print("=" * 72)
    print(f"ROI: {roi if roi is not None else 'all (pooled across ROIs)'}")
    print(f"save_dir: {save_dir}")

    labels = ptc.electrode_labels(
        runs, roi=roi, alpha=args.alpha, correction=args.correction,
        use_npz=args.use_npz, require_all=args.require_all)
    print(f"labels: {len(labels)} electrodes | "
          f"{labels['subject'].nunique()} subjects | "
          f"dropped {labels.attrs.get('n_dropped', 0)} (not in every run)")

    res = ptc.run_power_traces_conjunction(
        labels, n_perm=args.n_perm_null, seed=getattr(args, 'seed', 0),
        thresholds=tuple(args.thresholds))

    save_results(labels, res, save_dir)
    make_plots(labels, res, save_dir,
               roi_tag=f"  [{roi}]" if roi is not None else "")
    return labels, res


# ---------------------------------------------------------------------------
# orchestrator
# ---------------------------------------------------------------------------
def main(args):
    os.makedirs(args.save_dir, exist_ok=True)

    if args.data_source == 'synthetic':
        print("DATA SOURCE: synthetic (writes fake within-electrode ANOVA runs)")
        runs = make_synthetic_runs(
            args.save_dir, overlap=getattr(args, 'synthetic_overlap', 0.6),
            base_rate=getattr(args, 'synthetic_base_rate', 0.25),
            seed=getattr(args, 'seed', 0))
        args.run_dirs, args.run_dir = runs, None
    else:
        print("DATA SOURCE: real power_traces within-electrode ANOVA run(s)")

    runs = resolve_runs(args)
    if isinstance(runs, str):
        print(f"single 4-factor run: {runs}")
    else:
        for g, p in sorted(runs.items()):
            print(f"  {g}: {p}")

    available = rois_in_run(runs)
    print(f"ROIs present in the run(s): {available}")
    if args.rois in (None, 'all', ['all']):
        roi_list = [None]                      # one pooled analysis
    else:
        roi_list = list(args.rois)
        missing = [r for r in roi_list if r not in available]
        if missing:
            raise ValueError(f"requested ROI(s) {missing} are not in the run; "
                             f"available: {available}")

    out = {}
    for roi in roi_list:
        tag = roi if roi is not None else 'all_rois'
        roi_dir = os.path.join(args.save_dir, tag)
        labels, res = run_one_roi(args, runs, roi, roi_dir)

        control = None
        if getattr(args, 'run_continuous', False):
            print("-" * 72)
            print("continuous confound control (disjoint trial halves, "
                  "responsiveness-residualised, all effect measures)")
            control = run_continuous_control(args, labels, runs, roi_dir)

        write_summary(labels, res, roi_dir, alpha=args.alpha, control=control,
                      meta=dict(
                          data_source=args.data_source,
                          runs=(runs if isinstance(runs, str)
                                else {g: os.path.basename(p) for g, p in runs.items()}),
                          roi=tag, correction=args.correction, alpha=args.alpha,
                          require_all=args.require_all,
                          n_perm_null=args.n_perm_null,
                          thresholds=list(args.thresholds),
                          continuous_control=bool(getattr(args, 'run_continuous', False)),
                          save_dir=roi_dir))
        out[tag] = dict(labels=labels, results=res, continuous=control)
    return out
