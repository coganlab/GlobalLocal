"""Runnable companion to `docs/stability_flexibility_data_flow.md`.

Every table and number printed in that document comes from this script. It uses
only synthetic data with **planted ground truth**, so it runs anywhere — no
cluster, no BIDS tree, no MNE, no iEEG data on disk.

    pip install numpy pandas scipy statsmodels scikit-learn
    python docs/examples/stability_flexibility_data_flow_demo.py            # everything
    python docs/examples/stability_flexibility_data_flow_demo.py a1 a2      # named stages

Stages: `design`, `a1`, `a2`, `a3`, `a4`, `a5`, `a6`, `split`.

Run it from the repo root (or `pip install -e .` first) so `src.analysis...`
imports resolve. Everything is seeded, so the numbers reproduce exactly.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make `src.analysis...` importable when this file is run directly from a clone
# that has not been `pip install -e .`-ed.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 40)


def banner(text):
    print(f"\n{'=' * 72}\n{text}\n{'=' * 72}")


# ---------------------------------------------------------------------------
# The fake dataset: 6 subjects x 8 electrodes x 240 trials
# ---------------------------------------------------------------------------
def make_mini(n_subj=6, n_elec=8, n_trials=240, seed=7):
    """One row per (electrode, trial) — the long-format table every stats
    analysis in the battery consumes.

    Ground truth planted per subject: electrodes 0-1 carry the LWPC (stability)
    interaction, 2-3 carry LWPS (flexibility), 4 carries BOTH, 5-7 carry nothing.
    A congruency MAIN effect is added to every electrode on purpose, so the
    analyses have something to *not* be fooled by.
    """
    rng = np.random.default_rng(seed)
    frames = []
    for s in range(n_subj):
        for e in range(n_elec):
            cpc = 0.55 if e in (0, 1, 4) else 0.0     # LWPC interaction amplitude
            sps = 0.55 if e in (2, 3, 4) else 0.0     # LWPS interaction amplitude
            # Block structure: each trial sits in a block with an incongruent
            # proportion and a switch proportion, and the condition frequencies
            # follow those proportions — the real design's ~75/25 cell imbalance.
            ip = rng.choice([25.0, 75.0], n_trials)
            sp = rng.choice([25.0, 75.0], n_trials)
            cong = np.where(rng.random(n_trials) < ip / 100.0, 'i', 'c')
            swt = np.where(rng.random(n_trials) < sp / 100.0, 's', 'r')
            c_code = np.where(cong == 'i', 1.0, -1.0)
            s_code = np.where(swt == 's', 1.0, -1.0)
            ip_code = np.where(ip == 75.0, 1.0, -1.0)
            sp_code = np.where(sp == 75.0, 1.0, -1.0)
            hg = (cpc * c_code * ip_code            # the LWPC interaction
                  + sps * s_code * sp_code          # the LWPS interaction
                  + 0.30 * c_code                   # a congruency MAIN effect
                  + rng.normal(0, 1.0, n_trials))   # noise
            frames.append(pd.DataFrame(dict(
                subject=s, electrode=f'{s}_{e}', congruency=cong, switchType=swt,
                incongruent_proportion=ip, switch_proportion=sp, hg=hg)))
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# design — why the estimator is a balanced difference-of-differences
# ---------------------------------------------------------------------------
def stage_design():
    banner("DESIGN — a pooled contrast leaks a main effect; the balanced "
           "difference-of-differences does not")
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm

    rng = np.random.default_rng(0)
    n = 20_000
    # 3x as many high-incongruent blocks as low -> the four cells are unequal in
    # BOTH factors, which is what makes a pooled contrast biased.
    ip = rng.choice([25.0, 75.0], n, p=[0.25, 0.75])
    cong = np.where(rng.random(n) < ip / 100.0, 'i', 'c')
    hg = 0.60 * np.where(cong == 'i', 1.0, -1.0) + rng.normal(0, 1, n)
    #    ^ a PURE congruency main effect. The true interaction is exactly zero.
    d = pd.DataFrame(dict(hg=hg, congruency=cong, incongruent_proportion=ip))

    cell = d.groupby(['congruency', 'incongruent_proportion'])['hg'].agg(['mean', 'size'])
    print(cell.round(3).to_string())

    pos = d[((d.congruency == 'i') & (d.incongruent_proportion == 75.0)) |
            ((d.congruency == 'c') & (d.incongruent_proportion == 25.0))]['hg']
    neg = d[((d.congruency == 'i') & (d.incongruent_proportion == 25.0)) |
            ((d.congruency == 'c') & (d.incongruent_proportion == 75.0))]['hg']
    m = cell['mean']
    dod = (m[('i', 75.0)] - m[('c', 75.0)]) - (m[('i', 25.0)] - m[('c', 25.0)])
    print('\n  true interaction                      =  0.000  (by construction)')
    print('  pooled  (+cells) - (-cells)           = %+.3f  <-- the main effect, leaked'
          % (pos.mean() - neg.mean()))
    print('  balanced equal-cell d-o-d             = %+.3f  <-- unbiased' % dod)

    aov = anova_lm(smf.ols('hg ~ C(congruency, Sum)*C(incongruent_proportion, Sum)',
                           data=d).fit(), typ=3)
    row = [i for i in aov.index if ':' in i][0]
    print('  sum-coded Type III interaction        =  F=%.3f, p=%.3f  <-- n.s., correct'
          % (aov.loc[row, 'F'], aov.loc[row, 'PR(>F)']))


# ---------------------------------------------------------------------------
# A1 — per-electrode interaction ANOVA
# ---------------------------------------------------------------------------
def stage_a1(df):
    from src.analysis.stats.stability_flexibility_segregation import \
        per_electrode_anova_labels

    banner("A1 — per_electrode_anova_labels: four interactions, FDR'd across electrodes")
    print('input (long single-trial table), shape %s:' % (df.shape,))
    print(df.head(6).to_string(index=False))

    labels = per_electrode_anova_labels(df, alpha=0.05, contrast_mode='proportion')
    cols = ['subject', 'electrode', 'F_cpc', 'p_cpc', 'q_cpc', 'cpc_sign',
            'F_sps', 'q_sps', 'q_cps', 'q_spc', 'CPC', 'SPS', 'CPS', 'SPC', 'S', 'F']
    print('\noutput (one row per electrode), subject 0:')
    print(labels[cols].head(8).round(4).to_string(index=False))
    print('\nselected: %s | both = %d | n_electrodes = %d'
          % ({k: int(labels[k].sum()) for k in ('CPC', 'SPS', 'CPS', 'SPC')},
             int(((labels.CPC == 1) & (labels.SPS == 1)).sum()), len(labels)))
    print('ground truth per subject: 3 CPC (elecs 0,1,4), 3 SPS (2,3,4), 1 both (4), '
          '0 cross')
    return labels


# ---------------------------------------------------------------------------
# A2 — conjunction (categorical) + correlation (continuous)
# ---------------------------------------------------------------------------
def stage_a2(df, labels):
    from src.analysis.stats.stability_flexibility_segregation import (
        cmh_conjunction, conjunction_permutation_null, conjunction_threshold_sweep,
        compute_sensitivities, add_responsiveness, prepare_continuous,
        subject_clustered_corr, per_electrode_anova_labels)

    banner("A2a — categorical: is 'both' more or less common than chance?")
    cmh = cmh_conjunction(labels)
    print('per-subject 2x2 tables:')
    print(cmh['per_subject'].to_string(index=False))
    print('\npooled 2x2 (descriptive only — ignores subject nesting):\n%s' % cmh['pooled_table'])
    print('\nMH odds ratio = %.3f   CMH p = %.4f   OR 95%% CI = (%.3f, %.3f)'
          % (cmh['mh_odds_ratio'], cmh['cmh'].pvalue, *cmh['or_95ci']))

    null = conjunction_permutation_null(labels, n_perm=2000, seed=0)
    print('permutation null (F shuffled WITHIN subject): observed=%d  null_mean=%.2f  '
          'z=%.2f  p=%.4f'
          % (null['observed'], null['null'].mean(), null['z'], null['p_two_sided']))

    banner("A2a' — threshold sweep: is the verdict stable across cutoffs?")
    sweep = conjunction_threshold_sweep(
        lambda t: per_electrode_anova_labels(df, alpha=t, contrast_mode='proportion'),
        [0.01, 0.05, 0.10])
    print(sweep.round(4).to_string(index=False))

    banner("A2b — continuous: correlate stability vs flexibility sensitivity")
    sens = compute_sensitivities(df, n_splits=20, seed=0, contrast_mode='proportion')
    print('compute_sensitivities (x and y from DISJOINT trial halves):')
    print(sens.head(6).round(4).to_string(index=False))

    sens = add_responsiveness(sens, df)
    cont = prepare_continuous(sens)
    print('\nafter add_responsiveness + prepare_continuous '
          '(x1/y1 = responsiveness-residualised, *_resid = within-subject centred):')
    print(cont.head(4).round(4).to_string(index=False))

    res = subject_clustered_corr(cont, n_perm=2000, seed=1)
    print('\nsubject_clustered_corr: corr = %+.3f   p = %.5f   n_elec = %d   n_subj = %d'
          % (res['corr'], res['p'], res['n_electrodes'], res['n_subjects']))
    return cmh, res


# ---------------------------------------------------------------------------
# A3 — anatomy
# ---------------------------------------------------------------------------
def stage_a3():
    from src.analysis.stats.stability_flexibility_anatomy import (
        _synthetic_anatomy, attach_roi, build_coverage_matrix,
        roi_group_enrichment_test)

    banner("A3 — anatomy: coverage-conditioned ROI enrichment")
    labels, e2r = _synthetic_anatomy(n_subj=12, seed=0, enrichment=0.6)
    print('labels (from A1) + electrode->ROI map:')
    print(labels.head(4).to_string(index=False))
    print('electrodes_to_rois:', {k: str(v) for k, v in list(e2r.items())[:4]}, '...')

    lab_roi = attach_roi(labels, e2r)
    print('\nattach_roi adds `roi` and the derived `group`:')
    print(lab_roi.head(5).to_string(index=False))

    cov = build_coverage_matrix(lab_roi)
    print('\nbuild_coverage_matrix (subject x ROI — who is even wired where):')
    print(cov.head(4).to_string())

    res = roi_group_enrichment_test(lab_roi, cov, min_subjects=3, n_perm=2000, seed=0)
    print('\ngroup x ROI contingency:')
    print(res['contingency'].to_string())
    print('\nper-ROI coverage (n subjects):')
    print(res['per_roi_coverage'].to_string())
    print('\nchi2 = %.3f   permutation p = %.4f   n_electrodes = %d   rois_tested = %s'
          % (res['observed_stat'], res['p'], res['n_electrodes'], res['rois_tested']))

    lab0, e2r0 = _synthetic_anatomy(n_subj=12, seed=0, enrichment=0.0)
    lr0 = attach_roi(lab0, e2r0)
    r0 = roi_group_enrichment_test(lr0, build_coverage_matrix(lr0), min_subjects=3,
                                   n_perm=2000, seed=0)
    print('NULL control (enrichment=0.0):  chi2 = %.3f   p = %.4f   <-- must be n.s.'
          % (r0['observed_stat'], r0['p']))
    return res


# ---------------------------------------------------------------------------
# A4 — cross-decoding
# ---------------------------------------------------------------------------
def stage_a4():
    import importlib.util

    from src.analysis.decoding.cross_decoding import (
        build_cross_decoding_arrays, circular_decode_for_group, filter_conditions,
        is_circular_decode, synthetic_roi_labeled_arrays)

    STAB = [['_i_'], ['_c_']]        # congruency
    FLEX = [['_s_'], ['_r_']]        # switch type

    banner("A4 — cross-decoding: one shared code, or two orthogonal codes?")
    arrs = synthetic_roi_labeled_arrays(code='shared', seed=1)
    conds = list(arrs['synthetic'])
    print('input is the ordinary ROI LabeledArray — the SAME structure the normal')
    print('decoding path consumes: {roi: {condition: (trials, channels, time)}}')
    print('  %d conditions, e.g. %s' % (len(conds), conds[:2]))
    print('  each: %s' % (arrs['synthetic'][conds[0]].shape,))

    out = build_cross_decoding_arrays(arrs, 'synthetic', STAB, FLEX)
    print('\nbuild_cross_decoding_arrays -> TWO labellings of the SAME trials:')
    print('  data          %s' % (out['data'].shape,))
    print('  labels_train  %s  (congruency, cats=%s)'
          % (out['labels_train'][:12], out['cats_train']))
    print('  labels_test   %s  (switchType, cats=%s)'
          % (out['labels_test'][:12], out['cats_test']))
    print('  strata        %s  (%d source conditions)'
          % (out['strata'][:12], len(set(out['strata']))))
    pairs, counts = np.unique(
        np.stack([out['labels_train'], out['labels_test']], axis=1),
        axis=0, return_counts=True)
    print('  joint (train, test) cells: %s -> counts %s  <-- the contrasts CROSS'
          % ([tuple(p) for p in pairs], list(counts)))

    sub = filter_conditions(arrs, 'synthetic', '25inc')
    print('\nfilter_conditions(..., "25inc") -> %d of %d conditions '
          '(this is how the within-block 2x2 is expressed)'
          % (len(sub['synthetic']), len(conds)))

    print("\ndouble-dipping guard — which decode each electrode group must NOT run:")
    for g in ('CPC', 'SPS', 'CPS', 'SPC'):
        print('  circular_decode_for_group(%-5r) = %s' % (g, circular_decode_for_group(g)))
    print('  is_circular_decode("CPC", "congruency", "incongruent_proportion") =',
          is_circular_decode('CPC', 'congruency', 'incongruent_proportion'))
    print('  is_circular_decode("CPC", "switchType", "switch_proportion")      =',
          is_circular_decode('CPC', 'switchType', 'switch_proportion'))

    # The decode itself goes through the project Decoder, which needs `ieeg`.
    # Everything above is pure numpy, so this stage still demonstrates the data
    # flow off-cluster; only the ground-truth check needs the full environment.
    if importlib.util.find_spec('ieeg') is None:
        print('\n[`ieeg` not installed — skipping the decode itself. The arrays above')
        print(' are exactly what `Decoder.cv_cm_jim_window_shuffle(data, labels_train,')
        print(' labels_test=..., stratify_labels=strata)` consumes.]')
        return

    from src.analysis.decoding.accuracy_stats import compute_accuracies
    from src.analysis.decoding.cross_decoding import run_cross_decoding
    # Deliberately small (20 channels, 20 trials/cell, 3 CV folds x 2 repeats) —
    # enough to recover the planted answer in ~45 s per decode.
    small = dict(n_channels=20, n_trials_per_cell=20, n_time=32)
    print('\nground truth — a shared code transfers, an orthogonal one does not')
    print('(~45 s per decode; this is the only slow stage in the script):')
    for code in ('shared', 'orthogonal'):
        a = synthetic_roi_labeled_arrays(code=code, seed=1, **small)
        for name, (tr, te) in (('within congruency', (STAB, STAB)),
                               ('cross stab->flex ', (STAB, FLEX))):
            res = run_cross_decoding(a, 'synthetic', tr, te, n_splits=3,
                                     n_repeats=2, window=16, step_size=16)
            acc_t, acc_s = compute_accuracies(res['cm_true'], res['cm_shuffle'])
            print('  %-10s %s  true = %.3f   shuffle = %.3f'
                  % (code, name, acc_t.mean(), acc_s.mean()))


# ---------------------------------------------------------------------------
# A5 — timing
# ---------------------------------------------------------------------------
def stage_a5():
    from src.analysis.stats.stability_flexibility_timing import (
        _synthetic_timecourse_df, interaction_time_course, onset_50pct_peak,
        peak_latency, jackknife_onset_difference)

    banner("A5 — timing: does one process's information arise earlier?")
    tdf, times = _synthetic_timecourse_df(n_subj=8, seed=0, n_time=40,
                                          stab_onset=0.20, flex_onset=0.40)
    print('input: same long table, but `hg` holds a TIME COURSE per trial')
    print('  shape %s | hg cell = %s of len %d'
          % (tdf.shape, type(tdf.hg.iloc[0]).__name__, len(tdf.hg.iloc[0])))

    _, pc = interaction_time_course(tdf, 'stability', times=times)
    _, ps = interaction_time_course(tdf, 'flexibility', times=times)
    print('\ninteraction_time_course -> one d-o-d value per time bin:')
    print('  times[:8] =', np.round(times[:8], 3))
    print('  lwpc[:8]  =', np.round(pc[:8], 3))
    print('  lwpc[12:20] =', np.round(pc[12:20], 3), ' <-- the rising flank')
    print('\nonset (50%% of own peak): lwpc = %.3f s   lwps = %.3f s'
          % (onset_50pct_peak(times, pc), onset_50pct_peak(times, ps)))
    print('peak latency            : lwpc = %.3f s   lwps = %.3f s'
          % (peak_latency(times, pc), peak_latency(times, ps)))

    jk = jackknife_onset_difference(tdf, times=times)
    print('\njackknife_onset_difference (Ulrich-Miller):')
    print('  diff (LWPC - LWPS) = %+.4f s   se = %.4f   ci = (%+.4f, %+.4f)'
          % (jk['diff'], jk['se'], *jk['ci']))
    print('  t_raw = %.2f  ->  t_corrected = %.2f  (df = %d)   p = %.3g'
          % (jk['t_raw'], jk['t_corrected'], jk['df'], jk['p']))
    print('  peak_diff = %+.4f s  (must agree in sign with the onset diff)'
          % jk['peak_diff'])

    tdf2, times2 = _synthetic_timecourse_df(n_subj=8, seed=0, n_time=40,
                                            stab_onset=0.40, flex_onset=0.20)
    jk2 = jackknife_onset_difference(tdf2, times=times2)
    print('\nFALSIFICATION — plant the REVERSE ordering; the sign must flip:')
    print('  diff = %+.4f s (was %+.4f s), p = %.3g' % (jk2['diff'], jk['diff'], jk2['p']))


# ---------------------------------------------------------------------------
# A6 — brain-behavior
# ---------------------------------------------------------------------------
def stage_a6():
    from src.analysis.stats.stability_flexibility_brain_behavior import (
        _synthetic_brain_behavior, neural_summary_by_subject,
        subject_level_brain_behavior, trialwise_brain_behavior)

    banner("A6 — brain-behavior: is the selectivity functional?")
    elec_labels, behavior, trial_df = _synthetic_brain_behavior(n_subj=16, seed=0)
    print('three inputs:')
    print('  elec_labels  (A1 output)      :', list(elec_labels.columns))
    print('  behavior     (per subject)    :', list(behavior.columns))
    print('  trial_df     (per trial)      :', list(trial_df.columns))
    print('\nbehavior:'); print(behavior.head(3).round(4).to_string(index=False))
    print('\ntrial_df:'); print(trial_df.head(3).round(4).to_string(index=False))

    print('\nneural_summary_by_subject:')
    print(neural_summary_by_subject(elec_labels).head(4).round(4).to_string(index=False))

    sl = subject_level_brain_behavior(elec_labels, behavior, neural='count')
    print('\nLEVEL 1 — across subjects (n = %d):' % sl['n_subjects'])
    print('  matched: corr_lwpc = %+.3f (p = %.4f)   corr_lwps = %+.3f (p = %.4f)'
          % (sl['corr_lwpc'], sl['p_lwpc'], sl['corr_lwps'], sl['p_lwps']))
    print('  cross  : stab->lwps = %+.3f (p = %.3f)  flex->lwpc = %+.3f (p = %.3f)'
          % (sl['corr_cross_stab_lwps'], sl['p_cross_stab_lwps'],
             sl['corr_cross_flex_lwpc'], sl['p_cross_flex_lwpc']))
    print('  caveat:', sl['caveat'])

    print('\nLEVEL 2 — within subject, single trial (the powered test):')
    for g, col in (('LWPC', 'hg_lwpc'), ('LWPS', 'hg_lwps')):
        tw = trialwise_brain_behavior(trial_df, group=g, hg_col=col)
        print('  %s: matched slope = %+.4f (z = %.1f)   cross slope = %+.4f (z = %.1f)'
              '   specificity_ok = %s'
              % (g, tw['slope'], tw['z'], tw['slope_cross'], tw['z_cross'],
                 tw['specificity_ok']))


# ---------------------------------------------------------------------------
# split — the decoding <-> electrode-definition circularity control
# ---------------------------------------------------------------------------
def stage_split():
    from src.analysis.decoding.trial_splitting import (
        stratified_trial_split, strata_key_from_metadata, select_responsive_channels)

    banner("Disjoint trial split — define electrodes on P_def, decode on P_dec")
    meta = pd.DataFrame(dict(
        congruency=['c', 'c', 'i', 'i', 'c', 'i', 'c', 'i', 'c', 'i', 'c', 'i'],
        switchType=['s', 'r', 's', 'r', 's', 'r', 's', 'r', 's', 'r', 's', 'r'],
        blockType=['A'] * 6 + ['B'] * 6))
    print('epochs.metadata (12 trials):')
    print(meta.to_string())

    strata = strata_key_from_metadata(meta, ['congruency', 'switchType', 'blockType'])
    print('\nstrata_key_from_metadata ->', [str(s) for s in strata])

    p_def, p_dec = stratified_trial_split(strata, frac_def=0.5, seed=0)
    print('P_def (define electrodes) =', sorted(int(i) for i in p_def))
    print('P_dec (decode)            =', sorted(int(i) for i in p_dec))
    print('disjoint = %s | together they cover every trial = %s'
          % (set(p_def).isdisjoint(p_dec),
             sorted(set(int(i) for i in p_def) | set(int(i) for i in p_dec))
             == list(range(len(meta)))))

    rng = np.random.default_rng(0)
    window_means = rng.normal(0, 1, (200, 5))
    baseline_means = rng.normal(0, 1, (200, 5))
    window_means[:, 1] += 0.8                      # only channel 1 is responsive
    window_means[:, 4] = 0.0                       # channel 4 is dead
    baseline_means[:, 4] = 0.0
    keep = select_responsive_channels(window_means, baseline_means, alpha=0.05)
    print('\nselect_responsive_channels (run on P_def only) ->', keep)
    print('  channel 1 planted responsive -> kept; channel 4 zero-variance -> dropped')


STAGES = ['design', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'split']


def main(argv):
    wanted = [a.lower() for a in argv[1:]] or list(STAGES)
    unknown = [w for w in wanted if w not in STAGES]
    if unknown:
        raise SystemExit(f"unknown stage(s) {unknown}; choose from {STAGES}")

    df = make_mini()
    labels = None
    for name in wanted:
        if name == 'design':
            stage_design()
        elif name == 'a1':
            labels = stage_a1(df)
        elif name == 'a2':
            if labels is None:
                labels = stage_a1(df)
            stage_a2(df, labels)
        elif name == 'a3':
            stage_a3()
        elif name == 'a4':
            stage_a4()
        elif name == 'a5':
            stage_a5()
        elif name == 'a6':
            stage_a6()
        elif name == 'split':
            stage_split()


if __name__ == '__main__':
    main(sys.argv)
