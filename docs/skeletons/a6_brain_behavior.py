"""A6 skeleton — brain–behavior correlation (plan §6).

DROP-IN TARGET
--------------
A new `src/analysis/stats/stability_flexibility_brain_behavior.py` (or add to
the segregation module). Behavioral effects come from the existing behavioral
pipeline:
    src/analysis/stats/erin_linear_mixed_effects_model.py
    combinedData.csv  (produced by preproc/makeRawBehavioralData.py)

WHY THIS EXISTS
---------------
Tie the neural selectivity to the ACTUAL behavioral control adjustment, so the
substrates are shown to be functional, not incidental. Two levels, very
different power:
  * across subjects (n = subjects, low power),
  * within subject / single-trial (preferred, far more power).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def subject_level_brain_behavior(elec_labels, behavior):
    """Across-subject correlation of neural selectivity vs behavioral LWPC/LWPS.

    Parameters
    ----------
    elec_labels : per-electrode S/F labels (+ effect sizes) from A1; reduce to a
        per-subject neural summary (e.g. count of S electrodes, or mean stability
        effect) for each process.
    behavior : per-subject behavioral LWPC and LWPS magnitudes (the congruency-
        sequence and switch-proportion RT effects), extracted from the mixed
        model, NOT recomputed from scratch.

    Returns
    -------
    dict: corr_lwpc, p_lwpc, corr_lwps, p_lwps, n_subjects
    plus an explicit "underpowered at n=subjects" caveat in the docstring/output.

    Steps: build one row per subject (neural summary, behavioral magnitude);
    correlate matching pairs; report n.
    """
    raise NotImplementedError("A6: across-subject brain-behavior correlation")


def trialwise_brain_behavior(trial_df, group="LWPC"):
    """Within-subject single-trial link (preferred).

    Does trial-by-trial HG in the LWPC electrode group predict the trial-by-trial
    congruency-sequence RT adjustment (and LWPS group ↔ switch adjustment)?

    Parameters
    ----------
    trial_df : per-trial rows with subject, single-trial HG averaged over the
        selected electrode group, and the behavioral adjustment for that trial.
    group : 'LWPC' or 'LWPS' — selects both the electrode group and the MATCHING
        behavioral adjustment.

    Returns
    -------
    dict: slope, p (from a mixed model with a subject random effect), n_trials.

    SPECIFICITY CONTROL: also fit the CROSS pairing (LWPC HG ↔ switch adjustment)
    and confirm it is weaker than the matched pairing.

    Steps: statsmodels mixedlm, e.g.
        smf.mixedlm('rt_adjustment ~ hg_group', trial_df, groups='subject').fit()
    """
    raise NotImplementedError("A6: within-subject single-trial mixed model")
