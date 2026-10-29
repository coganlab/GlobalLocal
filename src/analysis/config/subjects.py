"""Named participant sets for the cluster launchers.

Each ``run_*_dcc.py`` entry point takes its subjects from here instead of a
hard-coded list, and a launcher picks a set with the ``SUBJECT_SET``
environment variable (``sbatch --export=ALL`` carries it into the job).

Subject IDs use the BIDS spelling (``D0107A``, ``D0139A``), not the recon or
lab-notebook spelling (``D0107``, ``D0139``).
"""
import os

SUBJECT_SETS = {
    # Every paper run up to 2026-10-29 used this set.
    'all24': ['D0057', 'D0059', 'D0063', 'D0065', 'D0069', 'D0077', 'D0090',
              'D0094', 'D0100', 'D0102', 'D0103', 'D0107A', 'D0110', 'D0116',
              'D0117', 'D0121', 'D0133', 'D0134', 'D0137', 'D0138', 'D0139A',
              'D0144', 'D0145', 'D0146'],
    # Quality-checked list, 2026-10-29. Relative to all24: adds D0130; drops
    # D0144 (missing block D) and D0117 (accuracy 0.63). D0071 (abnormal
    # high-gamma traces) was already out of all24.
    'good': ['D0057', 'D0059', 'D0063', 'D0065', 'D0069', 'D0077', 'D0090',
             'D0094', 'D0100', 'D0102', 'D0103', 'D0107A', 'D0110', 'D0116',
             'D0121', 'D0130', 'D0133', 'D0134', 'D0137', 'D0138', 'D0139A',
             'D0145', 'D0146'],
    # Accuracy below is per participant from combinedData.csv: correct trials
    # over task trials, the first trial of each block dropped (the behavioral
    # notebook's measure). The equal-cell-weight acc_mean in
    # ieeg_behavioral_subject_level_effects.csv is a different measure.
    #
    # good minus D0102, the only participant below 70 % (67.3). Next lowest:
    # D0065 70.5, D0069 73.9.
    'good_rawacc70': ['D0057', 'D0059', 'D0063', 'D0065', 'D0069', 'D0077',
                      'D0090', 'D0094', 'D0100', 'D0103', 'D0107A', 'D0110',
                      'D0116', 'D0121', 'D0130', 'D0133', 'D0134', 'D0137',
                      'D0138', 'D0139A', 'D0145', 'D0146'],
    # Sensitivity check: good minus D0102, D0065 and D0069 (67.3, 70.5, 73.9).
    # The three fall below 70 % only on the equal-cell-weight acc_mean (68.1,
    # 68.8, 69.7), which is how this set was first (mistakenly) chosen.
    'good_acc70': ['D0057', 'D0059', 'D0063', 'D0077', 'D0090', 'D0094',
                   'D0100', 'D0103', 'D0107A', 'D0110', 'D0116', 'D0121',
                   'D0130', 'D0133', 'D0134', 'D0137', 'D0138', 'D0139A',
                   'D0145', 'D0146'],
}

# The default keeps existing launchers and their output paths unchanged.
DEFAULT_SUBJECT_SET = 'all24'


def resolve_subject_set(name=None):
    """Return the subjects of a named set and the suffix for its output paths.

    ``name`` is a key of ``SUBJECT_SETS``; ``None`` (or blank) reads
    ``$SUBJECT_SET`` and falls back to ``DEFAULT_SUBJECT_SET``.

    The suffix is ``''`` for the default set, so runs on it keep writing where
    they always have, and ``'_subjects-<name>'`` otherwise. Append it to a
    run's save directory so that a run on another set never overwrites
    the default set's results.

    Raises ``ValueError`` for an unknown name, so a typo fails at submission
    instead of silently running the default set.
    """
    name = (name or os.environ.get('SUBJECT_SET') or '').strip() or DEFAULT_SUBJECT_SET
    if name not in SUBJECT_SETS:
        raise ValueError(f"Unknown SUBJECT_SET {name!r}; choose from {sorted(SUBJECT_SETS)}")
    suffix = '' if name == DEFAULT_SUBJECT_SET else f'_subjects-{name}'
    return list(SUBJECT_SETS[name]), suffix
