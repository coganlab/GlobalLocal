"""`--all-subjects` must not import the sbatch submit script.

run_power_traces_dcc.py raises at import time unless CONDITION_LABEL,
EPOCHS_ROOT_FILE and ANOVA_UNIT are set. Those belong to the submit path, so
reading a subject list through an import made the diagnostic unrunnable without
inventing a condition label.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from dcc_scripts.power.diagnose_block_effects import (  # noqa: E402
    subjects_from_submit_script,
)


def test_reads_the_live_subject_list(monkeypatch):
    for var in ('CONDITION_LABEL', 'EPOCHS_ROOT_FILE', 'ANOVA_UNIT'):
        monkeypatch.delenv(var, raising=False)

    subjects = subjects_from_submit_script()

    assert len(subjects) >= 2
    assert all(isinstance(s, str) and s.startswith('D') for s in subjects)
    assert len(set(subjects)) == len(subjects)


def test_later_assignment_wins_and_comments_are_ignored(tmp_path):
    script = tmp_path / "submit.py"
    script.write_text(
        "# SUBJECTS = ['D9999']\n"
        "SUBJECTS = ['D0001', 'D0002']\n"
        "SUBJECTS = ['D0003']\n"
    )

    assert subjects_from_submit_script(script) == ['D0003']


def test_missing_assignment_is_an_explicit_error(tmp_path):
    script = tmp_path / "submit.py"
    script.write_text("OTHER = 1\n")

    with pytest.raises(RuntimeError, match="no top-level SUBJECTS"):
        subjects_from_submit_script(script)
