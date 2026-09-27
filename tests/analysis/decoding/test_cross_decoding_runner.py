"""The A4 runner's environment handling, which all happens before any data loads.

`run_stability_flexibility_cross_decoding_dcc.py` turns environment variables
into module constants at import, so each case imports it in a fresh process
(without running `run_analysis`) and reads back SAVE_DIR / CONTRAST_MODE, or the
error it raises.
"""

import json
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
RUNNER = os.path.join(ROOT, 'dcc_scripts', 'decoding',
                      'run_stability_flexibility_cross_decoding_dcc.py')
CONDITION_CSV = '/x/anova_conjunction_window_0.0to1.5s_sig_lpfc_condition_none'


def _import_runner(**env):
    """(module constants, stderr) of the runner imported under `env`."""
    code = ("import json, runpy; m = runpy.run_path(%r); "
            "print(json.dumps({k: m[k] for k in ('SAVE_DIR', 'CONTRAST_MODE')}))" % RUNNER)
    full = {k: v for k, v in os.environ.items()
            if k not in ('CONTRAST_MODE', 'ANOVA_LABEL_EFFECT', 'SAVE_DIR', 'TRAIN_LABEL',
                         'TEST_LABEL', 'ELECTRODE_DEFINITION', 'ANOVA_LABELS_CSV',
                         'FDR_CORRECTION', 'ANALYSIS')}
    full['DATA_SOURCE'] = 'synthetic'
    full.update(env)
    done = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True,
                          text=True, env=full)
    if done.returncode:
        return None, done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1]), done.stderr


def test_the_contrast_mode_is_read_off_the_a1_folder():
    got, err = _import_runner(ELECTRODE_DEFINITION='csv', ANOVA_LABELS_CSV=CONDITION_CSV,
                              ANOVA_LABEL_EFFECT='union')
    assert got is not None, err
    assert got['CONTRAST_MODE'] == 'condition'
    assert '_csv_condition_' in got['SAVE_DIR']
    assert '__effect-union__' in got['SAVE_DIR']


def test_a_contradicting_contrast_mode_is_refused():
    got, err = _import_runner(ELECTRODE_DEFINITION='csv', ANOVA_LABELS_CSV=CONDITION_CSV,
                              CONTRAST_MODE='proportion')
    assert got is None and 'condition-mode A1 table' in err


def test_an_effect_name_from_the_other_mode_is_refused():
    """'lwpc' on a condition-mode table would select the congruency electrodes
    under an LWPC name."""
    got, err = _import_runner(ELECTRODE_DEFINITION='csv', ANOVA_LABELS_CSV=CONDITION_CSV,
                              ANOVA_LABEL_EFFECT='lwpc')
    assert got is None and 'names a proportion-mode population' in err


def test_the_table_is_ignored_off_the_csv_route():
    got, err = _import_runner(ELECTRODE_DEFINITION='anova', ANOVA_LABELS_CSV=CONDITION_CSV,
                              CONTRAST_MODE='condition')
    assert got is not None, err
    assert 'anova_label_selections' not in got['SAVE_DIR']


def test_a_single_pair_gets_its_own_folder():
    got, err = _import_runner(TRAIN_LABEL='congruency', TEST_LABEL='congruency')
    assert got is not None, err
    assert got['SAVE_DIR'].endswith(os.path.join('train_congruency_test_congruency'))


def test_the_in_job_anova_refuses_saved_flag_correction():
    got, err = _import_runner(DATA_SOURCE='real', EPOCHS_ROOT_FILE='epochs',
                              ELECTRODE_DEFINITION='anova', FDR_CORRECTION='flags')
    assert got is None and "must be 'fdr_bh' or 'none'" in err


def test_the_task_transfer_analysis_has_its_own_folder():
    got, err = _import_runner(ANALYSIS='task_transfer')
    assert got is not None, err
    assert os.path.join('task_transfer_lpfc_sig_w20s10', 'pooled_design_conditions') \
        in got['SAVE_DIR']
