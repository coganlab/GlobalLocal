import pytest

from src.analysis.config.subjects import (
    DEFAULT_SUBJECT_SET, SUBJECT_SETS, resolve_subject_set)


def test_default_set_has_no_suffix(monkeypatch):
    monkeypatch.delenv('SUBJECT_SET', raising=False)
    subjects, suffix = resolve_subject_set()
    assert subjects == SUBJECT_SETS[DEFAULT_SUBJECT_SET]
    assert suffix == ''


def test_env_var_picks_set_and_names_output(monkeypatch):
    monkeypatch.setenv('SUBJECT_SET', 'good')
    subjects, suffix = resolve_subject_set()
    assert suffix == '_subjects-good'
    assert 'D0130' in subjects
    assert not {'D0071', 'D0117', 'D0144'} & set(subjects)


def test_acc70_is_good_minus_low_accuracy():
    assert set(SUBJECT_SETS['good']) - set(SUBJECT_SETS['good_acc70']) == {'D0102', 'D0065', 'D0069'}
    assert set(SUBJECT_SETS['good_acc70']) <= set(SUBJECT_SETS['good'])
    assert set(SUBJECT_SETS['good']) - set(SUBJECT_SETS['good_rawacc70']) == {'D0102'}


def test_explicit_name_beats_env_and_returns_a_copy(monkeypatch):
    monkeypatch.setenv('SUBJECT_SET', 'good')
    subjects, suffix = resolve_subject_set('all24')
    assert suffix == ''
    subjects.append('D9999')
    assert 'D9999' not in SUBJECT_SETS['all24']


def test_unknown_set_raises(monkeypatch):
    monkeypatch.setenv('SUBJECT_SET', 'goood')
    with pytest.raises(ValueError, match='goood'):
        resolve_subject_set()
