"""Decoder feature_mode='channels': trial x time points as samples, channels as features."""
import numpy as np
import pytest
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from src.analysis.decoding.decoder import Decoder


def _data(effect, n_trials=80, n_ch=6, n_t=32, seed=0):
    """(trials, channels, time) noise; class 1 gets `effect` added on channel 0
    at every time point (a spatial pattern constant in time)."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_trials, n_ch, n_t))
    y = np.repeat([0, 1], n_trials // 2)
    x[y == 1, 0, :] += effect
    return x, y


def _accuracy(x, y, mode, window=16, step=8, seed=0):
    dec = Decoder({('a',): 0, ('b',): 1}, explained_variance=0.9, n_splits=4, n_repeats=2,
                  clf=LinearDiscriminantAnalysis(), clf_params={}, random_state=seed,
                  feature_mode=mode)
    cm = dec.cv_cm_jim_window_shuffle(x, y, normalize=None, obs_axs=0, time_axs=-1,
                                      window=window, step_size=step)
    # (n_windows, n_repeats, 2, 2) counts; every test trial is counted once
    assert np.allclose(cm.sum(axis=(-2, -1)), len(y))
    return np.trace(cm, axis1=-2, axis2=-1).sum() / cm.sum()


def test_channels_mode_decodes_a_spatial_pattern():
    x, y = _data(effect=0.6)
    assert _accuracy(x, y, 'channels') > 0.85


def test_channels_mode_is_at_chance_on_noise():
    accs = [_accuracy(*_data(effect=0.0, seed=s), 'channels', seed=s) for s in range(6)]
    assert abs(np.mean(accs) - 0.5) < 0.06


def test_channels_mode_without_windows_uses_every_time_point():
    x, y = _data(effect=0.6)
    assert _accuracy(x, y, 'channels', window=None) > 0.85


def test_default_mode_is_unchanged_and_bad_modes_raise():
    x, y = _data(effect=0.6)
    assert _accuracy(x, y, 'channels_x_time') > 0.7
    with pytest.raises(ValueError, match='feature_mode'):
        Decoder({('a',): 0, ('b',): 1}, feature_mode='time')
