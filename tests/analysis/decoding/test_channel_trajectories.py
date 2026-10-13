"""Channel-space PCA trajectories (feature_mode='channels')."""
import numpy as np
import mne

from src.analysis.decoding.plots.trajectories import (
    channel_pca_trajectories, plot_channel_pca_3d_trajectory)
from src.analysis.utils.labeled_array_utils import (
    make_bootstrapped_roi_labeled_arrays_with_nan_trials_removed_for_each_channel)


def _planted(n_trials=40, n_ch=8, n_t=64, seed=0):
    """A response shared by both conditions along channel pattern `shared`, and
    a condition difference along the orthogonal pattern `diff`."""
    rng = np.random.default_rng(seed)
    shared, diff = np.zeros(n_ch), np.zeros(n_ch)
    shared[:4], diff[4:] = 1.0, 1.0
    ramp = np.linspace(0, 1, n_t)
    y = np.repeat([0, 1], n_trials // 2)
    x = 0.1 * rng.normal(size=(n_trials, n_ch, n_t))
    x += 3.0 * shared[None, :, None] * ramp
    x += np.where(y == 1, 0.5, -0.5)[:, None, None] * diff[None, :, None] * ramp
    return x, y, shared / 2, diff / 2


def test_pc1_is_the_shared_response_unless_the_condition_mean_is_removed():
    x, y, shared, diff = _planted()
    _, pca = channel_pca_trajectories(x, y, window_size=8, step_size=8)
    assert abs(pca.components_[0] @ shared) > 0.95
    _, pca_rm = channel_pca_trajectories(x, y, window_size=8, step_size=8,
                                         remove_condition_mean=True)
    assert abs(pca_rm.components_[0] @ diff) > 0.95


def test_trajectory_shapes_and_padding():
    x, y, _, _ = _planted(n_ch=2)
    traj, pca = channel_pca_trajectories(x, y, window_size=16, step_size=16)
    assert set(traj) == {0, 1}
    assert traj[0].shape == (4, 3)
    assert np.isnan(traj[0][:, 2]).all()      # two channels: only two PCs


def test_plot_channel_pca_3d_trajectory_writes_its_figure(tmp_path):
    mne.set_log_level('ERROR')
    rng = np.random.default_rng(0)

    def ep(n):
        info = mne.create_info([f'CH{i}' for i in range(4)], 100., 'eeg')
        return mne.EpochsArray(rng.normal(size=(n, 4, 50)), info, tmin=-0.2)

    subs = {s: {c: {'HG_ev1_power_rescaled': ep(10)} for c in ('Stimulus_c', 'Stimulus_i')}
            for s in ('sub1', 'sub2')}
    el = {'lpfc': {'sub1': ['CH0', 'CH1'], 'sub2': ['CH2', 'CH3']}}
    arrays = make_bootstrapped_roi_labeled_arrays_with_nan_trials_removed_for_each_channel(
        ['lpfc'], subs, ['Stimulus_c', 'Stimulus_i'], ['sub1', 'sub2'], el,
        n_bootstraps=1, chans_axs=1, time_axs=-1, n_jobs=1, random_state=0)
    out = plot_channel_pca_3d_trajectory(
        {'lpfc': arrays['lpfc'][0]}, 'lpfc', [['Stimulus_c'], ['Stimulus_i']],
        {('Stimulus_c',): 0, ('Stimulus_i',): 1}, str(tmp_path),
        window_size=10, step_size=10, sampling_rate=100)
    assert out is not None
    assert (tmp_path / 'channel_pca_3d_trajectory_lpfc.png').exists()
