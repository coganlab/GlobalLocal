"""Window geometry for the multi-panel ``mne.viz.Brain`` figures.

``hemi='split'`` puts one hemisphere per subplot, but ``Brain``'s ``size`` is the
size of the whole window. Left at MNE's default the two hemispheres end up in
400x800 viewports, overrun them, and touch in the middle -- these tests pin the
per-panel sizing and the per-panel zoom that fix it.

Skipped wherever the surface stack (mne + ieeg) isn't installed, like the rest of
the renderer's tests.
"""
import pytest

jim_mri = pytest.importorskip("src.analysis.vis.jim_mri",
                              reason="needs the mne/ieeg surface stack")


class _Camera:
    def __init__(self):
        self.zooms = []

    def zoom(self, value):
        self.zooms.append(value)


class _Plotter:
    """The slice of ``pyvista.Plotter`` that ``zoom_brain_panels`` touches."""

    def __init__(self, shape):
        self.shape = shape
        self.camera = _Camera()
        self.visited = []

    def subplot(self, *index):
        self.visited.append(index)


class _Brain:
    def __init__(self, shape):
        self.plotter = _Plotter(shape)


def test_split_gets_one_square_panel_per_hemisphere():
    panel = jim_mri.BRAIN_PANEL_SIZE
    assert jim_mri._brain_window_size('split') == (2 * panel, panel)
    for hemi in ('both', 'lh', 'rh'):
        assert jim_mri._brain_window_size(hemi) == (panel, panel)
    assert jim_mri._brain_window_size('split', panel=500) == (1000, 500)


def test_zoom_visits_every_panel():
    fig = _Brain((1, 2))
    jim_mri.zoom_brain_panels(fig, 0.7)
    # both hemispheres zoomed, and the first panel left active afterwards
    assert fig.plotter.camera.zooms == [0.7, 0.7]
    assert fig.plotter.visited == [(0, 0), (0, 1), (0, 0)]


def test_zoom_is_a_noop_without_a_factor():
    for zoom in (None, 1.0, 0):
        fig = _Brain((1, 2))
        jim_mri.zoom_brain_panels(fig, zoom)
        assert fig.plotter.camera.zooms == []


def test_zoom_survives_a_backend_without_panel_cameras():
    class _NoPanels(_Plotter):
        def subplot(self, *index):
            raise RuntimeError("no such panel")

    fig = _Brain((1, 2))
    fig.plotter = _NoPanels((1, 2))
    jim_mri.zoom_brain_panels(fig, 0.7)          # warns, does not raise
    jim_mri.zoom_brain_panels(object(), 0.7)     # no plotter at all
