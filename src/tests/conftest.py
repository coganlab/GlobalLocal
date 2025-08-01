import pytest
import numpy as np
import mne
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Set random seed for reproducible tests
np.random.seed(42)

@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory):
    """Create a temporary directory for test data."""
    return tmp_path_factory.mktemp("test_data")

@pytest.fixture
def sample_info():
    """Create a sample MNE Info object."""
    ch_names = ['CH0', 'CH1', 'CH2', 'CH3', 'CH4']
    ch_types = ['eeg'] * 5
    info = mne.create_info(ch_names=ch_names, sfreq=1000, ch_types=ch_types)
    return info

@pytest.fixture
def create_epochs():
    """Factory fixture for creating MNE Epochs with custom parameters."""
    def _create_epochs(n_epochs=10, n_channels=5, n_times=100, sfreq=1000):
        ch_names = [f'CH{i}' for i in range(n_channels)]
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
        
        # Create times array
        times = np.arange(n_times) / sfreq - 0.5
        
        # Create random data
        data = np.random.randn(n_epochs, n_channels, n_times)
        
        # Create epochs
        epochs = mne.EpochsArray(data, info, tmin=times[0])
        return epochs
    
    return _create_epochs

@pytest.fixture
def minimal_subjects_data(create_epochs):
    """Create minimal subjects data for quick tests."""
    return {
        'sub1': {
            'cond1': {'HG_ev1_power_rescaled': create_epochs(n_epochs=5)},
            'cond2': {'HG_ev1_power_rescaled': create_epochs(n_epochs=7)}
        },
        'sub2': {
            'cond1': {'HG_ev1_power_rescaled': create_epochs(n_epochs=6)},
            'cond2': {'HG_ev1_power_rescaled': create_epochs(n_epochs=4)}
        }
    }

@pytest.fixture
def simple_electrodes_mapping():
    """Create a simple electrodes mapping."""
    return {
        'roi1': {
            'sub1': ['CH0', 'CH1'],
            'sub2': ['CH2', 'CH3']
        }
    }

# Pytest hooks for custom behavior
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "mne: tests that require MNE-Python")
    config.addinivalue_line("markers", "requires_display: tests that require a display")

def pytest_collection_modifyitems(config, items):
    """Skip tests that require display if no display is available."""
    if not os.environ.get('DISPLAY'):
        skip_display = pytest.mark.skip(reason="Test requires display")
        for item in items:
            if "requires_display" in item.keywords:
                item.add_marker(skip_display)