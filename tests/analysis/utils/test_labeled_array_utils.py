import pytest
import numpy as np
import mne
from unittest.mock import Mock, MagicMock, patch
import sys
import os

# Add the project root to the path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from src.analysis.utils.labeled_array_utils import (
    detect_data_type,
    get_epochs_data_for_sub_and_condition_name_and_electrodes_from_subjects_mne_objects,
    get_epochs_tfr_data_for_sub_and_condition_name_and_electrodes_from_subjects_tfr_objects,
    get_max_trials_per_condition,
    make_subject_labeled_array,
    create_subject_labeled_array_from_dict,
    concatenate_subject_labeled_arrays,
    put_data_in_labeled_array_per_roi_subject,
    remove_nans_from_labeled_array,
    remove_nans_from_all_roi_labeled_arrays,
    concatenate_conditions_by_string,
    get_data_in_time_range
)

# Fixtures for creating mock data
@pytest.fixture
def mock_epochs():
    """Create a mock MNE Epochs object."""
    # Create fake data
    n_epochs = 10
    n_channels = 10
    n_times = 100
    sfreq = 1000
    
    # Create info object
    ch_names = [f'CH{i}' for i in range(n_channels)]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='seeg')
    
    # Create epochs array
    data = np.random.randn(n_epochs, n_channels, n_times)
    epochs = mne.EpochsArray(data, info)
    
    return epochs

@pytest.fixture
def mock_epochs_tfr():
    """Create a mock MNE EpochsTFR object."""
    mock_tfr = Mock(spec=mne.time_frequency.EpochsTFR)
    mock_tfr.get_data.return_value = np.random.randn(10, 10, 100, 20)  # 4D data
    mock_tfr.times = np.linspace(-0.5, 0.5, 100)
    mock_tfr.freqs = np.linspace(1, 40, 20)
    mock_tfr.copy.return_value = mock_tfr
    mock_tfr.pick.return_value = mock_tfr
    return mock_tfr

@pytest.fixture
def subjects_mne_objects(mock_epochs):
    """Create a mock subjects MNE objects dictionary."""
    return {
        'sub1': {
            'condition1': {'HG_ev1_power_rescaled': mock_epochs.copy()},
            'condition2': {'HG_ev1_power_rescaled': mock_epochs.copy()}
        },
        'sub2': {
            'condition1': {'HG_ev1_power_rescaled': mock_epochs.copy()},
            'condition2': {'HG_ev1_power_rescaled': mock_epochs.copy()}
        }
    }

@pytest.fixture
def subjects_tfr_objects(mock_epochs_tfr):
    """Create a mock subjects TFR objects dictionary."""
    return {
        'sub1': {
            'condition1': mock_epochs_tfr,
            'condition2': mock_epochs_tfr
        },
        'sub2': {
            'condition1': mock_epochs_tfr,
            'condition2': mock_epochs_tfr
        }
    }

@pytest.fixture
def electrodes_per_subject_roi():
    """Create mock electrodes per subject ROI dictionary."""
    return {
        'roi1': {
            'sub1': ['CH1', 'CH2', 'CH3'],
            'sub2': ['CH1', 'CH2', 'CH3']
        },
        'roi2': {
            'sub1': ['CH4', 'CH5', 'CH6'],
            'sub2': ['CH4', 'CH5', 'CH6']
        }
    }

# Test detect_data_type
class TestDetectDataType:
    def test_detect_epochs_type(self, subjects_mne_objects):
        """Test detection of regular Epochs objects."""
        data_type = detect_data_type(subjects_mne_objects)
        assert data_type == 'Epochs'
    
    def test_detect_epochs_tfr_type(self, subjects_tfr_objects):
        """Test detection of EpochsTFR objects."""
        data_type = detect_data_type(subjects_tfr_objects)
        assert data_type == 'EpochsTFR'
    
    def test_detect_data_type_error(self):
        """Test error when data type cannot be determined."""
        invalid_data = {'sub1': {'cond1': {'HG_ev1_power_rescaled': None}}}
        with pytest.raises(ValueError, match="Could not determine data type"):
            detect_data_type(invalid_data)

# Test get_epochs_data functions
class TestGetEpochsData:
    def test_get_epochs_data_for_sub_and_condition(self, subjects_mne_objects):
        """Test getting epochs data for specific subject and condition."""
        electrodes = ['CH1', 'CH2']
        result = get_epochs_data_for_sub_and_condition_name_and_electrodes_from_subjects_mne_objects(
            subjects_mne_objects, 'condition1', 'sub1', electrodes
        )
        assert isinstance(result, mne.BaseEpochs)
        assert all(ch in result.ch_names for ch in electrodes)
    
    def test_get_epochs_tfr_data_for_sub_and_condition(self, subjects_tfr_objects):
        """Test getting epochs TFR data for specific subject and condition."""
        electrodes = ['CH1', 'CH2']
        result = get_epochs_tfr_data_for_sub_and_condition_name_and_electrodes_from_subjects_tfr_objects(
            subjects_tfr_objects, 'condition1', 'sub1', electrodes
        )
        assert result is not None

# Test get_max_trials_per_condition
class TestGetMaxTrialsPerCondition:
    def test_get_max_trials_epochs(self, subjects_mne_objects, electrodes_per_subject_roi):
        """Test finding maximum trials per condition for Epochs."""
        condition_names = ['condition1', 'condition2']
        subjects = ['sub1', 'sub2']
        roi = 'roi1'
        
        max_trials, max_subjects = get_max_trials_per_condition(
            subjects_mne_objects, condition_names, subjects,
            electrodes_per_subject_roi, roi, obs_axs=0
        )
        print(max_trials)
        print(max_subjects)
        assert isinstance(max_trials, dict)
        assert all(cond in max_trials for cond in condition_names)
        assert all(isinstance(val, int) for val in max_trials.values())
        assert all(isinstance(max_subjects[cond], list) for cond in condition_names)

# Test LabeledArray creation
class TestLabeledArrayCreation:
    @patch('src.analysis.utils.labeled_array_utils.LabeledArray')
    def test_create_subject_labeled_array_from_dict(self, mock_labeled_array):
        """Test creating LabeledArray from dictionary."""
        # Setup mock
        mock_instance = MagicMock() # Use MagicMock for more flexibility
        mock_labeled_array.from_dict.return_value = mock_instance
    
        # FIX: Create a list of mocks for the labels attribute.
        # The length should match the number of dimensions of the expected LabeledArray (e.g., 4 for Cond, Trial, Chan, Time).
        mock_instance.labels = [MagicMock() for _ in range(4)]
    
        # Test data
        subject_dict = {
            'cond1': np.random.randn(10, 5, 100),
            'cond2': np.random.randn(10, 5, 100)
        }
        channel_names = ['CH1', 'CH2', 'CH3', 'CH4', 'CH5']
        times = np.array([str(t) for t in np.linspace(-0.5, 0.5, 100)])
    
        # Call the function under test
        result = create_subject_labeled_array_from_dict(
            subject_dict, channel_names, times, None, 
            chans_axs=1, time_axs=2, freq_axs=None
        )
    
        # Assertions
        mock_labeled_array.from_dict.assert_called_once_with(subject_dict)
    
        # Assert that the correct labels were assigned to the correct axes
        # The axes are shifted by +1 because LabeledArray.from_dict adds a 'conditions' axis at the beginning.
        # Assert that the 'values' attribute of the correct label object was set correctly.
        # The axes are shifted by +1 because LabeledArray.from_dict adds a 'conditions' axis at the beginning.
        assert mock_instance.labels[1 + 1].values == channel_names
        
        # Use np.array_equal for robust comparison of numpy arrays
        assert np.array_equal(mock_instance.labels[2 + 1].values, times)
    
        assert result == mock_instance

# Test NaN removal
class TestNaNRemoval:
    @patch('src.analysis.utils.labeled_array_utils.LabeledArray')
    def test_remove_nans_from_labeled_array(self, mock_labeled_array):
        """Test removing NaN trials from LabeledArray."""
        # Create mock labeled array with some NaN values
        mock_instance = MagicMock()
        mock_instance.labels = [['cond1', 'cond2']]
        
        # Create data with some NaN trials
        data_cond1 = np.random.randn(10, 5, 100)
        data_cond1[2, :, :] = np.nan  # Make trial 2 all NaN
        data_cond2 = np.random.randn(10, 5, 100)
        
        mock_instance.__getitem__.side_effect = lambda x: data_cond1 if x == 'cond1' else data_cond2
        mock_instance.keys.return_value = ['cond1', 'cond2']
        
        mock_labeled_array.from_dict.return_value = MagicMock()
        
        result, no_valid = remove_nans_from_labeled_array(
            mock_instance, obs_axs=0, chans_axs=1, time_axs=2
        )
        
        assert mock_labeled_array.from_dict.called
        assert len(no_valid) == 0  # Both conditions should have valid trials

# Test concatenation functions
class TestConcatenation:
    def test_concatenate_subject_labeled_arrays(self):
        """Test concatenating subject labeled arrays."""
        # Create mock labeled arrays
        roi_array = Mock()
        subject_array = Mock()
        
        # Test first subject (roi_array is None)
        result = concatenate_subject_labeled_arrays(None, subject_array, concatenation_axis=1)
        assert result == subject_array
        
        # Test subsequent subjects
        roi_array.concatenate.return_value = Mock()
        result = concatenate_subject_labeled_arrays(roi_array, subject_array, concatenation_axis=1)
        roi_array.concatenate.assert_called_once_with(subject_array, axis=2)  # axis adjusted by +1

# Test time range extraction
class TestTimeRangeExtraction:
    def test_get_data_in_time_range(self):
        """Test extracting data within a time range."""
        # Create mock labeled array
        mock_array = Mock()
        mock_array.labels = [None, None, None, np.array(['-0.5', '-0.25', '0.0', '0.25', '0.5'])]
        mock_array.take.return_value = Mock()
        
        time_range = (-0.3, 0.3)
        result = get_data_in_time_range(mock_array, time_range, time_axs=-1)
        
        # Check that take was called with correct indices
        mock_array.take.assert_called_once()
        indices = mock_array.take.call_args[0][0]
        assert len(indices) == 3  # Should include -0.25, 0.0, 0.25

# Integration tests
class TestIntegration:
    @pytest.mark.integration
    def test_full_pipeline(self, subjects_mne_objects, electrodes_per_subject_roi):
        """Test the full pipeline from MNE objects to labeled arrays."""
        condition_names = ['condition1', 'condition2']
        rois = ['roi1']
        subjects = ['sub1', 'sub2']
        
        # This would need mocking of LabeledArray to work properly
        with patch('src.analysis.utils.labeled_array_utils.LabeledArray'):
            result = put_data_in_labeled_array_per_roi_subject(
                subjects_mne_objects, condition_names, rois, subjects,
                electrodes_per_subject_roi, random_state=42
            )
            
            assert isinstance(result, dict)
            assert 'roi1' in result

# Parametrized tests for edge cases
class TestEdgeCases:
    @pytest.mark.parametrize("n_trials,n_channels,n_times", [
        (1, 1, 10),    # Minimal data
        (100, 50, 1000),  # Large data
        (5, 3, 50),    # Small typical data
    ])
    def test_various_data_sizes(self, n_trials, n_channels, n_times):
        """Test functions with various data sizes."""
        # Create epochs with specified dimensions
        ch_names = [f'CH{i}' for i in range(n_channels)]
        info = mne.create_info(ch_names=ch_names, sfreq=1000, ch_types='eeg')
        data = np.random.randn(n_trials, n_channels, n_times)
        epochs = mne.EpochsArray(data, info)
        
        assert epochs.get_data().shape == (n_trials, n_channels, n_times)

# Performance tests
class TestPerformance:
    @pytest.mark.slow
    def test_large_dataset_performance(self):
        """Test performance with large datasets."""
        import time
        
        # Create large dataset
        n_subjects = 10
        n_conditions = 5
        n_trials = 100
        n_channels = 64
        n_times = 1000
        
        start_time = time.time()
        
        # Create mock data structure
        subjects_data = {}
        for i in range(n_subjects):
            subjects_data[f'sub{i}'] = {}
            for j in range(n_conditions):
                ch_names = [f'CH{k}' for k in range(n_channels)]
                info = mne.create_info(ch_names=ch_names, sfreq=1000, ch_types='eeg')
                data = np.random.randn(n_trials, n_channels, n_times)
                epochs = mne.EpochsArray(data, info)
                subjects_data[f'sub{i}'][f'cond{j}'] = {'HG_ev1_power_rescaled': epochs}
        
        # Test detect_data_type performance
        data_type = detect_data_type(subjects_data)
        
        elapsed = time.time() - start_time
        assert elapsed < 1.0  # Should complete within 1 second
        assert data_type == 'Epochs'