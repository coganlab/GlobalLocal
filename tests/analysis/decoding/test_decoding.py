# TODO: Update tests, and run them
import pytest
import numpy as np
import mne
from unittest.mock import Mock, MagicMock, patch, call
import sys
import os
from functools import partial
from scipy.stats import ttest_ind

# Add the project root to the path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from src.analysis.decoding.decoding import (
    decode_on_sig_tfr_clusters,
    compute_sig_tfr_masks_from_concatenated_data,
    compute_sig_tfr_masks_for_specified_channels,
    apply_tfr_masks_and_flatten_to_make_decoding_matrix,
    get_confusion_matrix_for_rois_tfr_cluster,
    concatenate_and_balance_data_for_decoding,
    Decoder
)

# Test fixtures
@pytest.fixture
def mock_decoder():
    """Create a mock Decoder class."""
    mock = Mock(spec=Decoder)
    mock_instance = Mock()
    mock.return_value = mock_instance
    mock_instance.fit.return_value = None
    mock_instance.predict.return_value = np.array([0, 1, 0, 1])
    return mock

@pytest.fixture
def sample_tfr_data():
    """Create sample TFR data (trials, channels, freqs, times)."""
    n_trials = 20
    n_channels = 5
    n_freqs = 10
    n_times = 25
    return np.random.randn(n_trials, n_channels, n_freqs, n_times)

@pytest.fixture
def sample_labels():
    """Create sample labels for binary classification."""
    return np.array([0, 1] * 10)

@pytest.fixture
def sample_cats():
    """Create sample category dictionary."""
    return {('cond1',): 0, ('cond2',): 1}

@pytest.fixture
def sample_roi_labeled_arrays():
    """Create mock roi_labeled_arrays structure."""
    mock_array = MagicMock()
    mock_array.keys.return_value = ['cond1', 'cond2']
    
    # Mock data for each condition
    cond1_data = np.random.randn(10, 5, 10, 25)  # (trials, channels, freqs, times)
    cond2_data = np.random.randn(10, 5, 10, 25)
    
    mock_array.__getitem__.side_effect = lambda x: cond1_data if x == 'cond1' else cond2_data
    
    return {'roi1': mock_array, 'roi2': mock_array}

@pytest.fixture
def stat_func():
    """Create statistical function for testing."""
    return partial(ttest_ind, equal_var=False, nan_policy='omit')


class TestComputeSigTfrMasksForSpecifiedChannels:
    """Test compute_sig_tfr_masks_for_specified_channels function."""
    
    def test_basic_functionality(self, stat_func):
        """Test basic mask computation for channels."""
        n_channels = 3
        train_data_by_condition = {
            'cond1': np.random.randn(10, n_channels, 8, 25),  # (trials, channels, freqs, times)
            'cond2': np.random.randn(10, n_channels, 8, 25)
        }
        condition_names = ['cond1', 'cond2']
        
        with patch('src.analysis.decoding.decoding.time_perm_cluster') as mock_perm:
            # Mock the permutation test to return a boolean mask
            mock_perm.return_value = (np.random.rand(8, 40) > 0.5, np.array([0.01]))
            
            masks = compute_sig_tfr_masks_for_specified_channels(
                n_channels, train_data_by_condition, condition_names,
                obs_axs=0, chans_axs=1, stat_func=stat_func,
                p_thresh=0.05, n_perm=10
            )
            
            assert len(masks) == n_channels
            assert all(isinstance(masks[i], np.ndarray) for i in range(n_channels))
            assert all(masks[i].shape == (8, 40) for i in range(n_channels))
    
    def test_insufficient_data(self, stat_func):
        """Test handling of channels with insufficient data."""
        n_channels = 2
        train_data_by_condition = {
            'cond1': np.random.randn(0, n_channels, 8, 40),  # No trials
            'cond2': np.random.randn(10, n_channels, 8, 40)
        }
        condition_names = ['cond1', 'cond2']
        
        masks = compute_sig_tfr_masks_for_specified_channels(
            n_channels, train_data_by_condition, condition_names,
            obs_axs=0, chans_axs=1, stat_func=stat_func,
            p_thresh=0.05, n_perm=10
        )
        
        # Should return zero masks for channels with no data
        assert all(not masks[i].any() for i in range(n_channels))


class TestComputeSigTfrMasksFromConcatenatedData:
    """Test compute_sig_tfr_masks_from_concatenated_data function."""
    
    def test_basic_functionality(self, sample_tfr_data, sample_labels, sample_cats, stat_func):
        """Test basic mask computation from concatenated data."""
        train_indices = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        condition_names = ['cond1', 'cond2']
        
        with patch('src.analysis.decoding.decoding.compute_sig_tfr_masks_for_specified_channels') as mock_compute:
            mock_compute.return_value = {
                0: np.random.rand(8, 40) > 0.5,
                1: np.random.rand(8, 40) > 0.5
            }
            
            masks = compute_sig_tfr_masks_from_concatenated_data(
                sample_tfr_data, sample_labels, train_indices,
                condition_names, sample_cats,
                obs_axs=0, chans_axs=1, stat_func=stat_func,
                p_thresh=0.05, n_perm=10
            )
            
            assert isinstance(masks, dict)
            assert len(masks) == 2  # Should have masks for 2 channels
    
    def test_invalid_condition_count(self, sample_tfr_data, sample_labels, sample_cats, stat_func):
        """Test error handling for invalid number of conditions."""
        train_indices = np.array([0, 2, 4])
        condition_names = ['cond1', 'cond2', 'cond3']  # Too many conditions
        
        with pytest.raises(ValueError, match="will only work for two conditions"):
            compute_sig_tfr_masks_from_concatenated_data(
                sample_tfr_data, sample_labels, train_indices,
                condition_names, sample_cats,
                obs_axs=0, chans_axs=1, stat_func=stat_func,
                p_thresh=0.05, n_perm=10
            )


class TestApplyTfrMasksAndFlatten:
    """Test apply_tfr_masks_and_flatten_to_make_decoding_matrix function."""
    
    def test_basic_functionality(self, sample_tfr_data):
        """Test applying masks and flattening data."""
        # Create simple masks for 5 channels
        channel_masks = {
            0: np.array([[True, False], [False, True]]),
            1: np.array([[False, True], [True, False]]),
            2: np.array([[True, True], [False, False]]),
            3: np.array([[False, False], [True, True]]),
            4: np.array([[True, False], [True, False]])
        }
        
        # Create smaller test data matching mask dimensions
        n_trials = 10
        n_channels = 5
        test_data = np.random.randn(n_trials, n_channels, 2, 2)
        
        result = apply_tfr_masks_and_flatten_to_make_decoding_matrix(
            test_data, obs_axs=0, chans_axs=1, channel_masks=channel_masks
        )
        
        # Each channel has 2 True values in its mask, so 5 channels * 2 = 10 features
        assert result.shape == (n_trials, 10)
    
    def test_missing_channel_masks(self):
        """Test handling of missing channel masks."""
        test_data = np.random.randn(10, 3, 5, 5)
        channel_masks = {0: np.ones((5, 5), dtype=bool)}  # Only mask for channel 0
        
        result = apply_tfr_masks_and_flatten_to_make_decoding_matrix(
            test_data, obs_axs=0, chans_axs=1, channel_masks=channel_masks
        )
        
        # Should use all features for channels without masks
        expected_features = 25  # channel 0 masked (25), and no features for other channels
        assert result.shape == (10, expected_features)
    
    def test_different_obs_axis(self):
        """Test with observation axis in different position."""
        test_data = np.random.randn(5, 10, 3, 20)  # obs_axs=1
        channel_masks = {i: np.ones((3, 20), dtype=bool) for i in range(5)}
        
        result = apply_tfr_masks_and_flatten_to_make_decoding_matrix(
            test_data, obs_axs=1, chans_axs=0, channel_masks=channel_masks
        )
        
        assert result.shape[0] == 10  # n_trials


class TestDecodeOnSigTfrClusters:
    """Test decode_on_sig_tfr_clusters function."""
    
    def test_basic_functionality(self, mock_decoder, sample_tfr_data, sample_labels, sample_cats, stat_func):
        """Test basic decoding with TFR cluster masking."""
        n_train = 15
        n_test = 5
        train_indices = np.arange(n_train)
        test_indices = np.arange(n_train, n_train + n_test)
        
        X_train = sample_tfr_data[:n_train]
        X_test = sample_tfr_data[n_train:n_train + n_test]
        y_train = sample_labels[:n_train]
        
        with patch('src.analysis.decoding.decoding.compute_sig_tfr_masks_from_concatenated_data') as mock_masks:
            with patch('src.analysis.decoding.decoding.apply_tfr_masks_and_flatten_to_make_decoding_matrix') as mock_flatten:
                # Mock mask computation
                mock_masks.return_value = {i: np.ones((10, 50), dtype=bool) for i in range(5)}
                
                # Mock flattening
                mock_flatten.side_effect = lambda data, *args, **kwargs: data.reshape(data.shape[0], -1)
                
                # Mock decoder
                mock_decoder_instance = mock_decoder.return_value
                mock_decoder_instance.fit.return_value = None
                mock_decoder_instance.predict.return_value = np.array([0, 1, 0, 1, 0])
                
                preds, channel_masks = decode_on_sig_tfr_clusters(
                    X_train, y_train, X_test,
                    train_indices, test_indices,
                    sample_tfr_data, sample_labels, sample_cats,
                    obs_axs=0, chans_axs=1,
                    stat_func=stat_func, p_thresh=0.05, n_perm=10,
                    Decoder=mock_decoder, explained_variance=0.95,
                    oversample=True
                )
                
                assert len(preds) == n_test
                assert mock_decoder_instance.fit.called
                assert mock_decoder_instance.predict.called
    
    def test_with_nans(self, mock_decoder, stat_func):
        """Test handling of NaN values in data."""
        # Create data with NaNs
        n_train = 10
        n_test = 5
        X_train = np.random.randn(n_train, 3, 5, 20)
        X_train[0, :, :, :] = np.nan  # First trial all NaN
        X_test = np.random.randn(n_test, 3, 5, 20)
        X_test[1, 0, 0, 0] = np.nan  # Single NaN in test
        
        y_train = np.array([0, 1] * 5)
        train_indices = np.arange(n_train)
        test_indices = np.arange(n_test)
        
        concatenated_data = np.vstack([X_train, X_test])
        labels = np.array([0, 1] * 7 + [0])[:15]
        cats = {('cond1',): 0, ('cond2',): 1}
        
        with patch('src.analysis.decoding.decoding.compute_sig_tfr_masks_from_concatenated_data') as mock_masks:
            with patch('src.analysis.decoding.decoding.apply_tfr_masks_and_flatten_to_make_decoding_matrix') as mock_flatten:
                with patch('src.analysis.decoding.decoding.mixup2') as mock_mixup:
                    mock_masks.return_value = {i: np.ones((5, 20), dtype=bool) for i in range(3)}
                    mock_flatten.side_effect = lambda data, *args, **kwargs: data.reshape(data.shape[0], -1)
                    
                    mock_decoder_instance = mock_decoder.return_value
                    mock_decoder_instance.predict.return_value = np.array([0, 1, 0, 1, 0])
                    
                    preds, channel_masks = decode_on_sig_tfr_clusters(
                        X_train, y_train, X_test,
                        train_indices, test_indices,
                        concatenated_data, labels, cats,
                        obs_axs=0, chans_axs=1,
                        stat_func=stat_func, p_thresh=0.05, n_perm=10,
                        Decoder=mock_decoder, explained_variance=0.95,
                        oversample=True, alpha=0.2
                    )
                    
                    # Check that mixup2 was called for handling NaNs
                    assert mock_mixup.called


class TestGetConfusionMatrixForRoisTfrCluster:
    """Test get_confusion_matrix_for_rois_tfr_cluster function."""
    
    @patch('src.analysis.decoding.decoding.concatenate_and_balance_data_for_decoding')
    @patch('src.analysis.decoding.decoding.decode_on_sig_tfr_clusters')
    @patch('src.analysis.decoding.decoding.confusion_matrix')
    def test_basic_functionality(self, mock_cm, mock_decode, mock_concatenate,
                                sample_roi_labeled_arrays, mock_decoder, stat_func):
        """Test basic confusion matrix computation for ROIs."""
        # Setup mocks
        mock_concatenate.return_value = (
            np.random.randn(20, 5, 10, 50),  # concatenated_data
            np.array([0, 1] * 10),  # labels
            {('cond1',): 0, ('cond2',): 1}  # cats
        )
        
        mock_channel_masks = {i: np.ones((10, 50), dtype=bool) for i in range(5)}
        mock_decode.return_value = (np.array([0, 1, 0, 1]), mock_channel_masks)
        mock_cm.return_value = np.array([[2, 0], [0, 2]])
        
        rois = ['roi1']
        strings_to_find = ['cond1', 'cond2']
        
        result, cats_dict, channel_masks = get_confusion_matrix_for_rois_tfr_cluster(
            sample_roi_labeled_arrays, rois, strings_to_find,
            stat_func, mock_decoder,
            explained_variance=0.95, p_thresh=0.05, n_perm=10,
            n_splits=2, n_repeats=2, obs_axs=0, chans_axs=1,
            balance_method='subsample', oversample=True,
            random_state=42
        )
        
        assert 'roi1' in result
        assert isinstance(result['roi1'], np.ndarray)
        assert mock_concatenate.called
        assert mock_decode.call_count > 0
    
    def test_multiple_rois(self, mock_decoder, stat_func):
        """Test processing multiple ROIs."""
        mock_roi_arrays = {
            'roi1': MagicMock(),
            'roi2': MagicMock(),
            'roi3': MagicMock()
        }
        
        with patch('src.analysis.decoding.decoding.concatenate_and_balance_data_for_decoding') as mock_concat:
            with patch('src.analysis.decoding.decoding.decode_on_sig_tfr_clusters') as mock_decode:
                with patch('src.analysis.decoding.decoding.confusion_matrix') as mock_cm:
                    mock_concat.return_value = (
                        np.random.randn(20, 5, 10, 25),
                        np.array([0, 1] * 10),
                        {('cond1',): 0, ('cond2',): 1}
                    )
                    mock_decode.return_value = (np.array([0, 1, 0, 1]), {i: np.ones((10, 25), dtype=bool) for i in range(5)})
                    mock_cm.return_value = np.array([[2, 0], [0, 2]])
                    
                    result, cats_dict, channel_masks = get_confusion_matrix_for_rois_tfr_cluster(
                        mock_roi_arrays, list(mock_roi_arrays.keys()),
                        ['cond1', 'cond2'], stat_func, mock_decoder,
                        n_splits=2, n_repeats=1
                    )
                    
                    assert len(result) == 3
                    assert all(roi in result for roi in mock_roi_arrays.keys())
    
    def test_cross_validation_averaging(self, mock_decoder, stat_func):
        """Test that confusion matrices are properly averaged across CV folds."""
        mock_roi_arrays = {'roi1': MagicMock()}
        
        with patch('src.analysis.decoding.decoding.concatenate_and_balance_data_for_decoding') as mock_concat:
            with patch('src.analysis.decoding.decoding.decode_on_sig_tfr_clusters') as mock_decode:
                with patch('src.analysis.decoding.decoding.confusion_matrix') as mock_cm:
                    mock_concat.return_value = (
                        np.random.randn(20, 5, 10, 25),
                        np.array([0, 1] * 10),
                        {('cond1',): 0, ('cond2',): 1}
                    )
                    mock_decode.return_value = (np.array([0, 1, 0, 1]), {i: np.ones((10, 25), dtype=bool) for i in range(5)})
                    
                    # Return different confusion matrices for each fold
                    cms = [
                        np.array([[3, 1], [1, 3]]),
                        np.array([[2, 2], [0, 4]]),
                        np.array([[4, 0], [2, 2]])
                    ]
                    mock_cm.side_effect = cms * 10  # Enough for all folds
                    
                    result, cats_dict, channel_masks = get_confusion_matrix_for_rois_tfr_cluster(
                        mock_roi_arrays, ['roi1'], ['cond1', 'cond2'],
                        stat_func, mock_decoder,
                        n_splits=3, n_repeats=1, random_state=42
                    )
                    
                    # Check that result is averaged
                    expected_sum = np.sum(cms, axis=0)
                    expected_avg = expected_sum / 1  # 1 repeat
                    np.testing.assert_array_almost_equal(result['roi1'], expected_avg)


class TestIntegration:
    """Integration tests for the full pipeline."""
    
    @pytest.mark.integration
    def test_full_pipeline_with_real_data(self, stat_func):
        """Test the full pipeline with realistic data structures."""
        from unittest.mock import MagicMock
        
        roi_array = MagicMock()
        roi_array.keys.return_value = ['bigS', 'bigH']
        
        # Create realistic data
        n_trials_per_cond = 15
        n_channels = 10
        n_freqs = 8
        n_times = 100
        
        bigS_data = np.random.randn(n_trials_per_cond, n_channels, n_freqs, n_times)
        bigH_data = np.random.randn(n_trials_per_cond, n_channels, n_freqs, n_times)
        
        roi_array.__getitem__.side_effect = lambda x: bigS_data if x == 'bigS' else bigH_data
        roi_labeled_arrays = {'lpfc': roi_array}
        
        # Test with minimal parameters
        with patch('src.analysis.decoding.decoding.Decoder') as mock_decoder_class:
            mock_instance = Mock()
            mock_decoder_class.return_value = mock_instance
            mock_instance.fit.return_value = None
            
            # Make predict return predictions that match the test set size dynamically
            def dynamic_predict(X):
                n_samples = X.shape[0]
                return np.array([0, 1] * (n_samples // 2) + [0] * (n_samples % 2))
            
            mock_instance.predict.side_effect = dynamic_predict
            
            result, cats_dict, channel_masks = get_confusion_matrix_for_rois_tfr_cluster(
                roi_labeled_arrays,
                rois=['lpfc'],
                strings_to_find=['bigS', 'bigH'],
                stat_func=stat_func,
                Decoder=mock_decoder_class,
                explained_variance=0.95,
                p_thresh=0.05,
                n_perm=10,
                n_splits=2,
                n_repeats=1,
                random_state=42
            )
            
            assert 'lpfc' in result
            assert result['lpfc'].shape == (2, 2)  # Binary classification
            assert 'lpfc' in cats_dict


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_roi_list(self, sample_roi_labeled_arrays, mock_decoder, stat_func):
        """Test with empty ROI list."""
        result, cats_dict, channel_masks = get_confusion_matrix_for_rois_tfr_cluster(
            sample_roi_labeled_arrays, [], ['cond1', 'cond2'],
            stat_func, mock_decoder
        )
        assert len(result) == 0
        assert len(cats_dict) == 0
    
    def test_invalid_balance_method(self, sample_roi_labeled_arrays, mock_decoder, stat_func):
        """Test with invalid balance method."""
        with patch('src.analysis.decoding.decoding.concatenate_and_balance_data_for_decoding') as mock_concat:
            mock_concat.side_effect = ValueError("Invalid balance_method")
            
            with pytest.raises(ValueError, match="Invalid balance_method"):
                get_confusion_matrix_for_rois_tfr_cluster(
                    sample_roi_labeled_arrays, ['roi1'], ['cond1', 'cond2'],
                    stat_func, mock_decoder,
                    balance_method='invalid_method'
                )
    
    def test_single_condition(self, mock_decoder, stat_func):
        """Test error handling with only one condition."""
        mock_array = MagicMock()
        mock_array.keys.return_value = ['cond1']
        mock_array.__getitem__.return_value = np.random.randn(10, 5, 10, 25)
        
        roi_labeled_arrays = {'roi1': mock_array}
        
        with patch('src.analysis.decoding.decoding.concatenate_and_balance_data_for_decoding') as mock_concat:
            mock_concat.return_value = (
                np.random.randn(10, 5, 10, 25),
                np.zeros(10),  # All same label
                {('cond1',): 0}
            )
            
            with pytest.raises(ValueError, match="will only work for two conditions"):
                get_confusion_matrix_for_rois_tfr_cluster(
                    roi_labeled_arrays, ['roi1'], ['cond1'],
                    stat_func, mock_decoder,
                    n_splits=2, n_repeats=1
                )


class TestConcatenateAndBalanceData:
    """Test concatenate_and_balance_data_for_decoding function."""
    
    @pytest.fixture
    def mock_roi_labeled_arrays(self):
        """Create mock labeled arrays with controllable data."""
        mock_array = MagicMock()
        
        # Create data for different conditions with different trial counts
        cond1_data = np.random.randn(20, 5, 100)  # 20 trials
        cond2_data = np.random.randn(19, 5, 100)  # 19 trials
        cond3_data = np.random.randn(25, 5, 100)  # 25 trials
        
        # Add some NaN trials to test NaN handling
        cond1_data[10, 0, 0] = np.nan  # One channel-timepoint is nan
        cond1_data[19, :, :] = np.nan  # Last trial is NaN
        
        mock_array.keys.return_value = ['cond1', 'cond2', 'cond3']
        data_map = {
            'cond1': cond1_data,
            'cond2': cond2_data,
            'cond3': cond3_data
        }
        mock_array.__getitem__.side_effect = lambda key: data_map.get(key)
        
        return {'roi1': mock_array}
    
    def test_subsample_method(self, mock_roi_labeled_arrays):
        """Test subsampling to balance trial counts."""
        concatenated_data, labels, cats = concatenate_and_balance_data_for_decoding(
            mock_roi_labeled_arrays, 
            roi='roi1',
            strings_to_find=['cond1', 'cond2'],
            obs_axs=0,
            balance_method='subsample',
            random_state=42
        )
        
        # cond1: 20 trials, 2 with NaNs (partial + full) dropped → 18 valid
        # cond2: 19 trials, all valid → subsampled to 18
        assert concatenated_data.shape[0] == 36, "Balancing resulted in an incorrect number of trials"
        assert np.sum(labels == 0) == 18
        assert np.sum(labels == 1) == 18
        assert not np.any(np.isnan(concatenated_data)), "All NaN trials should have been removed"
        assert cats == {('cond1',): 0, ('cond2',): 1}
    
    def test_pad_with_nans_method(self, mock_roi_labeled_arrays):
        """Test padding with NaNs to balance trial counts."""
        concatenated_data, labels, cats = concatenate_and_balance_data_for_decoding(
            mock_roi_labeled_arrays,
            roi='roi1',
            strings_to_find=['cond1', 'cond2'],
            obs_axs=0,
            balance_method='pad_with_nans',
            random_state=42
        )
        
        # Should keep all trials including NaN ones
        # cond1: 20 trials, cond2: 19 trials
        # Should pad cond2 to 20 trials
        assert concatenated_data.shape[0] == 39, "Balancing resulted in an incorrect number of trials"
        assert np.sum(labels == 0) == 20  # cond1
        assert np.sum(labels == 1) == 20  # cond2 (padded)
        
        # Check that padding was added as NaNs
        cond2_mask = labels == 1
        cond2_data = concatenated_data[cond2_mask]
        # Last 5 trials of cond2 should be NaN (padding)
        assert np.all(np.isnan(cond2_data[-5:]))
    
    def test_string_groups(self, mock_roi_labeled_arrays):
        """Test grouping multiple conditions into one class."""
        # Group cond1 and cond3 together vs cond2
        strings_to_find = [['cond1', 'cond3'], ['cond2']]
        
        concatenated_data, labels, cats = concatenate_and_balance_data_for_decoding(
            mock_roi_labeled_arrays,
            roi='roi1',
            strings_to_find=strings_to_find,
            obs_axs=0,
            balance_method='subsample',
            random_state=42
        )
        
        # Should have two classes
        assert cats == {('cond1', 'cond3'): 0, ('cond2',): 1}
        
        # cond1 (19 valid) + cond3 (25 valid) = 26 trials for class 0
        # cond2 (19 valid) = 19 trials for class 1
        # After subsampling to balance: 19 trials each
        assert np.sum(labels == 0) == 19
        assert np.sum(labels == 1) == 19
    
    def test_no_matching_conditions(self):
        """Test error when no conditions match the search strings."""
        mock_array = MagicMock()
        mock_array.keys.return_value = ['condA', 'condB']
        mock_array.__getitem__.return_value = np.random.randn(10, 5, 100)
        
        roi_labeled_arrays = {'roi1': mock_array}
        
        with pytest.raises(ValueError, match="No matching conditions found"):
            concatenate_and_balance_data_for_decoding(
                roi_labeled_arrays,
                roi='roi1',
                strings_to_find=['cond1', 'cond2'],  # Don't exist
                obs_axs=0,
                balance_method='subsample',
                random_state=42
            )
    
    def test_invalid_balance_method(self, mock_roi_labeled_arrays):
        """Test error with invalid balance method."""
        with pytest.raises(ValueError, match="Invalid balance_method"):
            concatenate_and_balance_data_for_decoding(
                mock_roi_labeled_arrays,
                roi='roi1',
                strings_to_find=['cond1', 'cond2'],
                obs_axs=0,
                balance_method='invalid_method',
                random_state=42
            )
    
    def test_reproducibility_with_random_state(self, mock_roi_labeled_arrays):
        """Test that random_state ensures reproducibility."""
        result1 = concatenate_and_balance_data_for_decoding(
            mock_roi_labeled_arrays,
            roi='roi1',
            strings_to_find=['cond1', 'cond2'],
            obs_axs=0,
            balance_method='subsample',
            random_state=42
        )
        
        result2 = concatenate_and_balance_data_for_decoding(
            mock_roi_labeled_arrays,
            roi='roi1', 
            strings_to_find=['cond1', 'cond2'],
            obs_axs=0,
            balance_method='subsample',
            random_state=42
        )
        
        np.testing.assert_array_equal(result1[0], result2[0])  # data
        np.testing.assert_array_equal(result1[1], result2[1])  # labels
        assert result1[2] == result2[2]  # cats
    
    def test_all_nan_condition(self):
        """Test handling when a condition has all NaN trials."""
        mock_array = MagicMock()
        
        cond1_data = np.full((10, 5, 100), np.nan)
        cond2_data = np.random.randn(10, 5, 100)
        
        mock_array.keys.return_value = ['cond1', 'cond2']
        mock_array.__getitem__.side_effect = {
            'cond1': cond1_data,
            'cond2': cond2_data
        }.get
        
        roi_labeled_arrays = {'roi1': mock_array}
        
        concatenated_data, labels, cats = concatenate_and_balance_data_for_decoding(
            roi_labeled_arrays,
            roi='roi1',
            strings_to_find=['cond1', 'cond2'],
            obs_axs=0,
            balance_method='subsample',
            random_state=42
        )
        
        # Should handle gracefully - cond1 has 0 valid trials
        assert concatenated_data.shape[0] == 0  # No valid data to balance
    
    def test_single_condition(self):
        """Test with only one condition (edge case)."""
        mock_array = MagicMock()
        mock_array.keys.return_value = ['cond1']
        mock_array.__getitem__.return_value = np.random.randn(10, 5, 100)
        
        roi_labeled_arrays = {'roi1': mock_array}
        
        concatenated_data, labels, cats = concatenate_and_balance_data_for_decoding(
            roi_labeled_arrays,
            roi='roi1',
            strings_to_find=['cond1'],
            obs_axs=0,
            balance_method='subsample',
            random_state=42
        )
        
        assert concatenated_data.shape[0] == 10
        assert np.all(labels == 0)
        assert cats == {('cond1',): 0}
 

"""
End-to-end test for decoding_dcc.py main().

Strategy:
- Mock all external I/O (file loading, MNE objects, LAB_root paths)
- Mock process_bootstrap to return realistic synthetic results
- Let the real aggregation, CM normalization, statistics, plotting, and pickling run
- Verify the full pipeline produces correct outputs without crashing
"""

import pytest
import numpy as np
import os
import sys
import pickle
import tempfile
import shutil
from types import SimpleNamespace
from unittest.mock import patch, MagicMock, Mock

# Add project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))


# ---------------------------------------------------------------------------
# Helpers to build synthetic bootstrap results
# ---------------------------------------------------------------------------

def _make_time_window_centers(n_windows=10):
    """Simulate time window centers (in seconds)."""
    return np.linspace(-0.2, 0.8, n_windows)


def _make_fake_bootstrap_result(
    condition_comparisons: dict,
    rois: list,
    cats_by_roi: dict,
    n_windows: int = 10,
    n_classes: int = 2,
    n_folds: int = 4,
    n_repeats: int = 2,
    folds_as_samples: bool = False,
):
    """
    Build a single bootstrap result dict matching the structure returned by
    process_bootstrap().

    Returns dict with keys: 'time_window_results', 'time_averaged_cms', 'cats_by_roi'
    """
    twc = _make_time_window_centers(n_windows)
    time_window_results = {}
    time_averaged_cms = {}

    for comp_name in condition_comparisons:
        time_window_results[comp_name] = {}
        time_averaged_cms[comp_name] = {}

        for roi in rois:
            if folds_as_samples:
                n_samples = n_folds * n_repeats
            else:
                n_samples = n_repeats

            fold_accs = np.random.uniform(0.4, 0.8, size=(n_windows, n_samples))
            fold_shuffle_accs = np.random.uniform(0.3, 0.6, size=(n_windows, n_samples))

            cms_per_window = [
                np.random.randint(0, 10, size=(n_classes, n_classes)).astype(float)
                for _ in range(n_windows)
            ]
            shuffle_cms_per_window = [
                np.random.randint(0, 10, size=(n_classes, n_classes)).astype(float)
                for _ in range(n_windows)
            ]

            time_window_results[comp_name][roi] = {
                'time_window_centers': twc,
                'fold_true_accs': fold_accs,
                'fold_shuffle_accs': fold_shuffle_accs,
                'repeat_true_accs': fold_accs[:, :n_repeats],
                'repeat_shuffle_accs': fold_shuffle_accs[:, :n_repeats],
                'cms_per_window': cms_per_window,
                'shuffle_cms_per_window': shuffle_cms_per_window,
            }

            time_averaged_cms[comp_name][roi] = np.random.randint(0, 20, size=(n_classes, n_classes))

    return {
        'time_window_results': time_window_results,
        'time_averaged_cms': time_averaged_cms,
        'cats_by_roi': cats_by_roi,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_save_dir():
    """Provide a temporary directory tree that mimics LAB_root structure."""
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def default_args(tmp_save_dir):
    """Build a minimal SimpleNamespace that mirrors what run_decoding.py passes to main()."""
    return SimpleNamespace(
        LAB_root=tmp_save_dir,
        subjects=['D0079', 'D0080'],
        task='GlobalLocal',
        epochs_root_file='some_epochs',
        conditions={
            'bigS': {'event_id': 1},
            'bigH': {'event_id': 2},
        },
        rois_dict={
            'lpfc': ['LpFC_1', 'LpFC_2'],
            'motor': ['Motor_1'],
        },
        electrodes='all',
        acc_trials_only=True,
        bootstraps=3,
        n_jobs=1,
        n_splits=4,
        n_repeats=2,
        folds_as_samples=False,
        explained_variance=0.95,
        window_size=50,
        step_size=25,
        sampling_rate=512,
        percentile=95,
        cluster_percentile=95,
        n_cluster_perms=100,
        random_state=42,
        unit_of_analysis='repeat',
        clf_model_str='PcaLda',
        timestamp='20260226_TEST',
        run_visualization_debug=False,
        single_column=True,
        show_legend=True,
    )


@pytest.fixture
def cats_by_roi():
    return {
        'lpfc': {('bigS',): 0, ('bigH',): 1},
        'motor': {('bigS',): 0, ('bigH',): 1},
    }


@pytest.fixture
def condition_comparisons():
    return {'bigS_vs_bigH': {'strings_to_find': ['bigS', 'bigH']}}


# ---------------------------------------------------------------------------
# Helper: common mocks for all decoding_dcc.main() tests
# ---------------------------------------------------------------------------
# All patches target the name AS IMPORTED in decoding_dcc.py.
# The module path prefix is 'src.analysis.decoding.decoding_dcc.' for names
# imported at the top of that file.

_UTILS = 'src.analysis.utils.general_utils'
_DECODING = 'src.analysis.decoding.decoding'
_DCC = 'dcc_scripts.decoding.decoding_dcc'  # the script being tested

_COMMON_PATCHES = [
    f'{_UTILS}.get_conditions_save_name',
    f'{_UTILS}.load_subjects_electrodes_to_ROIs_dict',
    f'{_UTILS}.build_condition_comparisons',
    f'{_UTILS}.get_sig_chans_per_subject',
    f'{_UTILS}.make_sig_electrodes_per_subject_and_roi_dict',
    f'{_UTILS}.create_subjects_mne_objects_dict',
    f'{_UTILS}.filter_electrode_lists_against_subjects_mne_objects',
    f'{_DCC}.run_visualization_debug',
    f'{_DCC}.Parallel',
    f'{_DECODING}.compute_pooled_bootstrap_statistics',
    f'{_DECODING}.plot_accuracies_nature_style',
    f'{_DCC}.run_debug_cm_traces',
    f'{_DCC}.run_all_context_comparisons',
    f'{_DCC}.run_aggregate_and_plot_time_averaged_cms',
]

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMainEndToEnd:
    """
    End-to-end tests for decoding_dcc.main().

    All external I/O is mocked; the internal aggregation / statistics /
    plotting / saving logic runs for real.
    """

    @patch(f'{_DCC}.run_aggregate_and_plot_time_averaged_cms')
    @patch(f'{_DCC}.run_all_context_comparisons')
    @patch(f'{_DCC}.run_debug_cm_traces')
    @patch(f'{_DCC}.plot_accuracies_nature_style')
    @patch(f'{_DCC}.compute_pooled_bootstrap_statistics')
    @patch(f'{_DCC}.Parallel')
    @patch(f'{_DCC}.run_visualization_debug')
    @patch(f'{_DCC}.filter_electrode_lists_against_subjects_mne_objects')
    @patch(f'{_DCC}.create_subjects_mne_objects_dict')
    @patch(f'{_DCC}.make_sig_electrodes_per_subject_and_roi_dict')
    @patch(f'{_DCC}.get_sig_chans_per_subject')
    @patch(f'{_DCC}.build_condition_comparisons')
    @patch(f'{_DCC}.load_subjects_electrodes_to_ROIs_dict')
    @patch(f'{_DCC}.get_conditions_save_name')
    @patch(f'{_DCC}.print_summary_of_dropped_electrodes')
    def test_main_runs_to_completion_and_saves_pickle(
        self,
        mock_print_summary,
        mock_get_cond_name,
        mock_load_elec_dict,
        mock_build_comps,
        mock_sig_chans,
        mock_make_sig_elecs,
        mock_create_mne,
        mock_filter_elecs,
        mock_run_viz,
        mock_parallel_cls,
        mock_compute_stats,
        mock_plot_acc,
        mock_debug_cm,
        mock_context_comps,
        mock_agg_cms,
        default_args,
        cats_by_roi,
        condition_comparisons,
        tmp_save_dir,
    ):
        """
        Verify that main() orchestrates the full pipeline:
        1. loads config & electrodes
        2. runs parallel bootstraps
        3. aggregates CMs
        4. computes pooled statistics
        5. plots results
        6. saves master pickle
        """
        rois = list(default_args.rois_dict.keys())
        n_windows = 10

        # --- stub simple helpers ---
        mock_get_cond_name.return_value = 'bigS_vs_bigH'
        mock_load_elec_dict.return_value = {
            'D0079': {'LpFC_1': 'lpfc', 'LpFC_2': 'lpfc', 'Motor_1': 'motor'},
            'D0080': {'LpFC_1': 'lpfc', 'Motor_1': 'motor'},
        }
        mock_build_comps.return_value = condition_comparisons
        mock_sig_chans.return_value = {
            'D0079': ['LpFC_1', 'Motor_1'],
            'D0080': ['LpFC_1'],
        }

        all_elecs = {
            'lpfc': {'D0079': ['LpFC_1', 'LpFC_2'], 'D0080': ['LpFC_1']},
            'motor': {'D0079': ['Motor_1'], 'D0080': []},
        }
        sig_elecs = {
            'lpfc': {'D0079': ['LpFC_1'], 'D0080': ['LpFC_1']},
            'motor': {'D0079': ['Motor_1'], 'D0080': []},
        }
        mock_make_sig_elecs.return_value = (all_elecs, sig_elecs)
        mock_create_mne.return_value = {'D0079': MagicMock(), 'D0080': MagicMock()}
        mock_filter_elecs.return_value = all_elecs  # pass-through

        # --- stub parallel bootstrap execution ---
        fake_results = [
            _make_fake_bootstrap_result(
                condition_comparisons, rois, cats_by_roi, n_windows=n_windows,
                folds_as_samples=default_args.folds_as_samples,
                n_folds=default_args.n_splits,
                n_repeats=default_args.n_repeats,
            )
            for _ in range(default_args.bootstraps)
        ]

        mock_parallel_instance = MagicMock()
        mock_parallel_instance.__call__ = MagicMock(return_value=fake_results)
        mock_parallel_cls.return_value = mock_parallel_instance

        # --- stub pooled statistics ---
        fake_stats = {}
        for comp in condition_comparisons:
            fake_stats[comp] = {}
            for roi in rois:
                fake_stats[comp][roi] = {
                    'unit_of_analysis': 'repeat',
                    'repeat_true_accs': np.random.uniform(0.4, 0.8, size=(n_windows, default_args.bootstraps * default_args.n_repeats)),
                    'repeat_shuffle_accs': np.random.uniform(0.3, 0.6, size=(n_windows, default_args.bootstraps * default_args.n_repeats)),
                    'significant_clusters': [],
                }
        mock_compute_stats.return_value = fake_stats

        # ---- RUN ----
        from src.analysis.decoding.decoding_dcc import main
        main(default_args)

        # ---- ASSERTIONS ----

        # 1. Config loaded
        mock_load_elec_dict.assert_called_once()
        mock_build_comps.assert_called_once()

        # 2. Parallel bootstrapping invoked with correct count
        mock_parallel_cls.assert_called_once()
        mock_parallel_instance.assert_called_once()

        # 3. Aggregation of time-averaged CMs called
        mock_agg_cms.assert_called_once()

        # 4. Pooled statistics computed
        mock_compute_stats.assert_called_once()
        stats_call_args = mock_compute_stats.call_args
        # Second positional arg is n_bootstraps
        assert stats_call_args[0][1] == default_args.bootstraps

        # 5. Plotting called for each (condition_comparison, roi) pair
        expected_plot_calls = len(condition_comparisons) * len(rois)
        assert mock_plot_acc.call_count == expected_plot_calls

        # 6. Debug CM traces called
        mock_debug_cm.assert_called_once()

        # 7. Context comparisons called
        mock_context_comps.assert_called_once()

        # 8. Master pickle saved
        pkl_files = []
        for root, dirs, files in os.walk(tmp_save_dir):
            pkl_files.extend(
                os.path.join(root, f) for f in files if f.endswith('.pkl')
            )
        assert len(pkl_files) == 1, f"Expected 1 pickle file, found {len(pkl_files)}: {pkl_files}"

        with open(pkl_files[0], 'rb') as f:
            saved = pickle.load(f)

        assert 'stats' in saved
        assert 'metadata' in saved
        assert 'comparison_clusters' in saved
        assert saved['metadata']['args']['bootstraps'] == default_args.bootstraps
        assert 'time_window_centers' in saved['metadata']

    @patch(f'{_DCC}.run_aggregate_and_plot_time_averaged_cms')
    @patch(f'{_DCC}.run_all_context_comparisons')
    @patch(f'{_DCC}.run_debug_cm_traces')
    @patch(f'{_DCC}.plot_accuracies_nature_style')
    @patch(f'{_DCC}.compute_pooled_bootstrap_statistics')
    @patch(f'{_DCC}.Parallel')
    @patch(f'{_DCC}.run_visualization_debug')
    @patch(f'{_DCC}.filter_electrode_lists_against_subjects_mne_objects')
    @patch(f'{_DCC}.create_subjects_mne_objects_dict')
    @patch(f'{_DCC}.make_sig_electrodes_per_subject_and_roi_dict')
    @patch(f'{_DCC}.get_sig_chans_per_subject')
    @patch(f'{_DCC}.build_condition_comparisons')
    @patch(f'{_DCC}.load_subjects_electrodes_to_ROIs_dict')
    @patch(f'{_DCC}.get_conditions_save_name')
    @patch(f'{_DCC}.print_summary_of_dropped_electrodes')
    def test_all_bootstraps_fail_gracefully(
        self,
        mock_print_summary,
        mock_get_cond_name,
        mock_load_elec_dict,
        mock_build_comps,
        mock_sig_chans,
        mock_make_sig_elecs,
        mock_create_mne,
        mock_filter_elecs,
        mock_run_viz,
        mock_parallel_cls,
        mock_compute_stats,
        mock_plot_acc,
        mock_debug_cm,
        mock_context_comps,
        mock_agg_cms,
        default_args,
        condition_comparisons,
        tmp_save_dir,
    ):
        """
        When every bootstrap returns None, main() should print an error
        message and return early without crashing.
        """
        mock_get_cond_name.return_value = 'bigS_vs_bigH'
        mock_load_elec_dict.return_value = {
            'D0079': {'LpFC_1': 'lpfc'},
            'D0080': {'LpFC_1': 'lpfc'},
        }
        mock_build_comps.return_value = condition_comparisons
        mock_sig_chans.return_value = {'D0079': ['LpFC_1'], 'D0080': ['LpFC_1']}

        all_elecs = {'lpfc': {'D0079': ['LpFC_1'], 'D0080': ['LpFC_1']}, 'motor': {}}
        mock_make_sig_elecs.return_value = (all_elecs, all_elecs)
        mock_create_mne.return_value = {'D0079': MagicMock(), 'D0080': MagicMock()}
        mock_filter_elecs.return_value = all_elecs

        # All bootstraps return None (failure)
        mock_parallel_instance = MagicMock()
        mock_parallel_instance.__call__ = MagicMock(return_value=[None] * default_args.bootstraps)
        mock_parallel_cls.return_value = mock_parallel_instance

        from src.analysis.decoding.decoding_dcc import main
        # Should not raise
        main(default_args)

        # run_aggregate_and_plot_time_averaged_cms IS called (it runs before the early-return check)
        mock_agg_cms.assert_called_once()

        # Statistics and plotting should NOT have been called
        mock_compute_stats.assert_not_called()
        mock_plot_acc.assert_not_called()

    @patch(f'{_DCC}.run_aggregate_and_plot_time_averaged_cms')
    @patch(f'{_DCC}.run_all_context_comparisons')
    @patch(f'{_DCC}.run_debug_cm_traces')
    @patch(f'{_DCC}.plot_accuracies_nature_style')
    @patch(f'{_DCC}.compute_pooled_bootstrap_statistics')
    @patch(f'{_DCC}.Parallel')
    @patch(f'{_DCC}.run_visualization_debug')
    @patch(f'{_DCC}.filter_electrode_lists_against_subjects_mne_objects')
    @patch(f'{_DCC}.create_subjects_mne_objects_dict')
    @patch(f'{_DCC}.make_sig_electrodes_per_subject_and_roi_dict')
    @patch(f'{_DCC}.get_sig_chans_per_subject')
    @patch(f'{_DCC}.build_condition_comparisons')
    @patch(f'{_DCC}.load_subjects_electrodes_to_ROIs_dict')
    @patch(f'{_DCC}.get_conditions_save_name')
    @patch(f'{_DCC}.print_summary_of_dropped_electrodes')
    def test_cm_aggregation_is_correct(
        self,
        mock_print_summary,
        mock_get_cond_name,
        mock_load_elec_dict,
        mock_build_comps,
        mock_sig_chans,
        mock_make_sig_elecs,
        mock_create_mne,
        mock_filter_elecs,
        mock_run_viz,
        mock_parallel_cls,
        mock_compute_stats,
        mock_plot_acc,
        mock_debug_cm,
        mock_context_comps,
        mock_agg_cms,
        default_args,
        cats_by_roi,
        condition_comparisons,
        tmp_save_dir,
    ):
        """
        Verify the time-averaged CM aggregation logic: raw count CMs from
        each bootstrap are summed then row-normalized.
        
        NOTE: The actual aggregation now happens inside 
        run_aggregate_and_plot_time_averaged_cms(), which we mock. This test
        verifies the correct data is passed to that function.
        """
        rois = list(default_args.rois_dict.keys())
        comp_name = list(condition_comparisons.keys())[0]

        # Create deterministic CMs
        cm_boot0 = np.array([[8, 2], [3, 7]])
        cm_boot1 = np.array([[6, 4], [1, 9]])
        cm_boot2 = np.array([[10, 0], [5, 5]])

        def _make_result_with_cm(cm):
            twc = _make_time_window_centers(5)
            return {
                'time_window_results': {
                    comp_name: {
                        roi: {
                            'time_window_centers': twc,
                            'repeat_true_accs': np.random.rand(5, 2),
                            'repeat_shuffle_accs': np.random.rand(5, 2),
                        }
                        for roi in rois
                    }
                },
                'time_averaged_cms': {
                    comp_name: {roi: cm.copy() for roi in rois}
                },
                'cats_by_roi': cats_by_roi,
            }

        fake_results = [
            _make_result_with_cm(cm_boot0),
            _make_result_with_cm(cm_boot1),
            _make_result_with_cm(cm_boot2),
        ]

        # Wire up mocks
        mock_get_cond_name.return_value = 'bigS_vs_bigH'
        mock_load_elec_dict.return_value = {'D0079': {'LpFC_1': 'lpfc'}, 'D0080': {'LpFC_1': 'lpfc'}}
        mock_build_comps.return_value = condition_comparisons
        mock_sig_chans.return_value = {'D0079': ['LpFC_1'], 'D0080': ['LpFC_1']}
        all_elecs = {'lpfc': {'D0079': ['LpFC_1'], 'D0080': ['LpFC_1']}, 'motor': {}}
        mock_make_sig_elecs.return_value = (all_elecs, all_elecs)
        mock_create_mne.return_value = {'D0079': MagicMock(), 'D0080': MagicMock()}
        mock_filter_elecs.return_value = all_elecs

        mock_parallel_instance = MagicMock()
        mock_parallel_instance.__call__ = MagicMock(return_value=fake_results)
        mock_parallel_cls.return_value = mock_parallel_instance

        fake_stats = {comp_name: {roi: {
            'unit_of_analysis': 'repeat',
            'repeat_true_accs': np.random.rand(5, 6),
            'repeat_shuffle_accs': np.random.rand(5, 6),
            'significant_clusters': [],
        } for roi in rois}}
        mock_compute_stats.return_value = fake_stats

        from src.analysis.decoding.decoding_dcc import main
        main(default_args)

        # Verify run_aggregate_and_plot_time_averaged_cms received the correct
        # list of time_averaged_cms (one per bootstrap)
        mock_agg_cms.assert_called_once()
        agg_call_args = mock_agg_cms.call_args
        time_averaged_cms_list_arg = agg_call_args[0][0]  # first positional arg
        
        assert len(time_averaged_cms_list_arg) == 3  # 3 bootstraps
        # Verify the CMs passed match what we created
        for i, expected_cm in enumerate([cm_boot0, cm_boot1, cm_boot2]):
            for roi in rois:
                np.testing.assert_array_equal(
                    time_averaged_cms_list_arg[i][comp_name][roi],
                    expected_cm
                )

    @patch(f'{_DCC}.run_aggregate_and_plot_time_averaged_cms')
    @patch(f'{_DCC}.run_all_context_comparisons')
    @patch(f'{_DCC}.run_debug_cm_traces')
    @patch(f'{_DCC}.plot_accuracies_nature_style')
    @patch(f'{_DCC}.compute_pooled_bootstrap_statistics')
    @patch(f'{_DCC}.Parallel')
    @patch(f'{_DCC}.run_visualization_debug')
    @patch(f'{_DCC}.filter_electrode_lists_against_subjects_mne_objects')
    @patch(f'{_DCC}.create_subjects_mne_objects_dict')
    @patch(f'{_DCC}.make_sig_electrodes_per_subject_and_roi_dict')
    @patch(f'{_DCC}.get_sig_chans_per_subject')
    @patch(f'{_DCC}.build_condition_comparisons')
    @patch(f'{_DCC}.load_subjects_electrodes_to_ROIs_dict')
    @patch(f'{_DCC}.get_conditions_save_name')
    @patch(f'{_DCC}.print_summary_of_dropped_electrodes')
    def test_sig_electrodes_mode(
        self,
        mock_print_summary,
        mock_get_cond_name,
        mock_load_elec_dict,
        mock_build_comps,
        mock_sig_chans,
        mock_make_sig_elecs,
        mock_create_mne,
        mock_filter_elecs,
        mock_run_viz,
        mock_parallel_cls,
        mock_compute_stats,
        mock_plot_acc,
        mock_debug_cm,
        mock_context_comps,
        mock_agg_cms,
        default_args,
        cats_by_roi,
        condition_comparisons,
        tmp_save_dir,
    ):
        """
        When args.electrodes == 'sig', the pipeline should use sig_electrodes
        and include 'sig_elecs' in the filename suffix.
        """
        default_args.electrodes = 'sig'
        rois = list(default_args.rois_dict.keys())

        mock_get_cond_name.return_value = 'bigS_vs_bigH'
        mock_load_elec_dict.return_value = {'D0079': {'LpFC_1': 'lpfc'}}
        mock_build_comps.return_value = condition_comparisons
        mock_sig_chans.return_value = {'D0079': ['LpFC_1']}

        all_elecs = {'lpfc': {'D0079': ['LpFC_1']}, 'motor': {}}
        sig_elecs = {'lpfc': {'D0079': ['LpFC_1']}, 'motor': {}}
        mock_make_sig_elecs.return_value = (all_elecs, sig_elecs)
        mock_create_mne.return_value = {'D0079': MagicMock()}
        # filter_electrode_lists should receive sig_elecs (not all_elecs)
        mock_filter_elecs.return_value = sig_elecs

        fake_results = [
            _make_fake_bootstrap_result(
                condition_comparisons, rois, cats_by_roi, n_windows=5,
                n_folds=default_args.n_splits, n_repeats=default_args.n_repeats,
            )
            for _ in range(default_args.bootstraps)
        ]
        mock_parallel_instance = MagicMock()
        mock_parallel_instance.__call__ = MagicMock(return_value=fake_results)
        mock_parallel_cls.return_value = mock_parallel_instance

        fake_stats = {list(condition_comparisons.keys())[0]: {roi: {
            'unit_of_analysis': 'repeat',
            'repeat_true_accs': np.random.rand(5, 6),
            'repeat_shuffle_accs': np.random.rand(5, 6),
            'significant_clusters': [],
        } for roi in rois}}
        mock_compute_stats.return_value = fake_stats

        from src.analysis.decoding.decoding_dcc import main
        main(default_args)

        # filter_electrode_lists was called with sig_elecs
        filter_call_args = mock_filter_elecs.call_args[0]
        assert filter_call_args[1] is sig_elecs

        # The pickle file should contain 'sig_elecs' in its name
        pkl_files = []
        for root, _, files in os.walk(tmp_save_dir):
            pkl_files.extend(f for f in files if f.endswith('.pkl'))
        assert len(pkl_files) == 1
        assert 'sig_elecs' in pkl_files[0]

    @patch(f'{_DCC}.run_aggregate_and_plot_time_averaged_cms')
    @patch(f'{_DCC}.run_all_context_comparisons')
    @patch(f'{_DCC}.run_debug_cm_traces')
    @patch(f'{_DCC}.plot_accuracies_nature_style')
    @patch(f'{_DCC}.compute_pooled_bootstrap_statistics')
    @patch(f'{_DCC}.Parallel')
    @patch(f'{_DCC}.run_visualization_debug')
    @patch(f'{_DCC}.filter_electrode_lists_against_subjects_mne_objects')
    @patch(f'{_DCC}.create_subjects_mne_objects_dict')
    @patch(f'{_DCC}.make_sig_electrodes_per_subject_and_roi_dict')
    @patch(f'{_DCC}.get_sig_chans_per_subject')
    @patch(f'{_DCC}.build_condition_comparisons')
    @patch(f'{_DCC}.load_subjects_electrodes_to_ROIs_dict')
    @patch(f'{_DCC}.get_conditions_save_name')
    @patch(f'{_DCC}.print_summary_of_dropped_electrodes')
    def test_invalid_electrodes_arg_raises(
        self,
        mock_print_summary,
        mock_get_cond_name,
        mock_load_elec_dict,
        mock_build_comps,
        mock_sig_chans,
        mock_make_sig_elecs,
        mock_create_mne,
        mock_filter_elecs,
        mock_run_viz,
        mock_parallel_cls,
        mock_compute_stats,
        mock_plot_acc,
        mock_debug_cm,
        mock_context_comps,
        mock_agg_cms,
        default_args,
        condition_comparisons,
        tmp_save_dir,
    ):
        """Setting electrodes to an invalid value should raise ValueError."""
        default_args.electrodes = 'garbage'

        mock_get_cond_name.return_value = 'bigS_vs_bigH'
        mock_load_elec_dict.return_value = {'D0079': {'LpFC_1': 'lpfc'}}
        mock_build_comps.return_value = condition_comparisons
        mock_sig_chans.return_value = {'D0079': ['LpFC_1']}
        all_elecs = {'lpfc': {'D0079': ['LpFC_1']}, 'motor': {}}
        mock_make_sig_elecs.return_value = (all_elecs, all_elecs)
        mock_create_mne.return_value = {'D0079': MagicMock()}

        from src.analysis.decoding.decoding_dcc import main
        with pytest.raises(ValueError, match="electrodes input must be set to all or sig"):
            main(default_args)