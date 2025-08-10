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
        # Create realistic labeled array structure
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
                # Return predictions matching the size of the input
                n_samples = X.shape[0]
                return np.array([0, 1] * (n_samples // 2) + [0] * (n_samples % 2))
            
            mock_instance.predict.side_effect = dynamic_predict
            
            # Unpack the tuple return value
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
            # Make concatenate_and_balance_data_for_decoding raise an error for invalid method
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
        cond2_data = np.random.randn(15, 5, 100)  # 15 trials
        cond3_data = np.random.randn(25, 5, 100)  # 25 trials
        
        # Add some NaN trials to test NaN handling
        cond1_data[18:, :, :] = np.nan  # Last 2 trials are NaN
        cond2_data[13:, :, :] = np.nan  # Last 2 trials are NaN
        
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
        
        # Should remove NaN trials first
        assert not np.any(np.isnan(concatenated_data))
        
        # Should have equal trials per condition (min of valid trials)
        # cond1: 18 valid trials, cond2: 13 valid trials
        # So should have 13 trials per condition = 26 total
        assert concatenated_data.shape[0] == 26
        assert np.sum(labels == 0) == 13  # cond1
        assert np.sum(labels == 1) == 13  # cond2
        
        # Check cats dictionary
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
        # cond1: 20 trials, cond2: 15 trials
        # Should pad cond2 to 20 trials
        assert concatenated_data.shape[0] == 40  # 20 + 20
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
        
        # cond1 (18 valid) + cond3 (25 valid) = 43 trials for class 0
        # cond2 (13 valid) = 13 trials for class 1
        # After subsampling to balance: 13 trials each
        assert np.sum(labels == 0) == 13
        assert np.sum(labels == 1) == 13
    
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
        # Run twice with same random state
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
        
        # Should get identical results
        np.testing.assert_array_equal(result1[0], result2[0])  # data
        np.testing.assert_array_equal(result1[1], result2[1])  # labels
        assert result1[2] == result2[2]  # cats
    
    def test_all_nan_condition(self):
        """Test handling when a condition has all NaN trials."""
        mock_array = MagicMock()
        
        # Create condition with all NaN trials
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
        # So final data should be empty or raise an appropriate error
        # This depends on your implementation details
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
        
        # Should return data for single condition
        assert concatenated_data.shape[0] == 10
        assert np.all(labels == 0)  # All same label
        assert cats == {('cond1',): 0}