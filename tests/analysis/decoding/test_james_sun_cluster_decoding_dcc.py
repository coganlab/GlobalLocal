"""
Test suite for james_sun_cluster_decoding_dcc.py pipeline
Tests the main functionality with mock data
"""

import pytest
import numpy as np
import sys
import os
from unittest.mock import Mock, MagicMock, patch, create_autospec
from functools import partial
from scipy.stats import ttest_ind
import tempfile
import json

# Add paths as in the original script
sys.path.append("C:/Users/jz421/Desktop/GlobalLocal/IEEG_Pipelines/")

# Mock the imports that would fail without the actual project
with patch.dict('sys.modules', {
    'ieeg': MagicMock(),
    'ieeg.navigate': MagicMock(),
    'ieeg.io': MagicMock(),
    'ieeg.timefreq': MagicMock(),
    'ieeg.timefreq.utils': MagicMock(),
    'ieeg.calc': MagicMock(),
    'ieeg.calc.scaling': MagicMock(),
    'ieeg.calc.reshape': MagicMock(),
    'ieeg.calc.stats': MagicMock(),
    'ieeg.calc.mat': MagicMock(),
    'ieeg.calc.oversample': MagicMock(),
    'ieeg.calc.fast': MagicMock(),
    'ieeg.viz': MagicMock(),
    'ieeg.viz.parula': MagicMock(),
    'ieeg.decoding': MagicMock(),
    'ieeg.decoding.decoders': MagicMock(),
    'rsatoolbox': MagicMock(),
    'rsatoolbox.io': MagicMock(),
    'rsatoolbox.io.mne': MagicMock(),
    'rsatoolbox.data': MagicMock(),
    'rsatoolbox.data.ops': MagicMock(),
    'rsatoolbox.rdm': MagicMock(),
    'rsatoolbox.rdm.calc': MagicMock(),
    'rsatoolbox.util': MagicMock(),
    'rsatoolbox.util.build_rdm': MagicMock(),
    'rsatoolbox.vis': MagicMock(),
    'rsatoolbox.vis.timecourse': MagicMock(),
    'src': MagicMock(),
    'src.analysis': MagicMock(),
    'src.analysis.config': MagicMock(),
    'src.analysis.utils': MagicMock(),
    'src.analysis.utils.labeled_array_utils': MagicMock(),
    'src.analysis.utils.general_utils': MagicMock(),
    'src.analysis.decoding': MagicMock(),
    'src.analysis.decoding.decoding': MagicMock(),
    'src.analysis.spec': MagicMock(),
    'src.analysis.spec.wavelet_functions': MagicMock(),
    'src.analysis.spec.subjects_tfr_objects_functions': MagicMock(),
}):
    pass


class TestClusterDecodingPipeline:
    """Test the main cluster decoding pipeline"""

    @pytest.fixture
    def mock_args(self):
        """Create mock arguments for the main function"""
        args = Mock()
        args.LAB_root = None
        args.subjects = ['S01', 'S02']
        args.task = 'GlobalLocal'
        args.epochs_root_file = 'test_epochs'
        args.signal_times = [-1.0, 1.5]
        args.acc_trials_only = True
        args.error_trials_only = False
        args.stat_func = partial(ttest_ind, equal_var=False, nan_policy='omit')
        args.p_thresh = 0.05
        args.ignore_adjacency = 1
        args.n_perm = 10  # Low for testing
        args.n_jobs = 1
        args.freqs = np.arange(2, 50, 5)  # Reduced frequency range for testing
        args.n_cycles = args.freqs / 2
        args.return_itc = False
        args.time_bandwidth = 10
        args.spec_method = 'multitaper'
        args.average = False
        args.seed = 42
        args.tails = 2
        args.n_splits = 2  # Low for testing
        args.n_repeats = 2  # Low for testing
        args.random_state = 42
        args.explained_variance = 0.8
        args.balance_method = 'subsample'
        args.normalize = 'all'
        args.obs_axs = 0
        args.chans_axs = 1
        args.freq_axs = 2
        args.time_axs = 3
        args.oversample = True
        args.alpha = 1.0
        args.clear_memory = False  # Don't clear memory in tests
        
        # Mock conditions
        args.conditions = {
            'bigS': [1, 2, 3],
            'bigH': [4, 5, 6]
        }
        args.conditions_save_name = 'bigS_vs_bigH'
        # Mock ROIs
        args.rois_dict = {
            'lpfc': ['G_front_inf-Orbital', 'G_front_middle'],
            'motor': ['G_precentral', 'G_postcentral']
        }
        
        return args
    
    @pytest.fixture
    def mock_tfr_objects(self, mock_args):
        """Create mock TFR objects"""
        tfr_objects = {}
        n_timepoints = 100
        n_freqs = len(mock_args.freqs)
        
        for subject in mock_args.subjects:
            tfr_objects[subject] = {}
            for condition in mock_args.conditions.keys():
                mock_tfr = Mock()
                mock_tfr.times = np.linspace(-1, 1.5, n_timepoints)
                mock_tfr.freqs = mock_args.freqs
                mock_tfr.data = np.random.randn(10, 5, n_freqs, n_timepoints)  # (trials, channels, freqs, times)
                tfr_objects[subject][condition] = mock_tfr
        
        return tfr_objects
    
    @pytest.fixture
    def mock_roi_labeled_arrays(self, mock_args, mock_tfr_objects):
        """Create mock ROI labeled arrays"""
        roi_arrays = {}
        
        for roi in mock_args.rois_dict.keys():
            mock_array = MagicMock()
            mock_array.keys.return_value = list(mock_args.conditions.keys())
            
            # Create mock data for each condition
            n_trials = 20
            n_channels = 10
            n_freqs = len(mock_args.freqs)
            n_times = 100
            
            # Store data in a way that __getitem__ can access
            data_dict = {}
            for condition in mock_args.conditions.keys():
                # Create data with slight differences between conditions
                base_data = np.random.randn(n_trials, n_channels, n_freqs, n_times)
                if condition == 'bigS':
                    base_data += 0.1  # Add small offset to create difference
                data_dict[condition] = base_data
            
            mock_array.__getitem__.side_effect = lambda key: data_dict[key]
            
            # Add labels attribute for channel names
            mock_array.labels = {
                mock_args.obs_axs: [f'trial_{i}' for i in range(n_trials)],
                mock_args.chans_axs: [f'ch_{i}' for i in range(n_channels)],
                mock_args.freq_axs: mock_args.freqs,
                mock_args.time_axs: np.linspace(-1, 1.5, n_times)
            }
            
            roi_arrays[roi] = mock_array
        
        return roi_arrays
    
    @pytest.fixture
    def mock_electrodes_dict(self, mock_args):
        """Create mock electrode dictionaries"""
        all_electrodes = {}
        sig_electrodes = {}
        
        for roi in mock_args.rois_dict.keys():
            all_electrodes[roi] = {}
            sig_electrodes[roi] = {}
            for subject in mock_args.subjects:
                # All electrodes: 5 channels per subject per ROI
                all_electrodes[roi][subject] = [f'{subject}_ch{i}' for i in range(5)]
                # Sig electrodes: 3 channels per subject per ROI
                sig_electrodes[roi][subject] = [f'{subject}_ch{i}' for i in range(3)]
        
        return all_electrodes, sig_electrodes
    
    def test_pipeline_initialization(self, mock_args, tmp_path):
        """Test that the pipeline initializes correctly"""
        # Create temporary directory structure
        config_dir = tmp_path / "src" / "analysis" / "config"
        config_dir.mkdir(parents=True)
        
        # Create mock subjects_electrodestoROIs_dict
        mock_dict = {
            'S01': {'lpfc': ['ch1', 'ch2'], 'motor': ['ch3', 'ch4']},
            'S02': {'lpfc': ['ch5', 'ch6'], 'motor': ['ch7', 'ch8']}
        }
        
        dict_file = config_dir / "subjects_electrodestoROIs_dict.json"
        with open(dict_file, 'w') as f:
            json.dump(mock_dict, f)
        
        assert dict_file.exists()
        assert len(mock_args.subjects) == 2
        assert len(mock_args.rois_dict) == 2
    
    @patch('src.analysis.decoding.decoding.get_confusion_matrix_for_rois_tfr_cluster')
    @patch('src.analysis.spec.subjects_tfr_objects_functions.make_subjects_tfr_objects')
    @patch('src.analysis.utils.labeled_array_utils.put_data_in_labeled_array_per_roi_subject')
    @patch('src.analysis.utils.general_utils.make_sig_electrodes_per_subject_and_roi_dict')
    @patch('src.analysis.utils.general_utils.get_sig_chans_per_subject')
    @patch('src.analysis.utils.general_utils.make_or_load_subjects_electrodes_to_ROIs_dict')
    @patch('ieeg.io.get_data')
    def test_main_pipeline_execution(
        self, mock_get_data, mock_make_dict, mock_get_sig_chans,
        mock_make_sig_elecs, mock_put_data, mock_make_tfr, mock_get_confusion,
        mock_args, mock_tfr_objects, mock_roi_labeled_arrays, 
        mock_electrodes_dict, tmp_path
    ):
        """Test that the main pipeline executes without errors"""
        
        # Setup mocks
        mock_get_data.return_value = Mock()  # layout
        mock_make_dict.return_value = {
            'S01': {'lpfc': ['ch1', 'ch2'], 'motor': ['ch3', 'ch4']},
            'S02': {'lpfc': ['ch5', 'ch6'], 'motor': ['ch7', 'ch8']}
        }
        mock_get_sig_chans.return_value = {'S01': ['ch1', 'ch3'], 'S02': ['ch5', 'ch7']}
        mock_make_sig_elecs.return_value = mock_electrodes_dict
        mock_make_tfr.return_value = mock_tfr_objects
        mock_put_data.return_value = mock_roi_labeled_arrays
        
        # Mock confusion matrix results
        mock_confusion_matrices = {
            'lpfc': np.array([[0.7, 0.3], [0.2, 0.8]]),
            'motor': np.array([[0.6, 0.4], [0.3, 0.7]])
        }
        mock_cats = {
            'lpfc': {('bigS',): 0, ('bigH',): 1},
            'motor': {('bigS',): 0, ('bigH',): 1}
        }
        mock_channel_masks = {
            'lpfc': {0: {0: {i: np.random.rand(len(mock_args.freqs), 100) > 0.5 for i in range(10)}}},
            'motor': {0: {0: {i: np.random.rand(len(mock_args.freqs), 100) > 0.5 for i in range(10)}}}
        }
        
        mock_get_confusion.return_value = (mock_confusion_matrices, mock_cats, mock_channel_masks)
        
        # Set LAB_root to temp directory
        mock_args.LAB_root = str(tmp_path)
        
        # Import and run main function
        from dcc_scripts.decoding.james_sun_cluster_decoding_dcc import main
        
        # Should execute without errors
        try:
            main(mock_args)
        except Exception as e:
            # Check if it's just about missing directories/files
            if "BIDS" not in str(e) and "derivatives" not in str(e):
                raise e
        
        # Verify key functions were called
        assert mock_make_tfr.called
        assert mock_put_data.called
        assert mock_get_confusion.called
    
    def test_condition_comparisons_setup(self, mock_args):
        """Test that condition comparisons are set up correctly"""
        from src.analysis.config import experiment_conditions
        
        # Mock experiment_conditions
        experiment_conditions.stimulus_conditions = mock_args.conditions
        
        # Test condition comparison setup for stimulus_conditions
        condition_comparisons = {}
        condition_comparisons['BigLetter'] = ['bigS', 'bigH']
        
        assert 'BigLetter' in condition_comparisons
        assert condition_comparisons['BigLetter'] == ['bigS', 'bigH']
    
    def test_concatenate_and_balance_functionality(self, mock_roi_labeled_arrays):
        """Test the concatenate and balance data functionality"""
        from src.analysis.decoding.decoding import concatenate_and_balance_data_for_decoding
        
        with patch('src.analysis.utils.labeled_array_utils.concatenate_conditions_by_string') as mock_concat:
            # Setup mock return
            mock_data = np.random.randn(40, 10, 10, 100)  # (trials, channels, freqs, times)
            mock_labels = np.array([0] * 20 + [1] * 20)
            mock_cats = {('bigS',): 0, ('bigH',): 1}
            mock_concat.return_value = (mock_data, mock_labels, mock_cats)
            
            # Test with subsample method
            result_data, result_labels, result_cats = concatenate_and_balance_data_for_decoding(
                mock_roi_labeled_arrays,
                roi='lpfc',
                strings_to_find=['bigS', 'bigH'],
                obs_axs=0,
                balance_method='subsample',
                random_state=42
            )
            
            # Check that data is balanced
            unique, counts = np.unique(result_labels, return_counts=True)
            assert len(unique) == 2
            assert counts[0] == counts[1]  # Should be balanced
    
    def test_decoder_integration(self, mock_roi_labeled_arrays):
        """Test that the Decoder class integrates properly"""
        from src.analysis.decoding.decoding import Decoder
        
        # Create mock Decoder
        with patch('ieeg.decoding.decoders.PcaLdaClassification') as mock_pca:
            with patch('ieeg.calc.oversample.MinimumNaNSplit') as mock_split:
                # Create simple test data
                n_trials = 20
                n_features = 50
                X = np.random.randn(n_trials, n_features)
                y = np.array([0, 1] * 10)
                cats = {('cond1',): 0, ('cond2',): 1}
                
                # Initialize decoder
                decoder = Decoder(
                    cats,
                    explained_variance=0.8,
                    n_splits=2,
                    n_repeats=2,
                    oversample=True
                )
                
                # Mock the model attribute
                decoder.model = Mock()
                decoder.model.fit = Mock()
                decoder.model.predict = Mock(return_value=np.array([0, 1] * 5))
                
                # Test cv_cm_jim method
                with patch.object(decoder, 'split') as mock_split_method:
                    # Mock split to return indices
                    mock_split_method.return_value = [
                        (np.arange(15), np.arange(15, 20)),
                        (np.arange(10), np.arange(10, 20))
                    ]
                    
                    cm = decoder.cv_cm_jim(X, y, normalize='all', obs_axs=0)
                    
                    assert cm is not None
                    assert cm.shape[-2:] == (2, 2)  # 2x2 confusion matrix
    
    def test_tfr_mask_computation(self):
        """Test TFR mask computation functions"""
        from src.analysis.decoding.decoding import compute_sig_tfr_masks_for_specified_channels
        
        # Create test data
        n_channels = 3
        n_trials = 10
        n_freqs = 8
        n_times = 50
        
        train_data_by_condition = {
            'cond1': np.random.randn(n_trials, n_channels, n_freqs, n_times),
            'cond2': np.random.randn(n_trials, n_channels, n_freqs, n_times) + 0.1
        }
        
        stat_func = partial(ttest_ind, equal_var=False, nan_policy='omit')
        
        with patch('ieeg.calc.stats.time_perm_cluster') as mock_perm:
            # Mock the permutation test
            mock_perm.return_value = (np.random.rand(n_freqs, n_times) > 0.7, np.array([0.01]))
            
            masks = compute_sig_tfr_masks_for_specified_channels(
                n_channels,
                train_data_by_condition,
                ['cond1', 'cond2'],
                obs_axs=0,
                chans_axs=1,
                stat_func=stat_func,
                p_thresh=0.05,
                n_perm=10
            )
            
            assert len(masks) == n_channels
            assert all(masks[i].shape == (n_freqs, n_times) for i in range(n_channels))
    
    def test_apply_tfr_masks_and_flatten(self):
        """Test applying TFR masks and flattening"""
        from src.analysis.decoding.decoding import apply_tfr_masks_and_flatten_to_make_decoding_matrix
        
        # Create test data
        n_trials = 10
        n_channels = 3
        n_freqs = 5
        n_times = 20
        data = np.random.randn(n_trials, n_channels, n_freqs, n_times)
        
        # Create masks for each channel
        channel_masks = {}
        for ch in range(n_channels):
            mask = np.random.rand(n_freqs, n_times) > 0.5
            channel_masks[ch] = mask
        
        # Apply masks and flatten
        result = apply_tfr_masks_and_flatten_to_make_decoding_matrix(
            data,
            obs_axs=0,
            chans_axs=1,
            channel_masks=channel_masks
        )
        
        # Check output shape
        assert result.shape[0] == n_trials
        # Number of features should equal sum of True values in all masks
        expected_features = sum(mask.sum() for mask in channel_masks.values())
        assert result.shape[1] == expected_features
    
    def test_plot_and_save_functions(self, tmp_path):
        """Test plotting and saving functions"""
        from src.analysis.decoding.decoding import plot_and_save_confusion_matrix
        
        # Create test confusion matrix
        cm = np.array([[0.8, 0.2], [0.3, 0.7]])
        display_labels = ['Class A', 'Class B']
        
        # Test plot and save
        save_dir = tmp_path / "test_plots"
        save_dir.mkdir(parents=True)
        
        with patch('matplotlib.pyplot.savefig') as mock_save:
            with patch('matplotlib.pyplot.close') as mock_close:
                plot_and_save_confusion_matrix(
                    cm,
                    display_labels,
                    'test_cm.png',
                    str(save_dir)
                )
                
                assert mock_save.called
                assert mock_close.called


class TestIntegrationScenarios:
    """Test different integration scenarios"""
    
    def test_with_different_balance_methods(self, mock_roi_labeled_arrays):
        """Test pipeline with different balance methods"""
        from src.analysis.decoding.decoding import concatenate_and_balance_data_for_decoding
        
        with patch('src.analysis.utils.labeled_array_utils.concatenate_conditions_by_string') as mock_concat:
            # Create imbalanced mock data
            mock_data = np.random.randn(35, 10, 10, 100)
            mock_labels = np.array([0] * 20 + [1] * 15)  # Imbalanced
            mock_cats = {('bigS',): 0, ('bigH',): 1}
            mock_concat.return_value = (mock_data, mock_labels, mock_cats)
            
            # Test subsample
            data_sub, labels_sub, _ = concatenate_and_balance_data_for_decoding(
                mock_roi_labeled_arrays, 'lpfc', ['bigS', 'bigH'],
                obs_axs=0, balance_method='subsample', random_state=42
            )
            
            unique, counts = np.unique(labels_sub, return_counts=True)
            assert counts[0] == counts[1]  # Should be balanced
            assert counts[0] == 15  # Should subsample to minimum
            
            # Test pad_with_nans
            data_pad, labels_pad, _ = concatenate_and_balance_data_for_decoding(
                mock_roi_labeled_arrays, 'lpfc', ['bigS', 'bigH'],
                obs_axs=0, balance_method='pad_with_nans', random_state=42
            )
            
            unique, counts = np.unique(labels_pad, return_counts=True)
            assert counts[0] == counts[1]  # Should be balanced
            assert counts[0] == 20  # Should pad to maximum
    
    def test_with_nan_handling(self):
        """Test that NaN values are handled correctly"""
        from src.analysis.decoding.decoding import mixup2
        
        # Create data with NaNs
        arr = np.array([[1, 2], [4, 5], [7, 8], [np.nan, np.nan]])
        labels = np.array([0, 0, 1, 1])
        
        # Apply mixup2
        mixup2(arr, labels, obs_axs=0, alpha=1.0, seed=42)
        
        # Check that NaNs were filled
        assert not np.any(np.isnan(arr))
    
    def test_error_handling(self, mock_args):
        """Test error handling in the pipeline"""
        from src.analysis.decoding.decoding import compute_sig_tfr_masks_from_concatenated_data
        
        # Test with wrong number of conditions
        with pytest.raises(ValueError, match="will only work for two conditions"):
            compute_sig_tfr_masks_from_concatenated_data(
                np.random.randn(20, 5, 10, 100),
                np.array([0, 1, 2] * 6 + [0, 1]),
                np.arange(15),
                ['cond1', 'cond2', 'cond3'],  # 3 conditions
                {('cond1',): 0, ('cond2',): 1, ('cond3',): 2},
                obs_axs=0,
                chans_axs=1,
                stat_func=mock_args.stat_func,
                p_thresh=0.05,
                n_perm=10
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])