# this tests get_sig_tfr_differences.py
import sys
import os
import numpy as np
from functools import partial
from scipy.stats import ttest_ind

# Path setup (same as yours)
sys.path.append("C:/Users/jz421/Desktop/GlobalLocal/IEEG_Pipelines/")

try:
    current_file_path = os.path.abspath(__file__)
    current_script_dir = os.path.dirname(current_file_path)
except NameError:
    current_script_dir = os.getcwd()

project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..'))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import after path is set up
from dcc_scripts.spec.get_sig_tfr_differences_dcc import main

# Define test conditions
stimulus_big_letter_conditions = {
    "Stimulus_bigS": {
        "BIDS_events": ["Stimulus/BigLetters"],
        "bigLetter": "s",
    },
    "Stimulus_bigH": {
        "BIDS_events": ["Stimulus/BigLetterh"],
        "bigLetter": "h",
    }
}

def test_minimal():
    """Test with minimal parameters for quick execution"""
    from types import SimpleNamespace
    
    # Create minimal test arguments
    test_args = SimpleNamespace(
        LAB_root=None,
        subjects=['D0103'],
        signal_times=[-0.2, 0.2],  # 400ms window
        acc_trials_only=True,
        error_trials_only=False,
        stat_func=partial(ttest_ind, equal_var=False, nan_policy='omit'),
        p_thresh=0.5,
        ignore_adjacency=1,
        n_perm=2,  # Minimal permutations
        n_jobs=1,
        freqs=np.array([70, 80, 90]),  # Just 3 frequencies
        n_cycles=np.array([35, 40, 45]),
        return_itc=False,
        time_bandwidth=10,
        spec_method='multitaper',
        average=False,
        seed=42,
        tails=2,
        n_splits=2,
        n_repeats=1,
        random_state=42,
        task='GlobalLocal',
        conditions=stimulus_big_letter_conditions,
        epochs_root_file='Stimulus_0.5sec_within1sec_randoffset_preStimulusBase_decFactor_8_outliers_10_passband_70.0-150.0_padLength_0.5s_stat_func_ttest_ind_equal_var_False',
        rois_dict={'test_roi': ['G_front_inf-Opercular']}  # Single ROI
    )
    
    print("Starting minimal test...")
    try:
        main(test_args)
        print("Test completed successfully!")
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_minimal()