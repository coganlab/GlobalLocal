#!/usr/bin/env python
"""
Submit script for plotting wavelet/multitaper spectrograms.
This sets up input args for and calls plot_wavelets_dcc.py
Should be wrapped in an sbatch script for cluster submission.
"""
import sys
import os
from types import SimpleNamespace

# ============================================================================
# PATH SETUP
# ============================================================================
# Detect if we're on cluster or local machine
if os.path.exists("/hpc/home"):
    # We're on the cluster
    USER = os.environ.get('USER')
    sys.path.append(f"/hpc/home/{USER}/coganlab/{USER}/GlobalLocal/IEEG_Pipelines/")
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
else:
    # Local machine (Windows)
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
from dcc_scripts.spec.plot_wavelets_dcc import main

# ============================================================================
# ANALYSIS PARAMETERS
# ============================================================================
LAB_ROOT = None # Will be determined automatically in main()

SUBJECT_ID = os.environ.get('SUBJECT_ID')
if SUBJECT_ID is None:
    raise ValueError("SUBJECT_ID environment variable not set. "
                     "Set it via sbatch --export=ALL,SUBJECT_ID=...")
# task
TASK = 'GlobalLocal'

# ============================================================================
# Condition selection - condition_label is a string key into
# condition_registry.CONDITION_REGISTRY; figures are made for every
# condition in that registry entry's conditions_obj.
CONDITION_LABEL = os.environ.get('CONDITION_LABEL')
if CONDITION_LABEL is None:
    raise ValueError("CONDITION_LABEL environment variable not set. "
                     "Set it via sbatch --export=ALL,CONDITION_LABEL=...")

SPEC_TYPE = 'wavelet' # wavelet or multitaper

RESCALED = True # plot the baseline-corrected (rescaled) TFRs

def run_analysis():
    """Plot wavelet or multitaper spectrograms."""
    args = SimpleNamespace(
        LAB_root=LAB_ROOT,
        task=TASK,
        subject_id=SUBJECT_ID,
        condition_label=CONDITION_LABEL,
        spec_type=SPEC_TYPE,
        rescaled=RESCALED
    )

    # Print configuration summary
    print("=" * 70)
    print("PLOT WAVELETS ANALYSIS")
    print("=" * 70)
    print(f"Subject id:        {SUBJECT_ID}")
    print(f"Condition label:   {CONDITION_LABEL}")
    print(f"spec type:         {SPEC_TYPE}")
    print(f"rescaled:          {RESCALED}")

    # Run the analysis
    print("\nStarting analysis...")
    try:
        main(args)
        print("\n✓ Analysis completed successfully!")
    except Exception as e:
        print(f"\n✗ Analysis failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_analysis()