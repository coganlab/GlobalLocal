#!/usr/bin/env python3
"""
Add experimentStart event to the first run of BIDS events files for specified subjects.

Usage:
    python add_experiment_start.py --subjects D0107A D0110 D0116
    python add_experiment_start.py --subjects D0107A --lab-root /custom/path/to/CoganLab
"""

import os
import pandas as pd
import argparse
from pathlib import Path
import glob
import shutil
from datetime import datetime


def get_lab_root(LAB_ROOT=None):
    """Get the lab root directory based on the operating system."""
    if LAB_ROOT is None:
        HOME = os.path.expanduser("~")
        if os.name == 'nt':  # windows
            LAB_ROOT = os.path.join(HOME, "Box", "CoganLab")
        else:  # mac
            LAB_ROOT = os.path.join(HOME, "Library", "CloudStorage", "Box-Box", "CoganLab")
    return LAB_ROOT


def add_experiment_start(events_file, backup=True):
    """
    Add experimentStart event to an events TSV file.
    
    Parameters:
    -----------
    events_file : str or Path
        Path to the events TSV file
    backup : bool
        Whether to create a backup of the original file
    
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    events_file = Path(events_file)
    
    # Check if file exists
    if not events_file.exists():
        print(f"Error: File not found: {events_file}")
        return False
    
    try:
        # Read the events file
        events = pd.read_csv(events_file, sep='\t')
        
        # Check if experimentStart already exists
        if 'trial_type' in events.columns and 'experimentStart' in events['trial_type'].values:
            print(f"  experimentStart already exists in {events_file.name}")
            return True
        
        # Create backup if requested
        if backup:
            backup_file = events_file.with_suffix('.tsv.bak')
            shutil.copy2(events_file, backup_file)
            print(f"  Created backup: {backup_file.name}")
        
        # Get the columns from the original file to maintain order
        columns = events.columns.tolist()
        
        # Create experimentStart row with appropriate values for each column
        experiment_start_data = {}
        for col in columns:
            if col == 'onset':
                # Use 0.0 for onset (beginning of experiment)
                experiment_start_data[col] = 0.0
            elif col == 'duration':
                experiment_start_data[col] = 0.0
            elif col == 'trial_type':
                experiment_start_data[col] = 'experimentStart'
            elif col == 'value':
                experiment_start_data[col] = 'n/a'
            elif col == 'sample':
                experiment_start_data[col] = 0
            else:
                # For any other columns, use 'n/a' or appropriate default
                if events[col].dtype in ['float64', 'int64']:
                    experiment_start_data[col] = 0
                else:
                    experiment_start_data[col] = 'n/a'
        
        # Create DataFrame with experimentStart
        experiment_start_df = pd.DataFrame([experiment_start_data])
        
        # Concatenate at the beginning
        events_updated = pd.concat([experiment_start_df, events], ignore_index=True)
        
        # Save back to file
        events_updated.to_csv(events_file, sep='\t', index=False, na_rep='n/a')
        print(f"  Successfully added experimentStart to {events_file.name}")
        return True
        
    except Exception as e:
        print(f"Error processing {events_file}: {str(e)}")
        return False


def process_subjects(subjects, lab_root=None, task='GlobalLocal', backup=True):
    """
    Process multiple subjects to add experimentStart to their first run.
    
    Parameters:
    -----------
    subjects : list
        List of subject IDs
    lab_root : str or None
        Path to lab root directory
    task : str
        Task name (default: 'GlobalLocal')
    backup : bool
        Whether to create backups of original files
    """
    # Get lab root
    lab_root = get_lab_root(lab_root)
    bids_root = Path(lab_root) / "BIDS-1.1_GlobalLocal" / "BIDS" / "derivatives" / "clean"
    
    print(f"Processing {len(subjects)} subjects...")
    print(f"BIDS root: {bids_root}")
    print(f"Task: {task}")
    print(f"Create backups: {backup}")
    print("-" * 50)
    
    # Track results
    successful = []
    failed = []
    
    for subject in subjects:
        print(f"\nProcessing subject: {subject}")
        
        # Build path to subject's ieeg directory
        subject_dir = bids_root / f"sub-{subject}" / "ieeg"
        
        if not subject_dir.exists():
            print(f"  Warning: Subject directory not found: {subject_dir}")
            failed.append(subject)
            continue
        
        # Find all runs for this subject and task
        pattern = f"sub-{subject}_task-{task}_acq-*_run-*_desc-clean_events.tsv"
        events_files = sorted(glob.glob(str(subject_dir / pattern)))
        
        if not events_files:
            print(f"  Warning: No events files found for pattern: {pattern}")
            failed.append(subject)
            continue
        
        print(f"  Found {len(events_files)} run(s)")
        
        # Process only the first run (run-01)
        first_run_pattern = f"sub-{subject}_task-{task}_acq-*_run-01_desc-clean_events.tsv"
        first_run_files = [f for f in events_files if 'run-01' in f]
        
        if not first_run_files:
            print(f"  Warning: No run-01 file found")
            failed.append(subject)
            continue
        
        # Process the first run file
        first_run_file = first_run_files[0]
        print(f"  Processing: {Path(first_run_file).name}")
        
        if add_experiment_start(first_run_file, backup=backup):
            successful.append(subject)
        else:
            failed.append(subject)
    
    # Print summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Successfully processed: {len(successful)} subjects")
    if successful:
        print(f"  Subjects: {', '.join(successful)}")
    
    if failed:
        print(f"\nFailed to process: {len(failed)} subjects")
        print(f"  Subjects: {', '.join(failed)}")
    
    print(f"\nBackup files created with .bak extension" if backup else "\nNo backup files created")


def main():
    parser = argparse.ArgumentParser(
        description="Add experimentStart event to BIDS events files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python add_experiment_start.py --subjects D0107A D0108B D0109C
  python add_experiment_start.py --subjects D0107A --lab-root /custom/path
  python add_experiment_start.py --subjects D0107A D0108B --task GlobalLocal --no-backup
        """
    )
    
    parser.add_argument(
        '--subjects', '-s',
        nargs='+',
        required=True,
        help='Subject IDs to process (without sub- prefix)'
    )
    
    parser.add_argument(
        '--lab-root', '-l',
        type=str,
        default=None,
        help='Path to CoganLab root directory (auto-detected if not provided)'
    )
    
    parser.add_argument(
        '--task', '-t',
        type=str,
        default='GlobalLocal',
        help='Task name (default: GlobalLocal)'
    )
    
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Do not create backup files'
    )
    
    args = parser.parse_args()
    
    # Process subjects
    process_subjects(
        subjects=args.subjects,
        lab_root=args.lab_root,
        task=args.task,
        backup=not args.no_backup
    )


if __name__ == "__main__":
    main()