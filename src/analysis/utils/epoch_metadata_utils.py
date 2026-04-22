"""
Utility to parse MNE epoch event name strings into a metadata DataFrame.

Add this to your general_utils.py or import it in make_epoched_data.py.

Usage:
    trials = trial_ieeg(good, event, times_adj, preload=True, reject_by_annotation=False)
    trials.metadata = make_metadata_from_event_names(trials)
"""

import re
import pandas as pd
import numpy as np
import mne


def parse_event_name(event_name):
    """
    Parse a single MNE event name string into a dict.
    
    Example event name:
        'Stimulus/i25.0/r25.0/BigLetters/SmallLetterh/Taskg/TargetLetters/
         Responded1.0/ParticipantResponse115.0/CorrectResponse115.0/
         TrialCount261.0/BlockTrialCount37.0/ReactionTime1350.0/Accuracy1.0/D57'
    
    Returns a dict like:
        {
            'event': 'Stimulus',
            'congruency': 'i',
            'incongruent_proportion': 25.0,
            'task_sequence': 'r',
            'switch_proportion': 25.0,
            'big_letter': 's',
            'small_letter': 'h',
            'task': 'g',
            'target_letter': 's',
            'responded': 1.0,
            'participant_response': 115.0,
            'correct_response': 115.0,
            'trial_count': 261.0,
            'block_trial_count': 37.0,
            'reaction_time': 1350.0,
            'accuracy': 1.0,
            'subject': 'D57',
            'full_event_name': <the original string>
        }
    """
    parts = event_name.split("/")
    info = {'full_event_name': event_name}
    
    for part in parts:
        
        # ---- Handle the congruency/proportion field (e.g., "i25.0", "i75.0") ----
        # "i" is always followed by a number. We want to extract BOTH:
        #   - The letter "i" itself (congruency marker)
        #   - The number after it (the incongruent proportion)
        if part.startswith(('i', 'c')) and len(part) > 1 and part[1].isdigit():
            # part = "i25.0"
            # part[0]  = "i"        -> the congruency label
            # part[1:] = "25.0"     -> the proportion as a string
            info['congruency'] = part[0]                # 'i'
            info['incongruent_proportion'] = float(part[1:])  # 25.0
        
        # ---- Handle the task sequence/switch proportion field (e.g., "r25.0", "s75.0", "n75.0") ----
        # Same pattern: a single letter followed by a number.
        # "r" = repeat, "s" = switch, "n" = neither (or similar)
        elif part.startswith(('r', 's', 'n')) and len(part) > 1 and part[1].isdigit():
            # part = "r25.0"
            # part[0]  = "r"        -> task sequence type
            # part[1:] = "25.0"     -> switch proportion
            info['task_sequence'] = part[0]              # 'r'
            info['switch_proportion'] = float(part[1:])  # 25.0
        
        # ---- Handle "BigLetters" -> big_letter = 's' ----
        # The actual letter identity is the last character(s) after "BigLetter"
        elif part.startswith('BigLetter'):
            # part = "BigLetters"
            # part[len('BigLetter'):]  = "s"
            # We slice off the prefix "BigLetter" (9 characters) to get the rest
            info['big_letter'] = part[len('BigLetter'):]   # 's'
        
        # ---- Handle "SmallLetterh" -> small_letter = 'h' ----
        elif part.startswith('SmallLetter'):
            # part = "SmallLetterh"
            # part[len('SmallLetter'):]  = "h"
            info['small_letter'] = part[len('SmallLetter'):]  # 'h'
        
        # ---- Handle "Taskg" -> task = 'g' ----
        elif part.startswith('Task'):
            # part = "Taskg"
            # part[len('Task'):]  = "g"
            info['task'] = part[len('Task'):]  # 'g'
        
        # ---- Handle "TargetLetters" -> target_letter = 's' ----
        elif part.startswith('TargetLetter'):
            # part = "TargetLetters"
            # part[len('TargetLetter'):]  = "s"
            info['target_letter'] = part[len('TargetLetter'):]  # 's'
        
        # ---- Handle subject ID like "D57" ----
        elif part.startswith('D') and part[1:].isdigit():
            info['subject'] = part
        
        # ---- Handle event type "Stimulus" or "Response" ----
        elif part in ('Stimulus', 'Response'):
            info['event'] = part
        
        # ---- Handle all remaining numeric fields ----
        # These are things like "Responded1.0", "TrialCount261.0", "Accuracy1.0"
        # Pattern: one or more letters, then a number (possibly negative, with decimal)
        else:
            # Try to split into a text key and numeric value
            # We walk backward from the end to find where the number starts
            split_idx = None
            for idx in range(len(part) - 1, -1, -1):
                char = part[idx]
                if not (char.isdigit() or char == '.' or char == '-'):
                    split_idx = idx + 1
                    break
            
            # If we found a split point and there's both a key and a value part
            if split_idx is not None and split_idx > 0 and split_idx < len(part):
                key = part[:split_idx]    # e.g., "TrialCount"
                val_str = part[split_idx:]  # e.g., "261.0"
                
                # Convert the key from CamelCase to snake_case
                # "TrialCount" -> "trial_count"
                # "ReactionTime" -> "reaction_time"
                snake_key = re.sub(r'([A-Z])', r'_\1', key).lower().lstrip('_')
                
                try:
                    info[snake_key] = float(val_str)
                except ValueError:
                    info[snake_key] = val_str
            else:
                # Unrecognized part, store as-is
                info[part] = True
    
    return info


def make_metadata_from_event_names(epochs):
    """
    Parse all event names in an Epochs object into a metadata DataFrame.
    
    Parameters
    ----------
    epochs : mne.Epochs or mne.EpochsArray
        The epochs object whose event names should be parsed.
    
    Returns
    -------
    pd.DataFrame
        A DataFrame with one row per epoch and columns parsed from the 
        event name strings.
    
    Example
    -------
    >>> trials = trial_ieeg(good, "Stimulus", times_adj, preload=True)
    >>> trials.metadata = make_metadata_from_event_names(trials)
    >>> accurate_trials = trials['accuracy == 1.0']
    >>> high_incongruent = trials['incongruent_proportion == 75.0']
    """
    # Map integer event codes back to their string names
    id_to_name = {v: k for k, v in epochs.event_id.items()}
    
    # Parse each epoch's event name
    records = []
    for code in epochs.events[:, 2]:
        name = id_to_name[code]
        records.append(parse_event_name(name))
    
    df = pd.DataFrame(records)
    df.index = pd.RangeIndex(len(epochs))
    
    return df


def add_previous_trial_info(metadata, fields=None):
    """
    Add columns for the previous trial's metadata, based on trial_count.
    
    For each epoch with trial_count = N, looks up the epoch with 
    trial_count = N-1 and copies specified fields with a 'prev_' prefix.
    
    Parameters
    ----------
    metadata : pd.DataFrame
        The metadata DataFrame (from make_metadata_from_event_names).
    fields : list of str, optional
        Which columns to copy from the previous trial.
    
    Returns
    -------
    pd.DataFrame
        The metadata with new 'prev_*' columns added.
    
    Example
    -------
    >>> trials.metadata = make_metadata_from_event_names(trials)
    >>> trials.metadata = add_previous_trial_info(trials.metadata)
    >>> prev_switch = trials['prev_task_sequence == "s"']
    """
    if fields is None:
        fields = ['congruency', 'incongruent_proportion', 'task_sequence',
                  'switch_proportion', 'accuracy', 'reaction_time',
                  'big_letter', 'small_letter', 'task']
    
    available_fields = [f for f in fields if f in metadata.columns]
    
    # Build lookup: trial_count -> {field: value, ...}
    lookup = {}
    for _, row in metadata.iterrows():
        tc = row.get('trial_count')
        if pd.notna(tc):
            lookup[tc] = {f: row[f] for f in available_fields}
    
    # For each row, look up trial_count - 1
    prev_data = []
    for _, row in metadata.iterrows():
        prev_tc = row.get('trial_count', np.nan)
        if pd.notna(prev_tc) and (prev_tc - 1) in lookup:
            prev_data.append(lookup[prev_tc - 1])
        else:
            prev_data.append({f: np.nan for f in available_fields})
    
    prev_df = pd.DataFrame(prev_data, index=metadata.index)
    prev_df.columns = [f'prev_{c}' for c in prev_df.columns]
    
    return pd.concat([metadata, prev_df], axis=1)


def select_and_balance_trials(
    epochs,
    query,
    class_col,
    balance_by=None,
    random_state=None
):
    """
    Select trials for decoding based on metadata query, assign class labels, optionally balance across strata.
    replaces concatenate_conditions_by_string and concatenate_and_balance_data_for_decoding
    Parameters
    ----------
    epochs : mne.Epochs
        Must have metadata attached.
    query : str or None
        Pandas query string applied to metadata. None means use all trials.
    class_col : str
        Metadata column whose values become the class labels
    balance_by : list of str or None
        Metadata columns defining the strata. Each (class x stratum) cell is subsampled to the minimum count across all cells.
        If none, classes are subsampled to the minimum class count (no stratification)
    random_state : int or RandomState
        RNG seed.
    
    Returns
    -------
    data : np.ndarray of shape (n_trials, n_channels, n_times)
    labels : np.ndarray of integer class labels
    cats : dict of {(class_value,): int} matching the old decoding cats format
    """
    rng = (random_state if isinstance(random_state, np.random.RandomState)
           else np.random.RandomState(random_state))
    
    md = epochs.metadata.reset_index(drop=True)
    if query is not None:
        mask = md.eval(query).fillna(False).to_numpy()
        idx = np.where(mask)[0]
    else:
        idx = np.arange(len(md))
    
    class_values = md.loc[idx, class_col].to_numpy()
    
    # drop rows where class is NaN (e.g., prev_* at block starts)
    keep = pd.notna(class_values)
    idx = idx[keep]
    
    class_values = class_values[keep]
    
    unique_classes = sorted(pd.unique(class_values).tolist())
    cats = {(c,): i for i,c in enumerate(unique_classes)} # TODO: check if c has to be a tuple according to the ieeg pipelines Decoder class
    
    # keep walking through the below code to understand and test it..
    # assemble the stratification key per trial - TODO: this requires adding blockType to the metadata - recreate this from the inc and switch prop, similar to how i do it for the fmri code.
    if balance_by:
        strata = md.loc[idx, balance_by].apply(
            lambda row: tuple(row.to_list()), axis=1
        ).to_numpy()
    else:
        strata = np.zeros(len(idx), dtype=object)
    
    # count per (class, stratum) cell
    cells = {} # (class, stratum) -> list of positional indices into 'idx'
    for pos, (cls, strat) in enumerate(zip(class_values, strata)):
        cells.setdefault((cls, strat), []).append(idx[pos])
    
    if not cells:
        raise ValueError(f"No trials returned by query {query!r}")
    
    n_per_cell = min(len(v) for v in cells.values())
    print(f"[select] query={query!r} class_col={class_col!r} balance_by={balance_by}")
    for (cls, strat), trials in sorted(cells.items()):
        print(f" class={cls} stratum={strat}: {len(trials)} trials")
    print(f" subsampling each cell to {n_per_cell}")
    
    # build final index list and labels
    chosen_idx = []
    labels = []
    for cls in unique_classes:
        for (c, strat), trials in cells.items():
            if c != cls:
                continue
            pick = rng.choice(trials, size=n_per_cell, replace=False)
            chosen_idx.extend(pick.tolist())
            labels.extend([cats[(cls,)]] * n_per_cell)
            
    chosen_idx = np.array(chosen_idx)
    labels = np.array(labels, dtype=int)
    
    # pull the data
    data = epochs.get_data(copy=True)[chosen_idx]
    return data, labels, cats
    
# ---- Quick test ----
if __name__ == "__main__":
    test_name = "Stimulus/i25.0/r25.0/BigLetters/SmallLetterh/Taskg/TargetLetters/Responded1.0/ParticipantResponse115.0/CorrectResponse115.0/TrialCount261.0/BlockTrialCount37.0/ReactionTime1350.0/Accuracy1.0/D57"
    result = parse_event_name(test_name)
    for k, v in result.items():
        print(f"  {k:30s} = {v}")