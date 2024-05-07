# %%
import os
import pandas as pd
import numpy as np

def load_dataframes(directory):
    """Load all CSV files from the specified directory into pandas DataFrames."""
    dataframes = []
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            df = pd.read_csv(file_path)
            dataframes.append(df)
    return dataframes

def combine_dataframes(dataframes):
    """Combine a list of DataFrames into a single DataFrame, ignoring index."""
    return pd.concat(dataframes, ignore_index=True)

def format_subject_ids(df, subject_id_col='subject_ID'):
    """Ensure all subject IDs are formatted as 'D' followed by four zero-padded digits."""
    def format_subject_id(subject_id):
        subject_id_str = str(subject_id)
        if not subject_id_str.startswith('D'):
            subject_id_str = 'D' + subject_id_str
        parts = subject_id_str.split('D')
        return 'D' + parts[1].zfill(4)
    
    df[subject_id_col] = df[subject_id_col].apply(format_subject_id)
    return df

def save_accuracy_arrays(df, save_directory, subject_col='subject_ID', acc_col='acc', trial_col='trialCount'):
    """Save accuracy data as .npy files for each unique subject."""
    unique_subjects = df[subject_col].unique()
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    for subject in unique_subjects:
        subject_df = df[df[subject_col] == subject].sort_values(by=trial_col)
        accuracy_array = subject_df[acc_col].values
        np.save(os.path.join(save_directory, f'{subject}_accuracy.npy'), accuracy_array)

def main():
    rawDataFolder = r'C:\Users\jz421\Box\CoganLab\D_Data\GlobalLocal\rawDataCopies'
    save_dir = r'C:\Users\jz421\Box\CoganLab\D_Data\GlobalLocal\accArrays'
    combined_csv_path = r'C:\Users\jz421\Box\CoganLab\D_Data\GlobalLocal\combinedData.csv'

    # Load, combine, and format data
    dfs = load_dataframes(rawDataFolder)
    combined_df = combine_dataframes(dfs)
    combined_df = format_subject_ids(combined_df)

    # Save the combined DataFrame to CSV
    combined_df.to_csv(combined_csv_path, index=False)

    # Save accuracy arrays
    save_accuracy_arrays(combined_df, save_dir)

if __name__ == '__main__':
    main()


