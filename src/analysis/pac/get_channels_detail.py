import os
import json
import numpy as np
import pandas as pd
import mne
from collections import OrderedDict, defaultdict
import matplotlib.pyplot as plt

# iEEG 相关工具函数
from ieeg.navigate import channel_outlier_marker, trial_ieeg, crop_empty_data, outliers_to_nan
from ieeg.io import raw_from_layout, get_data
from ieeg.timefreq.utils import crop_pad
from ieeg.timefreq import gamma
from ieeg.calc.scaling import rescale
from ieeg.calc.stats import time_perm_cluster, window_averaged_shuffle
from ieeg.viz.mri import gen_labels

# 统计建模
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests

# Python 脚本功能：生成每个被试电极到 ROI 的映射，并保存为 JSON 文件

def make_subjects_electrodes_to_ROIs_dict(subjects,
                                          task='GlobalLocal',
                                          LAB_root=None,
                                          save_dir=None,
                                          filename='subjects_electrodes_to_ROIs_dict.json'):
    """
    Creates mappings for each electrode to its corresponding Region of Interest (ROI)
    for a list of subjects and saves these mappings to a JSON file.
    """
    subjects_electrodes_to_ROIs_dict = {}

    # 检查 save_dir
    if save_dir is None:
        raise ValueError("save_dir must be specified to save the dictionary.")

    
    if LAB_root is None:
        raise ValueError("LAB_root must be specified and point to a BIDS dataset root.")
    print(f"Using LAB_root: {LAB_root}")

   
    for sub in subjects:
        print(f"\n---\nStarting subject: {sub}")
       
        try:
            layout = get_data(task, root=LAB_root)
            print(f"Loaded BIDS layout for task '{task}' successfully.")
        except Exception as e:
            print(f"Error loading BIDS layout: {e}")
            continue

        try:
            filt = raw_from_layout(
                layout.derivatives['derivatives/clean'],
                subject=sub,
                extension='.edf',
                desc='clean',
                preload=False
            )
            print("Loaded raw clean data successfully.")
        except Exception as e:
            print(f"Error loading raw clean data for subject {sub}: {e}")
            continue

    
        good = crop_empty_data(filt)
        bads = channel_outlier_marker(good, 3, 2)
        good.info['bads'] = bads
        print(f"Identified bad channels: {bads}")

    
        if 'Trigger' in good.ch_names:
            good.drop_channels(['Trigger'])
            print("Dropped 'Trigger' channel.")

     
        filt.drop_channels(bads)
        good.drop_channels(bads)
        print("Dropped bad channels from both filt and good objects.")

        
        good.load_data()
        ch_type = filt.get_channel_types(only_data_chs=True)[0]
        good.set_eeg_reference(ref_channels='average', ch_type=ch_type)
        print("Data loaded and EEG average reference set.")

        
        try:
            if sub == 'D0107A':
                default_dict = gen_labels(good.info, sub='D107A')
            else:
                default_dict = gen_labels(good.info)
            print(f"Generated electrode-to-ROI labels, total electrodes: {len(default_dict)}")
        except Exception as e:
            print(f"Error generating labels for subject {sub}: {e}")
            continue

        
        rawROI_dict = defaultdict(list)
        for elec, roi in default_dict.items():
            rawROI_dict[roi].append(elec)
        print(f"Built raw ROI dictionary with {len(rawROI_dict)} regions.")

        
        filtROI_dict = {roi: eles for roi, eles in rawROI_dict.items() if 'White-Matter' not in roi}
        print(f"Filtered ROI dictionary to {len(filtROI_dict)} regions (excluding 'White-Matter').")

        subjects_electrodes_to_ROIs_dict[sub] = {
            'default_dict': dict(default_dict),
            'rawROI_dict': dict(rawROI_dict),
            'filtROI_dict': filtROI_dict
        }


    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    with open(save_path, 'w') as f:
        json.dump(subjects_electrodes_to_ROIs_dict, f, indent=4, ensure_ascii=False)
    print(f"\nSaved combined mapping to {save_path}")

    return subjects_electrodes_to_ROIs_dict


if __name__ == '__main__':
    subjects = ['D0057']  
    task = 'GlobalLocal'
    LAB_root = '/cwork/rl330' 
    save_dir = '/hpc/home/rl330/coganlab/rl330/GlobalLocal/src/analysis/pac'

    all_subjects = make_subjects_electrodes_to_ROIs_dict(
        subjects=subjects,
        task=task,
        LAB_root=LAB_root,
        save_dir=save_dir
    )

    per_sub_dir = os.path.join(save_dir, 'per_subject')
    os.makedirs(per_sub_dir, exist_ok=True)

    for sub, mapping in all_subjects.items():
        filepath = os.path.join(per_sub_dir, f"{sub}_electrodes_to_ROIs.json")
        with open(filepath, 'w') as f:
            json.dump({sub: mapping}, f, indent=4, ensure_ascii=False)
        print(f"Saved {sub} mapping to {filepath}")

    print("\nAll subject-specific files generated.")
