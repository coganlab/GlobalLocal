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

    if save_dir is None:
        raise ValueError("save_dir must be specified to save the dictionary.")

    if LAB_root is None:
        HOME = os.path.expanduser("~")
        if os.name == 'nt':  # Windows
            LAB_root = os.path.join(HOME, "Box", "CoganLab")
        else:  # macOS/Linux
            LAB_root = os.path.join(HOME,
                                     "Library",
                                     "CloudStorage",
                                     "Box-Box",
                                     "CoganLab")

    for sub in subjects:
        print(f"Processing subject: {sub}")
        layout = get_data(task, root=LAB_root)
        filt = raw_from_layout(
            layout.derivatives['derivatives/clean'],
            subject=sub,
            extension='.edf',
            desc='clean',
            preload=False
        )

        good = crop_empty_data(filt)
        good.info['bads'] = channel_outlier_marker(good, 3, 2)

        if 'Trigger' in good.ch_names:
            good.drop_channels(['Trigger'])

        filt.drop_channels(good.info['bads'])
        good.drop_channels(good.info['bads'])

        good.load_data()
        ch_type = filt.get_channel_types(only_data_chs=True)[0]
        good.set_eeg_reference(ref_channels='average', ch_type=ch_type)

        if sub == 'D0107A':
            default_dict = gen_labels(good.info, sub='D107A')
        else:
            default_dict = gen_labels(good.info)

        rawROI_dict = defaultdict(list)
        for elec, roi in default_dict.items():
            rawROI_dict[roi].append(elec)

        filtROI_dict = {
            roi: eles
            for roi, eles in rawROI_dict.items()
            if 'White-Matter' not in roi
        }

        subjects_electrodes_to_ROIs_dict[sub] = {
            'default_dict': dict(default_dict),
            'rawROI_dict'    : dict(rawROI_dict),
            'filtROI_dict'   : filtROI_dict
        }

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    with open(save_path, 'w') as f:
        json.dump(subjects_electrodes_to_ROIs_dict, f, indent=4, ensure_ascii=False)
    print(f"Saved combined mapping to {save_path}")

    return subjects_electrodes_to_ROIs_dict


if __name__ == '__main__':
    subjects = ['D0057','D0059', 'D0063', 'D0065', 'D0069', 'D0071', 'D0077', 'D0090', 'D0094', 'D0100', 'D0102', 'D0103', 'D0107A', 'D0110', 'D0116', 'D0117', 'D0121']
    task = 'GlobalLocal'
    LAB_root = None
    save_dir = './output'

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

    print("All subject-specific files generated.")
