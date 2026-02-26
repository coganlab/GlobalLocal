import sys
import os

try:
    current_file_path = os.path.abspath(__file__)
    current_script_dir = os.path.dirname(current_file_path)
except NameError:
    current_script_dir = os.getcwd()

project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..', '..'))

if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def run_aggregate_and_plot_time_averaged_cms(
    time_averaged_cms_list,
    condition_comparisons,
    rois,
    cats_by_roi,
    args,
    save_dir,
):
    
    # --- Step 1: Aggregate and Plot Time-Averaged CMs ---
    print("\n📊 Aggregating and plotting time-averaged confusion matrices...")
    for condition_comparison in condition_comparisons.keys():
        for roi in rois:
            # Collect all raw CMs for this specific condition/ROI
            raw_cms = [
                boot_result[condition_comparison][roi] 
                for boot_result in time_averaged_cms_list 
                if condition_comparison in boot_result and roi in boot_result[condition_comparison]
            ]

            if not raw_cms:
                continue

            # Sum, normalize, and plot (same logic as before)
            total_cm_counts = np.sum(np.array(raw_cms), axis=0)
            row_sums = total_cm_counts.sum(axis=1)[:, np.newaxis]
            row_sums[row_sums == 0] = 1 
            normalized_cm = total_cm_counts.astype('float') / row_sums
            
            ## FIX: This check now correctly uses the `cats_by_roi` dictionary retrieved from the bootstrap results.
            if roi in cats_by_roi:
                display_labels = [str(key) for key in cats_by_roi[roi].keys()]
            else:
                print(f"Warning: 'cats' dictionary not found for ROI {roi}. Skipping CM plot.")
                continue

            # Plotting logic
            fig, ax = plt.subplots()
            disp = ConfusionMatrixDisplay(confusion_matrix=normalized_cm, display_labels=display_labels)
            disp.plot(ax=ax, im_kw={"vmin": 0, "vmax": 1}, colorbar=True)
            ax.set_title(f'{roi} - {condition_comparison}\n(Counts summed across {args.bootstraps} bootstraps)')

            filename = (
                f'{args.timestamp}_{roi}_{condition_comparison}_SUMMED_{args.bootstraps}boots_ev_{args.explained_variance}'
                f'time_averaged_confusion_matrix.png'
            )
            plot_save_path = os.path.join(save_dir, condition_comparison, roi)
            os.makedirs(plot_save_path, exist_ok=True)
            plt.savefig(os.path.join(plot_save_path, filename))
            plt.close()
            print(f"✅ Saved summed & normalized CM for {roi} to {plot_save_path}")