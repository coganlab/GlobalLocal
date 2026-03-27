import argparse
import logging
import sys
import os
from mne_bids import BIDSPath
import nibabel as nib
import numpy as np
import pandas as pd

# Simple logging: everything INFO and above to stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

def main(
    bids_root: str,
    recon_dir: str,
    subject: str,
    ref: str = 'car',
    radius_mm: float = 10.0,
    **kwargs
):

    """Parcellate iEEG electrodes to atlas labels using a MATLAB-compatible
    coordinate pipeline.

    Parameters
    ----------
    bids_root : str
        Root of the BIDS dataset.
    recon_dir : str
        Root of the FreeSurfer / ECoG_Recon directory (per-subject mr/atlas).
    subject : str
        BIDS subject ID, e.g. "D0019".
    ref : {"car", "bipolar"}
        Reference type. This decides which BIDS derivatives folder we use
        to read the electrodes.tsv ("clean" for car, "bipolar" for bipolar).
    radius_mm : float
        Radius of the spherical neighborhood (in mm/voxels) for majority
        voting around each electrode.
    """

    # -------------------------------------------------------------------------
    # 1. Load electrode coordinates from BIDS derivatives
    # -------------------------------------------------------------------------
    # We read the MNE-exported electrodes.tsv from BIDS derivatives. These
    # coordinates are in the same RAS space as the validation CSVs used to
    # match the MATLAB pipeline.
    description = 'bipolar' if ref == 'bipolar' else 'clean'
    elec_path = BIDSPath(
        root=os.path.join(bids_root, 'derivatives', description),
        subject=subject,
        suffix='electrodes',
        extension='.tsv',
        datatype='ieeg'
    ).match()
    
    logger.info(f"Found electrode files: {elec_path}")
    
    try:
        elec_tsv = pd.read_csv(elec_path[0], sep='\t')
        # insert subject column at the beginning
        elec_tsv.insert(0, 'subject', subject)
        # remove size column if it exists
        elec_tsv = elec_tsv.drop(columns=['size'], errors='ignore')
        # drop rows where x, y, z are NaN
        elec_tsv = elec_tsv.dropna(subset=['x', 'y', 'z']).reset_index(drop=True)
        logger.info(f"Loaded {len(elec_tsv)} electrodes from BIDS electrodes.tsv")
    except FileNotFoundError:
        raise FileNotFoundError(f"Electrodes.tsv not found for subject {subject}")
    
    # subject_id = subject.replace('00', '')
    # Convert BIDS subject ID (e.g. "D0019") to the FreeSurfer/ECoG_Recon ID
    # (e.g. "D19"). This mirrors how the MATLAB pipeline organizes subjects.
    subject_id = f"D{int(subject[1:])}"

    # -------------------------------------------------------------------------
    # 2. Load subject-specific atlas volume (aparc.a2009s+aseg.mgz)
    # -------------------------------------------------------------------------
    # We use the aparc.a2009s+aseg.mgz atlas, exactly as in the MATLAB
    # parcellation. This volume is conformed to 256x256x256 1mm isotropic
    # FreeSurfer space.
    subj_dir = os.path.join(recon_dir, subject_id)
    atlas_path = os.path.join(subj_dir, 'mri', 'aparc.a2009s+aseg.mgz')

    logger.info(f"Using atlas for subject {subject}: {atlas_path}")
    try:
        atlas_img = nib.load(atlas_path)
    except Exception as e:
        raise Exception(f"Atlas not found or not readable for subject {subject}: {atlas_path}") from e
    # Load atlas volume and log shape
    atlas_data = atlas_img.get_fdata().astype('int32')
    logger.info(f"Atlas loaded. shape={atlas_data.shape}")
    
    names = elec_tsv["name"].values
    
    # -------------------------------------------------------------------------
    # 3. Load FreeSurfer LUT (ID -> label name)
    # -------------------------------------------------------------------------
    # The LUT maps integer atlas IDs to human-readable label strings, e.g.
    #   1002 -> ctx_lh_G_front_inf-Opercular
    # This is necessary both for logging and for matching MATLAB labels.
    id_to_name = {}
    lut_path = os.path.join(bids_root, "code", "FreeSurferColorLUT.txt")
    if os.path.isfile(lut_path):
        try:
            with open(lut_path, 'r') as f:
                for line in f:
                    s = line.strip()
                    if not s or s.startswith('#'):
                        continue
                    parts = s.split()
                    if len(parts) >= 2 and parts[0].isdigit():
                        id_to_name[int(parts[0])] = parts[1]
            logger.info(f"Loaded LUT from {lut_path} with {len(id_to_name)} entries")
        except Exception as e:
            logger.info(f"Failed to load LUT {lut_path}: {e}")
    
    # -------------------------------------------------------------------------
    # 4. Load ROI mapping (atlas_label -> gross_label)
    # -------------------------------------------------------------------------
    # We optionally map fine-grained atlas labels to a smaller set of
    # "gross" ROIs using a2009s.csv, e.g.:
    #   ctx_lh_G_front_inf-Opercular -> IFG (L)
    # If the mapping file is missing, we simply fall back to using atlas
    # labels directly as ROI labels.
    atlas_to_gross = {}
    mapping_path = os.path.join(bids_root, "code", "a2009s.csv")
    if os.path.isfile(mapping_path):
        try:
            mapping_df = pd.read_csv(mapping_path)
            # Create dictionary: atlas_label -> gross_label
            atlas_to_gross = dict(zip(mapping_df['atlas_label'], mapping_df['gross_label']))
            logger.info(f"Loaded ROI mapping from {mapping_path} with {len(atlas_to_gross)} entries")
        except Exception as e:
            logger.info(f"Failed to load mapping {mapping_path}: {e}")

    # -------------------------------------------------------------------------
    # 5. Pre-compute sphere radius and bad labels
    # -------------------------------------------------------------------------
    # The atlas is a 256x256x256 conformed FreeSurfer volume with 1mm
    # isotropic voxels, so a radius in mm is equal to a radius in voxels.
    r_vox = int(np.ceil(radius_mm))
    
    # "Bad" labels that should be avoided when picking the final label.
    # IMPORTANT: This list MUST match the behavior of gen_labels/_pick_label
    # in the original MATLAB/Python code:
    #   - Center label is checked with EXACT string equality
    #   - Voted labels are checked with SUBSTRING matching
    # Only these four words are considered bad for the center; e.g.
    # "Right-Cerebral-White-Matter" is *not* in this tuple and therefore
    # treated as "good" for the center label, which is crucial to exactly
    # reproduce the MATLAB validation CSVs.
    bad_words = ('Unknown', 'unknown', 'hypointensities', 'White-Matter')
    for idx, row in elec_tsv.iterrows():
        ras = np.array([row['x'], row['y'], row['z']], dtype=float)

        # ---------------------------------------------------------------------
        # 6. RAS (mm) -> Atlas voxel indices (i, j, k)
        # ---------------------------------------------------------------------
        # The electrodes.tsv coordinates are in RAS (Right-Anterior-Superior)
        # in millimeters, in the same space as the validation CSVs. The
        # original MATLAB code converts these to FreeSurfer tkr/PIALVOX
        # coordinates and then to voxel indices using a series of axis swaps
        # and flips.
        #
        # Here we use the *combined* closed-form mapping that we derived by
        # reverse-engineering the MATLAB pipeline and validating against the
        # aparc.a2009s+aseg volume:
        #   MATLAB indices (1-based):
        #       mi = round(128 - ras[2])
        #       mj = round(128 - ras[0])
        #       mk = 126 + round(ras[1])
        #   NumPy indexing (0-based, row-major):
        #       atlas_data[mj, mi, mk]
        #
        # After considering MATLAB's column-major ordering vs NumPy's
        # row-major ordering, we end up with the following direct formulas
        # for NumPy indices (i, j, k):

        # Adaptive unit conversion: if coordinates look like meters (|value|
        # < 10), convert to mm. This is important because some BIDS inputs
        # store positions in meters, while the atlas and MATLAB pipeline use
        # millimeters.
        if np.max(np.abs(ras)) < 10:
            ras = ras * 1000
            
        # RAS -> atlas voxel index. These constants (128, 126) and axis
        # order mirror the MATLAB implementation and FreeSurfer tkr space.
        i = int(np.round(128 - ras[0]))  # X -> first index (with flip)
        j = int(np.round(128 - ras[2]))  # Z -> second index (with flip)
        k = int(126 + np.round(ras[1]))  # Y -> third index (offset by 126)
        
        # Clip to atlas bounds
        i = int(np.clip(i, 0, atlas_data.shape[0] - 1))
        j = int(np.clip(j, 0, atlas_data.shape[1] - 1))
        k = int(np.clip(k, 0, atlas_data.shape[2] - 1))
        
        # Look up the atlas label at this center voxel and convert the integer
        # ID to a human-readable name via the LUT.
        label_id = int(atlas_data[i, j, k])
        nm = names[idx] if idx < len(names) else str(idx)
        center_name = id_to_name.get(label_id, str(label_id)) if id_to_name else str(label_id)
        
        # Sanity log for center voxel (very useful when debugging mismatches)
        logger.info(f"{nm}: ras_mm={ras}, ijk=({i},{j},{k}), center={center_name}")

        # ---------------------------------------------------------------------
        # 7. Spherical neighborhood majority vote in voxel (IJK) space
        # ---------------------------------------------------------------------
        # We gather all voxels within a sphere of radius `radius_mm` (in
        # practice, voxels) around the center (i, j, k). We then examine the
        # distribution of atlas labels within this sphere and apply the
        # gen_labels/_pick_label logic to choose the final label.
        i0 = max(i - r_vox, 0)
        i1 = min(i + r_vox, atlas_data.shape[0] - 1)
        j0 = max(j - r_vox, 0)
        j1 = min(j + r_vox, atlas_data.shape[1] - 1)
        k0 = max(k - r_vox, 0)
        k1 = min(k + r_vox, atlas_data.shape[2] - 1)

        ii, jj, kk = np.mgrid[i0:i1+1, j0:j1+1, k0:k1+1]
        
        # Euclidean distance in voxel space (1mm isotropic)
        dist = np.sqrt((ii - i)**2 + (jj - j)**2 + (kk - k)**2)
        mask = dist <= radius_mm
        
        # Extract labels within the sphere
        valid_labels = atlas_data[i0:i1+1, j0:j1+1, k0:k1+1][mask].astype(int)

        if valid_labels.size == 0:
            # Edge case: sphere is empty (should be rare unless very near
            # volume borders). In this case we fall back to the center label
            # with fraction 1.0.
            logger.info(f"{nm}: no voxels within {radius_mm}mm sphere; using center label")
            primary_name = center_name
            frac = 1.0
        else:
            # Get the label distribution within the sphere and sort by
            # descending frequency.
            vals, counts = np.unique(valid_labels, return_counts=True)
            sorted_idx = np.argsort(-counts)  # Sort by descending count
            
            # -----------------------------------------------------------------
            # 8. _pick_label logic (faithful reimplementation)
            # -----------------------------------------------------------------
            # We mimic the behavior of gen_labels/_pick_label from the
            # original pipeline:
            #   1) If the *center* label is NOT a bad word (EXACT match),
            #      we immediately return the center label.
            #   2) Otherwise, we iterate over the sphere vote labels (in
            #      order of frequency) and pick the first label that:
            #         - does NOT contain any bad_word as a SUBSTRING, and
            #         - has fraction > 5% of the sphere.
            #   3) If no such label is found, we fall back to the center
            #      label even if it is a bad label.
            #
            # Critically, the center label uses EXACT membership in
            # `bad_words`, while the votes use substring matching. This
            # asymmetry is what allows labels like "Right-Cerebral-White-Matter"
            # to be accepted as a center label but rejected as a vote label,
            # matching the MATLAB behavior exactly.
            percent_thresh = 0.05
            center_is_bad = center_name in bad_words  # EXACT match for center!
            
            if not center_is_bad:
                # Center label is "good": use it directly.
                primary_name = center_name
                # Also compute the fraction of the sphere occupied by the
                # center label for completeness (this is not used in the
                # decision logic, but is informative for downstream use).
                center_label_id = int(atlas_data[i, j, k])
                frac = float(np.sum(valid_labels == center_label_id) / len(valid_labels))
            else:
                # Center is "bad": search the voted labels in order of
                # descending frequency for the first acceptable label.
                primary_name = None
                frac = 0.0
                
                for sort_i in sorted_idx:
                    label_id = int(vals[sort_i])
                    label_name = id_to_name.get(label_id, str(label_id)) if id_to_name else str(label_id)
                    label_frac = counts[sort_i] / len(valid_labels)
                    
                    # We reject voted labels that *contain* any bad word
                    # as a substring (different from the exact-match check
                    # used for the center label above), and we require that
                    # the label occupies at least 5% of the sphere.
                    if not any(bad_word in label_name for bad_word in bad_words) and label_frac > percent_thresh:
                        primary_name = label_name
                        frac = label_frac
                        break

                # If no acceptable vote label is found, fall back to the
                # (possibly bad) center label, again computing its fraction.
                if primary_name is None:
                    primary_name = center_name
                    center_label_id = int(atlas_data[i, j, k])
                    frac = float(np.sum(valid_labels == center_label_id) / len(valid_labels))
            
        logger.info(f"{nm}: VOTE center={center_name}, vote={primary_name}, fraction={frac:.3f}")
        
        # Store both the original center label and the final chosen label
        # (primary_name) plus the fraction of the sphere occupied by this
        # label. This mirrors the information in the MATLAB-generated CSVs.
        elec_tsv.at[idx, 'center'] = center_name
        elec_tsv.at[idx, 'label'] = primary_name
        elec_tsv.at[idx, 'fraction'] = frac
        
        # Store top 3 labels and their fractions
        top_labels = ['', '', '']
        top_fractions = [0.0, 0.0, 0.0]
        
        if valid_labels.size > 0:
            # Get the label distribution within the sphere and sort by
            # descending frequency.
            vals, counts = np.unique(valid_labels, return_counts=True)
            sorted_idx = np.argsort(-counts)  # Sort by descending count
            
            # Prepare top 3 labels and fractions
            for rank in range(min(3, len(sorted_idx))):
                label_id = int(vals[sorted_idx[rank]])
                label_name = id_to_name.get(label_id, str(label_id)) if id_to_name else str(label_id)
                label_frac = counts[sorted_idx[rank]] / len(valid_labels)
                top_labels[rank] = label_name
                top_fractions[rank] = label_frac
        
        # Store 2nd and 3rd labels and fractions
        elec_tsv.at[idx, '2nd label'] = top_labels[1]
        elec_tsv.at[idx, '2nd fraction'] = top_fractions[1]
        elec_tsv.at[idx, '3rd label'] = top_labels[2]
        elec_tsv.at[idx, '3rd fraction'] = top_fractions[2]
        
        # -----------------------------------------------------------------
        # 9. Determine final ROI label with intersection detection
        # -----------------------------------------------------------------
        # Logic:
        # 1. Default to Unknown if top1 fraction < 0.3
        # 2. Keep top1 if 2nd/3rd are Unknown or contain White-Matter
        # 3. Mark as Intersection if 2nd/3rd have different ROI and fraction > 0.15
        
        def get_roi_name(label):
            """Get ROI name from atlas label using mapping."""
            roi_full = atlas_to_gross.get(label, label)
            return roi_full.split('(')[0].strip() if '(' in roi_full else roi_full
        
        def is_ignorable(label):
            """Check if label is Unknown or White-Matter (should be ignored for intersection)."""
            return 'Unknown' in label or 'White-Matter' in label or label == ''
        
        # Get ROI for top1
        top1_roi = get_roi_name(primary_name)
        
        # Step 1: Default to Unknown if top1 fraction < 0.3
        if frac < 0.3:
            final_label = 'Unknown'
            final_roi = 'Unknown'
        else:
            final_label = primary_name
            final_roi = top1_roi
            
            # Step 2 & 3: Check 2nd and 3rd labels for intersection
            is_intersection = False
            for rank in [1, 2]:  # Check 2nd and 3rd
                other_label = top_labels[rank]
                other_frac = top_fractions[rank]
                
                # Skip if ignorable (Unknown or White-Matter)
                if is_ignorable(other_label):
                    continue
                
                # Get ROI for this label
                other_roi = get_roi_name(other_label)
                
                # Check if different ROI and fraction > 0.15
                if other_roi != top1_roi and other_frac > 0.15:
                    is_intersection = True
                    break
            
            if is_intersection:
                final_label = 'Intersection'
                final_roi = 'Intersection'
        
        # Update label and roi columns
        elec_tsv.at[idx, 'label'] = final_label
        elec_tsv.at[idx, 'roi'] = final_roi
        
        # Determine hemisphere from primary_name (even if marked as Intersection)
        roi_label_full = atlas_to_gross.get(primary_name, primary_name)
        hemi = roi_label_full.split('(')[1].split(')')[0] if '(' in roi_label_full and ')' in roi_label_full else 'unknown'
        elec_tsv.at[idx, 'hemi'] = hemi
        
        logger.info(f"{nm}: final_label={final_label}, final_roi={final_roi}")
    
    # After all electrodes are processed, if ref is car, prefix all
    # electrode names with the BIDS subject ID using a vectorized map.
    if ref == 'car':
        elec_tsv['name'] = elec_tsv['name'].map(lambda n: f"{subject}_{n}")

    # ---------------------------------------------------------------------
    # 10. Save parcellation results back to BIDS derivatives
    # ---------------------------------------------------------------------
    # We save to:
    #   derivatives/parcellation/sub-<subject>/<ref>/
    #       sub-<subject>_proc-<radius>mm_aparc2009s.csv
    # This mirrors the structure used by the validation CSVs so that
    # downstream analysis and comparisons are straightforward.
    save_path = BIDSPath(
        root=os.path.join(bids_root, 'derivatives', 'parcellation'),
        subject=subject,
        suffix='aparc2009s',
        datatype=ref,
        processing=f'{int(radius_mm)}mm',
        extension='.csv',
        check=False
    )
    save_path.mkdir(exist_ok=True)
    elec_tsv.to_csv(save_path, index=False)
    logger.info(f"Saved parcellated electrodes to {save_path}")
    
    print(elec_tsv)
    return

if __name__ == "__main__":
    # ---------------------------------------------------------------------
    # Command-line interface
    # ---------------------------------------------------------------------
    # This allows the script to be called directly, e.g.:
    #
    #   python parcellation.py \
    #       --subject D0019 \
    #       --radius_mm 10 \
    #       --ref car
    #
    # so that we can easily test subjects and compare against the MATLAB
    # validation CSVs.
    parser = argparse.ArgumentParser()
    parser.add_argument("--bids_root", default="/cwork/ns458/BIDS-1.0_LexicalDecRepDelay/BIDS", type=str,
                        help="Root directory of the BIDS dataset.")
    parser.add_argument("--recon_dir", default='/cwork/ns458/ECoG_Recon/', type=str,
                        help="Root directory containing FreeSurfer/ECoG_Recon subjects.")
    parser.add_argument("--ref", type=str, default='bipolar', choices=['car', 'bipolar'],
                        help="Reference type: 'car' (monopolar) or 'bipolar'.")
    parser.add_argument("--radius_mm", type=float, default=3.0,
                        help="Sphere radius in mm/voxels for neighborhood voting.")
    parser.add_argument("--subject", type=str, default='D0107',
                        help="BIDS subject ID, e.g. 'D0019'.")

    _args = parser.parse_args()
    main(**vars(_args))
