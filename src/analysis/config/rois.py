"""Bilateral coarse-ROI definitions built from Destrieux parcel names.

These entries are *selectors*, not replacement anatomical labels.  They omit
FreeSurfer's ``ctx_lh_`` / ``ctx_rh_`` prefix deliberately so one definition
selects the homologous parcel in either hemisphere.  Electrode tables retain
the complete atlas label (and the brain renderer uses electrode coordinates),
so this shorthand does not discard or alter laterality.
"""

rois_dict = {
    'dlpfc': ["G_front_middle", "G_front_sup", "S_front_inf", "S_front_middle", "S_front_sup"],
    'acc': ["G_and_S_cingul-Ant", "G_and_S_cingul-Mid-Ant"],
    'parietal': ["G_parietal_sup", "S_intrapariet_and_P_trans", "G_pariet_inf-Angular", "G_pariet_inf-Supramar"],
    'lpfc': ["G_front_inf-Opercular", "G_front_inf-Orbital", "G_front_inf-Triangul", "G_front_middle", "G_front_sup", "Lat_Fis-ant-Horizont", "Lat_Fis-ant-Vertical", "S_circular_insula_ant", "S_circular_insula_sup", "S_front_inf", "S_front_middle", "S_front_sup"],
    'v1': ["G_oc-temp_med-Lingual", "S_calcarine", "G_cuneus"],
    'occ': ["G_cuneus", "G_and_S_occipital_inf", "G_occipital_middle", "G_occipital_sup", "G_oc-temp_lat-fusifor", "G_oc-temp_med-Lingual", "Pole_occipital", "S_calcarine", "S_oc_middle_and_Lunatus", "S_oc_sup_and_transversal", "S_occipital_ant"]
}


def select_rois(value):
    """Return ROI definitions selected by a comma-separated name list.

    ``None`` and the string ``"all"`` disable ROI filtering.  The helper is
    intentionally strict so a misspelled ROI cannot silently produce an empty
    electrode set in a long-running cluster job.
    """
    if value is None or (isinstance(value, str) and value.strip().lower() == 'all'):
        return None

    names = ([part.strip() for part in value.split(',')]
             if isinstance(value, str) else list(value))
    names = [name for name in names if name]
    unknown = [name for name in names if name not in rois_dict]
    if unknown:
        raise ValueError(
            f"Unknown ROI(s): {unknown}. Available ROIs: {sorted(rois_dict)}")
    if not names:
        raise ValueError("ROIS must contain at least one ROI name, or 'all'")
    return {name: rois_dict[name] for name in names}
