import pytest

from src.analysis.config.rois import rois_dict, select_rois


def test_select_rois_supports_single_comma_separated_and_all():
    assert select_rois('lpfc') == {'lpfc': rois_dict['lpfc']}
    assert list(select_rois('lpfc,occ')) == ['lpfc', 'occ']
    assert select_rois('all') is None


def test_select_rois_rejects_unknown_or_empty_names():
    with pytest.raises(ValueError, match='Unknown ROI'):
        select_rois('lpfc,not-a-roi')
    with pytest.raises(ValueError, match='at least one'):
        select_rois(' , ')
