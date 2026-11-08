import numpy as np
import pandas as pd
import pytest

from mosa_disentangle import data


def test_emt_score_sign():
    rng = np.random.default_rng(0)
    genes = data.EPITHELIAL_GENES + data.MESENCHYMAL_GENES + ["OTHER"]
    expr = pd.DataFrame(rng.normal(size=(20, len(genes))), columns=genes)
    expr.loc[0, data.MESENCHYMAL_GENES] += 3.0
    expr.loc[1, data.EPITHELIAL_GENES] += 3.0
    score = data.emt_score(expr)
    assert score.iloc[0] > score.iloc[2:].max()
    assert score.iloc[1] < score.iloc[2:].min()


def test_emt_score_requires_markers():
    expr = pd.DataFrame(np.ones((5, 2)), columns=["A", "B"])
    with pytest.raises(ValueError):
        data.emt_score(expr)


def test_collapse_rare():
    labels = pd.Series(["Lung"] * 12 + ["Eye"] * 2)
    collapsed = data.collapse_rare(labels, min_size=10)
    assert set(collapsed) == {"Lung", "Other"}


def test_filter_features_drops_missing_and_constant():
    frame = pd.DataFrame(
        {"keep": [1.0, 2.0, 3.0, 4.0], "const": 1.0, "missing": [np.nan] * 3 + [1.0]}
    )
    out = data.filter_features(frame, "proteomics", max_missing=0.5)
    assert list(out.columns) == ["keep"]
    assert (
        data.filter_features(frame[["keep"]], "transcriptomics", top_variance=1).shape[
            1
        ]
        == 1
    )


def test_paired_rows_partner_alignment():
    proteins = ["P1", "P2"]
    sanger = pd.DataFrame(
        [[1.0, 2.0], [3.0, 4.0]], index=["SIDM1", "SIDM2"], columns=proteins
    )
    ccle = pd.DataFrame(
        [[5.0, 6.0], [7.0, 8.0]], index=["SIDM2", "SIDM3"], columns=proteins
    )
    observed, partner, obs = data._paired_rows(sanger, ccle, proteins)
    assert list(observed.index) == ["SIDM1", "SIDM2", "SIDM2__ccle", "SIDM3__ccle"]
    assert partner.loc["SIDM2__ccle"].tolist() == [3.0, 4.0]
    assert partner.loc["SIDM2"].tolist() == [5.0, 6.0]
    assert partner.loc["SIDM1"].isna().all()
    assert obs["has_partner"].tolist() == [False, True, True, False]
    assert obs["model_type"].tolist() == ["sanger", "sanger", "ccle", "ccle"]


def test_haematopoietic_lines_have_no_emt():
    tissue = pd.Series(["Lung", "Haematopoietic and Lymphoid"], index=["a", "b"])
    score = pd.Series([1.0, 2.0], index=tissue.index)
    masked = data.mask_non_epithelial(score, tissue)
    assert masked.isna().tolist() == [False, True]
    assert masked["a"] == 1.0
