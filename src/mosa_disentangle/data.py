"""Build MOSA-compatible datasets from the public MOSA cell line tables.

Two datasets are produced from the figshare deposit of Cai et al. (2024):

* ``cell_lines``: the multi-omic cell line cohort (transcriptomics,
  proteomics, methylation, metabolomics, drug response, CRISPR-Cas9, copy
  number) with tissue, EMT scores, MSI status and driver mutations in ``.obs``.
  All samples are cell lines, so the technical factor is added later by
  ``batch_sim`` with a known ground truth.
* ``proteomics_paired``: one view with two real laboratory sources (Sanger
  and CCLE proteomics); 290 cell lines were measured by both, which gives a
  real paired ground truth for cross-laboratory counterfactual decoding.

This module also holds the MuData read/write helpers shared by the pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
from mosa.data.dataset import MultiOmicDataset
from mosa.data.io import load_mudata
from mosa.utils import mudata_set_options
from mudata import MuData

logger = logging.getLogger(__name__)

MASK_LAYER = "mask"

# Tables are features x samples unless listed in SAMPLES_BY_FEATURES.
VIEW_FILES = {
    "transcriptomics": "transcriptomics.csv",
    "proteomics": "proteomics.csv",
    "methylation": "methylation.csv",
    "metabolomics": "metabolomics.csv",
    "drugresponse": "drugresponse.csv",
    "crisprcas9": "crisprcas9.csv",
    "copynumber": "cnv_summary_20230303_matrix.csv",
}
SAMPLES_BY_FEATURES = {"copynumber"}
# Views whose features are sparse measurements where MOSA drops features
# missing in most samples.
SPARSE_VIEWS = {"proteomics", "metabolomics", "drugresponse", "crisprcas9"}

# EMT signature: a compact epithelial / mesenchymal marker set drawn from the
# pan-cancer EMT signatures of Byers et al. (2013) and Mak et al. (2016).
EPITHELIAL_GENES = [
    "CDH1",
    "EPCAM",
    "CLDN4",
    "CLDN7",
    "KRT8",
    "KRT18",
    "KRT19",
    "ESRP1",
    "GRHL2",
    "OVOL1",
    "ST14",
    "TJP3",
    "CDS1",
    "MAL2",
    "RAB25",
    "SPINT1",
]
MESENCHYMAL_GENES = [
    "VIM",
    "ZEB1",
    "ZEB2",
    "SNAI2",
    "TWIST1",
    "FN1",
    "CDH2",
    "SPARC",
    "COL1A1",
    "COL1A2",
    "COL3A1",
    "COL5A1",
    "FAP",
    "PDGFRB",
    "TGFBI",
    "AXL",
    "MMP2",
    "LOX",
    "DCN",
    "POSTN",
]
MIN_TISSUE_SIZE = 10
MIN_MUTATION_OBS = 15
# EMT is not a meaningful axis for blood cancers; their score is left missing
# so the EMT factor does not degenerate into a haematopoietic-vs-solid switch.
NON_EPITHELIAL_TISSUES = ("Haematopoietic and Lymphoid",)


# MuData helpers --------------------------------------------------------------


def write_dataset(dataset: MultiOmicDataset, path: str | Path) -> None:
    """Write a MultiOmicDataset as an h5mu file MOSA can load."""
    obs = dataset.metadata.copy()
    obs.index = obs.index.astype(str)
    mods = {}
    for view, x in dataset.views.items():
        adata = anndata.AnnData(
            X=np.asarray(x, dtype=np.float32),
            obs=pd.DataFrame(index=obs.index),
            var=pd.DataFrame(index=pd.Index(dataset.feature_names[view], dtype=str)),
        )
        adata.layers[MASK_LAYER] = np.asarray(dataset.masks[view], dtype=bool)
        mods[view] = adata
    with mudata_set_options(pull_on_update=False):
        mdata = MuData(mods)
        mdata.obs = obs.loc[mdata.obs_names]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        mdata.write(str(path))
    logger.info("Wrote %s (%d samples, %d views)", path, len(obs), len(mods))


def read_dataset(path: str | Path, views: list[str] | None = None) -> MultiOmicDataset:
    """Load an h5mu file; all modalities when ``views`` is None."""
    if views is None:
        import h5py

        with h5py.File(path, "r") as handle:
            views = sorted(handle["mod"].keys())
    return load_mudata(str(path), views, MASK_LAYER)


def dataset_from_frames(
    views: dict[str, pd.DataFrame], obs: pd.DataFrame
) -> MultiOmicDataset:
    """Assemble a dataset from samples x features frames aligned to ``obs``."""
    matrices, masks, names = {}, {}, {}
    for view, frame in views.items():
        aligned = frame.reindex(index=obs.index)
        values = aligned.to_numpy(dtype=np.float32)
        masks[view] = ~np.isnan(values)
        matrices[view] = np.where(masks[view], values, np.nan).astype(np.float32)
        names[view] = list(aligned.columns.astype(str))
    dataset = MultiOmicDataset(matrices, masks, obs, names)
    dataset.validate()
    return dataset


# Raw tables ------------------------------------------------------------------


def load_view(raw_dir: Path, view: str) -> pd.DataFrame:
    """Read a raw table as samples x features."""
    frame = pd.read_csv(raw_dir / VIEW_FILES[view], index_col=0)
    if view not in SAMPLES_BY_FEATURES:
        frame = frame.T
    if view == "crisprcas9":
        frame.columns = frame.columns.str.split(" ").str[0]
    frame = frame.loc[:, ~frame.columns.duplicated()]
    frame.index = frame.index.astype(str)
    return frame.astype(np.float32)


def filter_features(
    frame: pd.DataFrame,
    view: str,
    max_missing: float = 0.85,
    top_variance: int | None = None,
) -> pd.DataFrame:
    """Drop mostly-missing features and keep the most variable ones."""
    if view in SPARSE_VIEWS:
        frame = frame.loc[:, frame.isna().mean() < max_missing]
    frame = frame.loc[:, frame.std() > 0]
    if top_variance is not None and frame.shape[1] > top_variance:
        keep = frame.var().sort_values(ascending=False).index[:top_variance]
        frame = frame[sorted(keep)]
    return frame


def emt_score(expression: pd.DataFrame) -> pd.Series:
    """Mesenchymal minus epithelial mean z-score of the marker genes."""
    present_e = [g for g in EPITHELIAL_GENES if g in expression.columns]
    present_m = [g for g in MESENCHYMAL_GENES if g in expression.columns]
    if len(present_e) < 3 or len(present_m) < 3:
        raise ValueError("Too few EMT marker genes found in the expression table")
    z = (expression - expression.mean()) / expression.std()
    return (z[present_m].mean(axis=1) - z[present_e].mean(axis=1)).rename("emt")


def mask_non_epithelial(score: pd.Series, tissue: pd.Series) -> pd.Series:
    """Set the EMT score to missing for tissues where EMT is undefined."""
    return score.where(~tissue.isin(NON_EPITHELIAL_TISSUES))


def collapse_rare(labels: pd.Series, min_size: int = MIN_TISSUE_SIZE) -> pd.Series:
    counts = labels.value_counts()
    rare = counts[counts < min_size].index
    return labels.where(~labels.isin(rare), "Other")


def driver_mutations(raw_dir: Path, min_obs: int = MIN_MUTATION_OBS) -> pd.DataFrame:
    """Binary cancer-driver mutation matrix (samples x genes) as ``mutation_*``."""
    table = pd.read_csv(raw_dir / "mutations_summary_20230202.csv")
    table = table[table["cancer_driver"].astype(str).str.lower() == "true"]
    matrix = pd.crosstab(table["model_id"], table["gene_symbol"]).clip(upper=1)
    matrix = matrix.loc[:, matrix.sum() >= min_obs]
    return matrix.add_prefix("mutation_").astype(np.float32)


def load_samplesheet(raw_dir: Path) -> pd.DataFrame:
    sheet = pd.read_csv(raw_dir / "samplesheet.csv", index_col=0)
    sheet.index = sheet.index.astype(str)
    return sheet


def build_obs(
    raw_dir: Path, samples: list[str], expression: pd.DataFrame
) -> pd.DataFrame:
    """Sample annotations: tissue, EMT scores, MSI, growth and driver mutations."""
    sheet = load_samplesheet(raw_dir).reindex(samples)
    cancercell = pd.read_csv(
        raw_dir / "samplesheet_cancercell.csv", index_col=0, encoding="utf-8-sig"
    )
    growth = (
        pd.read_csv(raw_dir / "growth_rate_20220907.csv")
        .groupby("model_id")[["day4_day1_ratio", "doubling_time_hours"]]
        .mean()
    )
    obs = pd.DataFrame(index=pd.Index(samples, name="model_id"))
    obs["model_type"] = "cell_line"
    obs["tissue"] = collapse_rare(sheet["tissue"].fillna("Unknown"))
    obs["cancer_type"] = sheet["cancer_type"]
    obs["msi_status"] = sheet["msi_status"]
    obs["emt_all_lines"] = emt_score(expression).reindex(samples)
    obs["emt"] = mask_non_epithelial(obs["emt_all_lines"], obs["tissue"])
    obs["emt_proteomics"] = cancercell["EMT"].reindex(samples)
    obs["growth_day4_day1"] = growth["day4_day1_ratio"].reindex(samples)
    mutations = driver_mutations(raw_dir).reindex(samples).fillna(0.0)
    return pd.concat([obs, mutations], axis=1)


def build_cell_lines(
    raw_dir: str | Path,
    views: list[str] | None = None,
    top_variance: dict[str, int] | None = None,
    min_views: int = 2,
) -> MultiOmicDataset:
    """Assemble the multi-omic cell line dataset."""
    raw_dir = Path(raw_dir)
    views = views or list(VIEW_FILES)
    top_variance = top_variance or {}
    frames = {}
    for view in views:
        frame = load_view(raw_dir, view)
        frames[view] = filter_features(frame, view, top_variance=top_variance.get(view))
        logger.info("%s: %s", view, frames[view].shape)
    counts = pd.concat([pd.Series(1, index=f.index) for f in frames.values()])
    counts = counts.groupby(level=0).sum()
    sheet = load_samplesheet(raw_dir)
    samples = sorted(set(counts[counts >= min_views].index) & set(sheet.index))
    obs = build_obs(raw_dir, samples, frames["transcriptomics"])
    return dataset_from_frames(frames, obs)


def _paired_rows(
    sanger: pd.DataFrame, ccle: pd.DataFrame, shared: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Observed frame, partner-measurement frame and obs for the paired design.

    Sanger rows keep their model identifier; CCLE rows get a ``__ccle`` suffix.
    The partner frame holds, for every row, the measurement of the same cell
    line in the other laboratory (NaN when the line was measured once), which
    serves as the ground truth for counterfactual decoding across laboratories.
    """
    ccle = ccle.loc[ccle.index.str.startswith("SIDM")]
    ccle_rows = ccle.index + "__ccle"
    observed = pd.concat([sanger[shared], ccle[shared].set_axis(ccle_rows)]).astype(
        np.float32
    )
    partner = pd.concat(
        [
            ccle[shared].reindex(sanger.index),
            sanger[shared].reindex(ccle.index).set_axis(ccle_rows),
        ]
    ).astype(np.float32)
    obs = pd.DataFrame(index=observed.index)
    obs.index.name = "model_id"
    obs["cell_line"] = [i.split("__")[0] for i in observed.index]
    obs["model_type"] = ["ccle" if "__ccle" in i else "sanger" for i in observed.index]
    obs["has_partner"] = partner.notna().any(axis=1).to_numpy()
    return observed, partner, obs


def build_proteomics_paired(
    raw_dir: str | Path,
    cell_lines: MultiOmicDataset | None = None,
    top_variance: int | None = 4000,
) -> tuple[MultiOmicDataset, MultiOmicDataset]:
    """Proteomics measured in two laboratories, with paired cell lines.

    Returns (observed, partner_truth). 290 lines were measured by both Sanger
    and CCLE, so the partner frame gives a real batch-free target: what the
    other laboratory measured for the same line. Tissue comes from the sample
    sheet and the EMT score from the transcriptomic cohort when available.
    """
    raw_dir = Path(raw_dir)
    sanger = load_view(raw_dir, "proteomics")
    ccle = pd.read_csv(raw_dir / "proteomics_ccle.csv", index_col=0).T
    ccle = ccle.loc[:, ~ccle.columns.duplicated()].astype(np.float32)
    shared = sorted(set(sanger.columns) & set(ccle.columns))
    observed, partner, obs = _paired_rows(sanger, ccle, shared)
    observed = filter_features(observed, "proteomics", top_variance=top_variance)
    partner = partner[observed.columns]
    sheet = load_samplesheet(raw_dir)
    tissue = sheet["tissue"].reindex(obs["cell_line"]).fillna("Unknown")
    obs["tissue"] = collapse_rare(tissue.set_axis(obs.index))
    if cell_lines is not None:
        emt = cell_lines.metadata["emt_all_lines"]
    else:
        emt = emt_score(load_view(raw_dir, "transcriptomics"))
    emt = pd.Series(emt.reindex(obs["cell_line"]).to_numpy(), index=obs.index)
    obs["emt"] = mask_non_epithelial(emt, obs["tissue"])
    dataset = dataset_from_frames({"proteomics": observed}, obs)
    truth = dataset_from_frames({"proteomics": partner}, obs)
    return dataset, truth
