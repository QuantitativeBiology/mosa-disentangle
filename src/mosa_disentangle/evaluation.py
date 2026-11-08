"""Evaluate a fitted run: reconstruction, counterfactuals, probes, mixing.

``evaluate_run`` produces one flat metrics dictionary plus the probe matrix
and latent tables used by the notebooks. All scores are computed on the
validation fold with models fitted on the training fold only.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from mosa.data.dataset import MultiOmicDataset

from mosa_disentangle import metrics as m
from mosa_disentangle.config import FREE_BLOCK
from mosa_disentangle.training import (
    FittedRun,
    Predictions,
    block_prototype,
    predict_dataset,
)
from mosa_disentangle.training import block_columns as latent_block_columns

logger = logging.getLogger(__name__)

SOURCE_FACTOR = "model_type"
PHENOTYPE_VIEWS = ("drugresponse", "crisprcas9")
GROUP_COLUMN = "cell_line"


def factor_kinds(run: FittedRun) -> dict[str, str]:
    kinds = {name: f.kind for name, f in run.model_cfg.factors.items()}
    kinds.setdefault(SOURCE_FACTOR, "categorical")
    kinds.setdefault("tissue", "categorical")
    return kinds


def block_columns(run: FittedRun) -> dict[str, list[str]]:
    """Latent column names of every block, keyed by block name."""
    layout = run.model.layout
    return {n: latent_block_columns(layout.columns(), n) for n in layout.names}


def biological_columns(run: FittedRun) -> list[str]:
    """All latent columns except the technical (source) block."""
    blocks = block_columns(run)
    return [c for name, cols in blocks.items() if name != SOURCE_FACTOR for c in cols]


def mask_view(data: MultiOmicDataset, view: str) -> MultiOmicDataset:
    """Copy of ``data`` with one view entirely unobserved at the encoder."""
    views = {k: v.copy() for k, v in data.views.items()}
    masks = {k: v.copy() for k, v in data.masks.items()}
    views[view][:] = np.nan
    masks[view][:] = False
    return MultiOmicDataset(views, masks, data.metadata.copy(), data.feature_names)


def reconstruction_metrics(
    run: FittedRun, val: MultiOmicDataset, pred: Predictions
) -> dict:
    """Held-out NMSE per view, plus cross-view imputation of each masked view."""
    scores = {}
    for view, score in m.view_nmse(val.views, pred.recon, val.masks).items():
        scores[f"nmse_{view}"] = score
    for view in val.view_names:
        if val.masks[view].sum() == 0:
            continue
        imputed = predict_dataset(run, mask_view(val, view)).recon[view]
        scores[f"impute_{view}"] = m.nmse(val.views[view], imputed, val.masks[view])
    observed = [
        v for k, v in scores.items() if k.startswith("nmse_") and np.isfinite(v)
    ]
    scores["nmse_mean"] = float(np.mean(observed)) if observed else float("nan")
    return scores


def counterfactual_metrics(
    run: FittedRun,
    train: MultiOmicDataset,
    val: MultiOmicDataset,
    truth: MultiOmicDataset,
    pred_train: Predictions,
    reference: str,
) -> dict:
    """Batch-removed reconstruction against the ground truth of the reference.

    Three decodings of the validation samples are compared to the truth:
    uncorrected, source conditional swapped (the MOSA mechanism), and
    conditional swapped plus the source block replaced by the reference
    prototype. ``*_missing`` scores cover entries unobserved in the input
    (views absent by design in a pseudo-source), i.e. counterfactual imputation.
    """
    categories = run.datamodule.batch_categories
    ref_id = categories.index(reference)
    non_ref = val.metadata[SOURCE_FACTOR].to_numpy() != reference
    strategies = {"none": {}, "cond": {"force_source_id": ref_id}}
    if SOURCE_FACTOR in run.model.layout.names:
        ref_rows = (train.metadata[SOURCE_FACTOR] == reference).to_numpy()
        prototype = block_prototype(pred_train.latent, SOURCE_FACTOR, ref_rows)
        strategies["cond_block"] = {
            "force_source_id": ref_id,
            "block_overrides": {SOURCE_FACTOR: prototype},
        }
    scores = {}
    for name, kwargs in strategies.items():
        recon = predict_dataset(run, val, **kwargs).recon
        for view in val.view_names:
            seen = val.masks[view] & truth.masks[view]
            unseen = ~val.masks[view] & truth.masks[view]
            seen[~non_ref], unseen[~non_ref] = False, False
            scores[f"cf_{name}_{view}"] = m.nmse(truth.views[view], recon[view], seen)
            scores[f"cf_{name}_{view}_missing"] = m.nmse(
                truth.views[view], recon[view], unseen
            )
        prefix = f"cf_{name}_"
        kept = [
            v
            for k, v in scores.items()
            if k.startswith(prefix) and not k.endswith("_missing") and np.isfinite(v)
        ]
        scores[f"cf_{name}_mean"] = float(np.mean(kept)) if kept else float("nan")
    return scores


def latent_metrics(
    run: FittedRun,
    train: MultiOmicDataset,
    val: MultiOmicDataset,
    pred_train: Predictions,
    pred_val: Predictions,
) -> tuple[dict, pd.DataFrame]:
    """Probe matrix, chance-corrected summary, mixing and tissue structure."""
    kinds = factor_kinds(run)
    kinds = {k: v for k, v in kinds.items() if k in val.metadata.columns}
    blocks = block_columns(run)
    matrix = m.probe_matrix(
        pred_train.latent, pred_val.latent, blocks, kinds, train.metadata, val.metadata
    )
    chance = m.chance_levels(val.metadata, kinds)
    scores = m.disentanglement_summary(matrix, chance)
    for factor in kinds:
        scores[f"probe_all_{factor}"] = float(matrix.loc["all", factor])
    bio = pred_val.latent[biological_columns(run)].to_numpy()
    everything = pred_val.latent.to_numpy()
    source = val.metadata[SOURCE_FACTOR]
    scores["source_mixing_bio"] = m.knn_label_entropy(bio, source)
    scores["source_mixing_all"] = m.knn_label_entropy(everything, source)
    if "tissue" in val.metadata:
        tissue = val.metadata["tissue"]
        scores["tissue_silhouette_bio"] = m.label_silhouette(bio, tissue)
        scores["tissue_silhouette_all"] = m.label_silhouette(everything, tissue)
        if "tissue" in blocks:
            scores["tissue_silhouette_block"] = m.label_silhouette(
                pred_val.block("tissue"), tissue
            )
        scores["tissue_mixing_bio"] = m.knn_label_entropy(bio, tissue)
    return scores, matrix


def downstream_metrics(
    run: FittedRun,
    train: MultiOmicDataset,
    val: MultiOmicDataset,
    pred_train: Predictions,
    pred_val: Predictions,
) -> dict:
    """Ridge R2 from the biological blocks to drug response and CRISPR."""
    scores = {}
    bio_cols = biological_columns(run)
    free_cols = block_columns(run).get(FREE_BLOCK, [])
    for view in PHENOTYPE_VIEWS:
        if view not in train.views:
            continue
        y_train = np.where(train.masks[view], train.views[view], np.nan)
        y_val = np.where(val.masks[view], val.views[view], np.nan)
        for label, cols in (("bio", bio_cols), ("free", free_cols)):
            if not cols:
                continue
            scores[f"downstream_{view}_{label}"] = m.downstream_r2(
                pred_train.latent[cols].to_numpy(),
                y_train,
                pred_val.latent[cols].to_numpy(),
                y_val,
            )
    return scores


def paired_metrics(
    run: FittedRun, val: MultiOmicDataset, pred_val: Predictions
) -> dict:
    """Partner retrieval across batches when a grouping column is present."""
    if GROUP_COLUMN not in val.metadata.columns:
        return {}
    groups, batches = val.metadata[GROUP_COLUMN], val.metadata[SOURCE_FACTOR]
    scores = {}
    for label, cols in (
        ("bio", biological_columns(run)),
        ("all", list(pred_val.latent)),
    ):
        result = m.paired_retrieval(pred_val.latent[cols].to_numpy(), groups, batches)
        scores.update({f"{k}_{label}": v for k, v in result.items()})
    return scores


def diverged(*preds: Predictions) -> bool:
    """True when any latent code is non-finite (the optimisation blew up)."""
    return any(not np.isfinite(p.latent.to_numpy()).all() for p in preds)


def evaluate_run(
    run: FittedRun,
    train: MultiOmicDataset,
    val: MultiOmicDataset,
    truth: MultiOmicDataset | None = None,
    reference: str | None = None,
) -> tuple[dict, pd.DataFrame, Predictions, Predictions]:
    """All metrics for one fold. Returns (metrics, probe matrix, preds).

    A run whose latent codes are non-finite is reported with ``diverged=1``
    and no other metric, so a sweep that hits an unstable setting still
    completes and the failure is visible in the aggregated tables.
    """
    pred_train = predict_dataset(run, train)
    pred_val = predict_dataset(run, val)
    n_epochs = int(len(run.history))
    if diverged(pred_train, pred_val):
        logger.warning("Non-finite latent codes: training diverged")
        scores = {"diverged": 1, "n_epochs": n_epochs}
        return scores, pd.DataFrame(), pred_train, pred_val
    scores = {"diverged": 0}
    scores.update(reconstruction_metrics(run, val, pred_val))
    latent_scores, matrix = latent_metrics(run, train, val, pred_train, pred_val)
    scores.update(latent_scores)
    scores.update(downstream_metrics(run, train, val, pred_train, pred_val))
    scores.update(paired_metrics(run, val, pred_val))
    if truth is not None and reference is not None:
        scores.update(
            counterfactual_metrics(run, train, val, truth, pred_train, reference)
        )
    scores["n_epochs"] = n_epochs
    return scores, matrix, pred_train, pred_val
