"""Semi-synthetic technical factor with a known ground truth.

The public MOSA data contains cell lines only, so the "model type" factor of
the thesis (cell line / organoid / tumour) is emulated: every cell line is
assigned to a pseudo-source, and the sources other than the reference receive
feature-wise shifts, feature-wise rescaling, a coherent "programme" shift and
source-specific missing views. The untouched matrices are kept as the ground
truth for batch-removed reconstruction. Assignment can be confounded with
tissue to study the biology-versus-technical trade-off that motivates the
thesis.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from mosa.data.dataset import MultiOmicDataset


@dataclass
class BatchSimConfig:
    """Parameters of the simulated technical factor."""

    sources: list[str] = field(
        default_factory=lambda: ["cell_line", "organoid", "tumour"]
    )
    # 0 = assignment independent of tissue; 1 = fully determined by tissue.
    confounding: float = 0.5
    # Additive shift per feature, in units of the feature's standard deviation.
    shift_sd: float = 0.8
    # Log-scale change per feature.
    scale_sd: float = 0.2
    # Fraction of features moved together by a source-specific programme.
    programme_fraction: float = 0.05
    programme_shift: float = 1.5
    # Views unobserved in a given pseudo-source.
    missing_views: dict[str, list[str]] = field(
        default_factory=lambda: {
            "organoid": ["proteomics", "metabolomics"],
            "tumour": ["crisprcas9", "drugresponse", "metabolomics"],
        }
    )
    seed: int = 0

    @property
    def reference(self) -> str:
        return self.sources[0]


def assign_sources(
    obs: pd.DataFrame, cfg: BatchSimConfig, rng: np.random.Generator
) -> pd.Series:
    """Draw a pseudo-source per sample with tissue-dependent probabilities."""
    n_sources = len(cfg.sources)
    uniform = np.full(n_sources, 1.0 / n_sources)
    tissues = obs["tissue"].astype(str)
    preferred = {t: rng.integers(n_sources) for t in tissues.unique()}
    assignment = np.empty(len(obs), dtype=object)
    for i, tissue in enumerate(tissues):
        onehot = np.eye(n_sources)[preferred[tissue]]
        probs = (1.0 - cfg.confounding) * uniform + cfg.confounding * onehot
        assignment[i] = cfg.sources[rng.choice(n_sources, p=probs)]
    return pd.Series(assignment, index=obs.index, name="model_type")


def _source_effects(
    n_features: int, n_sources: int, cfg: BatchSimConfig, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """Per-source feature shifts, log-scales and programme loadings."""
    shifts = rng.normal(0.0, cfg.shift_sd, size=(n_sources, n_features))
    scales = rng.normal(0.0, cfg.scale_sd, size=(n_sources, n_features))
    programme = np.zeros(n_features)
    n_prog = int(round(cfg.programme_fraction * n_features))
    programme[rng.choice(n_features, n_prog, replace=False)] = 1.0
    magnitudes = rng.normal(0.0, cfg.programme_shift, size=n_sources)
    shifts[0], scales[0], magnitudes[0] = 0.0, 0.0, 0.0  # reference source
    return {"shift": shifts, "scale": scales, "prog": programme, "mag": magnitudes}


def inject_source_effects(
    dataset: MultiOmicDataset, cfg: BatchSimConfig
) -> tuple[MultiOmicDataset, MultiOmicDataset]:
    """Return (observed dataset with technical effects, ground-truth dataset).

    Both share the ``.obs`` with the new ``model_type`` column; the truth keeps
    the original values and masks (including views made missing in the
    observed dataset), the observed one carries the distortions.
    """
    rng = np.random.default_rng(cfg.seed)
    obs = dataset.metadata.copy()
    obs["model_type"] = assign_sources(obs, cfg, rng)
    obs["model_type_true"] = dataset.metadata["model_type"].to_numpy()
    codes = pd.Categorical(obs["model_type"], categories=cfg.sources).codes
    views, masks = {}, {}
    for view, x in dataset.views.items():
        x = x.astype(np.float32).copy()
        mask = dataset.masks[view].copy()
        mean = np.nanmean(np.where(mask, x, np.nan), axis=0)
        sd = np.nanstd(np.where(mask, x, np.nan), axis=0)
        sd = np.where(np.isfinite(sd) & (sd > 0), sd, 1.0)
        eff = _source_effects(x.shape[1], len(cfg.sources), cfg, rng)
        centred = x - mean
        scaled = centred * np.exp(eff["scale"][codes])
        shift = (eff["shift"][codes] + eff["mag"][codes, None] * eff["prog"]) * sd
        x_obs = (mean + scaled + shift).astype(np.float32)
        reference = codes == 0
        x_obs[reference] = x[reference]  # the reference source is left untouched
        for source, dropped in cfg.missing_views.items():
            if view in dropped:
                rows = obs["model_type"].to_numpy() == source
                mask[rows] = False
        views[view] = np.where(mask, x_obs, np.nan).astype(np.float32)
        masks[view] = mask
    observed = MultiOmicDataset(views, masks, obs, dict(dataset.feature_names))
    truth = MultiOmicDataset(
        {k: v.copy() for k, v in dataset.views.items()},
        {k: v.copy() for k, v in dataset.masks.items()},
        obs.copy(),
        dict(dataset.feature_names),
    )
    observed.validate()
    truth.validate()
    return observed, truth
