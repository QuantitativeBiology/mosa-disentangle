"""Shared fixtures: a tiny synthetic multi-omic dataset with known factors."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from mosa.config import DataConfig

from mosa_disentangle.config import DisentangleConfig
from mosa_disentangle.data import dataset_from_frames

N_SAMPLES = 120
VIEWS = {"rna": 40, "prot": 25}


@pytest.fixture(scope="session")
def toy_dataset():
    rng = np.random.default_rng(0)
    source = rng.choice(["cell_line", "tumour"], N_SAMPLES)
    tissue = rng.choice(["Lung", "Breast", "Skin"], N_SAMPLES)
    emt = rng.normal(size=N_SAMPLES)
    obs = pd.DataFrame(
        {"model_type": source, "tissue": tissue, "emt": emt},
        index=[f"S{i:03d}" for i in range(N_SAMPLES)],
    )
    latent = np.stack(
        [
            (source == "tumour").astype(float),
            (tissue == "Lung").astype(float),
            (tissue == "Breast").astype(float),
            emt,
        ],
        axis=1,
    )
    frames = {}
    for view, width in VIEWS.items():
        loading = rng.normal(size=(latent.shape[1], width))
        values = latent @ loading + rng.normal(scale=0.3, size=(N_SAMPLES, width))
        values[rng.random(values.shape) < 0.05] = np.nan
        frames[view] = pd.DataFrame(
            values, index=obs.index, columns=[f"{view}_{j}" for j in range(width)]
        )
    frames["prot"].iloc[:10] = np.nan  # samples without the second view
    return dataset_from_frames(frames, obs)


@pytest.fixture
def raw_config():
    return {
        "data": {
            "path": "",
            "views": ["rna", "prot"],
            "use_tissue": True,
            "use_mutations": False,
        },
        "model": {
            "views": {
                "rna": {"hidden_layer_dims": [16]},
                "prot": {"hidden_layer_dims": [16]},
            },
            "free_dim": 4,
            "factors": {
                "model_type": {
                    "column": "model_type",
                    "block_dim": 2,
                    "supervised_weight": 1.0,
                    "adversarial_weight": 0.5,
                    "hsic_weight": 1.0,
                },
                "tissue": {
                    "column": "tissue",
                    "block_dim": 3,
                    "supervised_weight": 1.0,
                },
                "emt": {
                    "column": "emt",
                    "kind": "continuous",
                    "block_dim": 1,
                    "supervised_weight": 1.0,
                    "hsic_weight": 0.5,
                },
            },
            "block_hsic_weight": 0.1,
            "dip_lambda_od": 1.0,
            "dip_lambda_d": 0.1,
            "decoder_conditionals": ["tissue"],
            "num_epochs": 3,
            "batch_size": 32,
            "accelerator": "cpu",
            "early_stopping_patience": 5,
            "weighted_random_sampler": True,
        },
    }


@pytest.fixture
def configs(raw_config) -> tuple[DataConfig, DisentangleConfig]:
    from mosa_disentangle.training import build_configs

    return build_configs(raw_config)
