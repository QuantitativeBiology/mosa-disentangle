"""Training and inference harness around ``DisentangledVAE``.

The harness mirrors ``mosa.models.mosa.model.MOSAModel.fit`` but keeps the
pieces explicit so experiments can control splits, counterfactual decoding
and what gets saved. Configurations are MOSA-style YAML files with ``data``
and ``model`` sections; a dataset base file is merged with a variant file so
variants only state what they change.
"""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import yaml
from mosa.config import DataConfig
from mosa.data.dataset import MultiOmicDataset
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.model_selection import GroupKFold, StratifiedKFold

from mosa_disentangle.config import DisentangleConfig
from mosa_disentangle.datamodule import FactorDataModule
from mosa_disentangle.model import DisentangledVAE

logger = logging.getLogger(__name__)

MODEL_FILE = "model.pt"
CONFIG_FILE = "config.yaml"
HISTORY_FILE = "history.csv"


# Configuration ---------------------------------------------------------------


def deep_update(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into a copy of ``base``."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_yaml(path: str | Path) -> dict:
    with open(path) as handle:
        return yaml.safe_load(handle) or {}


def build_configs(raw: dict) -> tuple[DataConfig, DisentangleConfig]:
    """Instantiate the data and model configs from a merged YAML mapping."""
    data_raw = dict(raw["data"])
    model_raw = dict(raw["model"])
    model_raw.pop("type", None)
    data_cfg = DataConfig(**data_raw)
    model_cfg = DisentangleConfig.from_yaml_dict(model_raw)
    if set(model_cfg.views) != set(data_cfg.views):
        raise ValueError("model.views and data.views must list the same views")
    return data_cfg, model_cfg


def load_experiment(
    base_path: str | Path, variant_path: str | Path | None = None, **overrides
) -> dict:
    """Merge a dataset base config, an optional variant and keyword overrides."""
    raw = load_yaml(base_path)
    if variant_path is not None:
        raw = deep_update(raw, load_yaml(variant_path))
    return deep_update(raw, overrides)


# Splits ----------------------------------------------------------------------


def make_folds(
    obs: pd.DataFrame,
    n_folds: int,
    seed: int,
    stratify: str = "model_type",
    group: str | None = None,
) -> list[np.ndarray]:
    """Validation indices per fold, stratified by a column or grouped."""
    indices = np.arange(len(obs))
    if group is not None:
        splitter = GroupKFold(n_splits=n_folds)
        splits = splitter.split(indices, groups=obs[group].to_numpy())
    else:
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        splits = splitter.split(indices, obs[stratify].astype(str).to_numpy())
    return [val for _, val in splits]


def split_dataset(
    dataset: MultiOmicDataset, val_idx: np.ndarray
) -> tuple[MultiOmicDataset, MultiOmicDataset]:
    train_idx = np.setdiff1d(np.arange(dataset.n_samples), val_idx)
    return dataset.subset(train_idx), dataset.subset(val_idx)


# Fitting ---------------------------------------------------------------------


@dataclass
class FittedRun:
    """A trained model with the data module that holds its preprocessing."""

    model: DisentangledVAE
    datamodule: FactorDataModule
    data_cfg: DataConfig
    model_cfg: DisentangleConfig
    history: pd.DataFrame


class _History(pl.Callback):
    def __init__(self):
        self.rows: list[dict] = []

    def on_train_epoch_end(self, trainer, pl_module):
        row = {"epoch": trainer.current_epoch}
        for key, value in trainer.callback_metrics.items():
            row[key] = float(value)
        self.rows.append(row)


def build_model(dm: FactorDataModule, data_cfg, model_cfg) -> DisentangledVAE:
    return DisentangledVAE(
        config=model_cfg,
        view_input_dims=dm.view_input_dims,
        conditional_dim=dm.conditional_dim,
        n_batches=dm.n_batches,
        n_tissues=dm.n_tissues,
        factor_info=dm.factor_info,
        data_cfg=data_cfg,
    )


def fit_model(
    train: MultiOmicDataset,
    val: MultiOmicDataset | None,
    data_cfg: DataConfig,
    model_cfg: DisentangleConfig,
    out_dir: str | Path,
) -> FittedRun:
    """Fit a disentangled VAE and return it with its data module."""
    pl.seed_everything(model_cfg.random_seed, workers=True)
    dm = FactorDataModule(train, val, data_cfg, model_cfg)
    dm.setup()
    model = build_model(dm, data_cfg, model_cfg)
    if dm.class_weights is not None:
        model.class_weights = torch.tensor(dm.class_weights, dtype=torch.float32)
    model.model_type_names = dm.batch_categories

    history = _History()
    callbacks: list[pl.Callback] = [history]
    if val is not None:
        callbacks.append(
            EarlyStopping(
                monitor="val/loss",
                patience=model_cfg.early_stopping_patience,
                mode="min",
            )
        )
    trainer = pl.Trainer(
        max_epochs=model_cfg.num_epochs,
        callbacks=callbacks,
        default_root_dir=str(out_dir),
        accelerator=model_cfg.accelerator,
        devices=model_cfg.devices,
        precision=model_cfg.precision,
        log_every_n_steps=1,
        enable_checkpointing=False,
        enable_progress_bar=False,
        logger=False,
        limit_val_batches=1.0 if val is not None else 0,
        num_sanity_val_steps=0,
    )
    trainer.fit(model, dm)
    return FittedRun(model, dm, data_cfg, model_cfg, pd.DataFrame(history.rows))


# Inference -------------------------------------------------------------------


def block_columns(columns, block: str) -> list[str]:
    """Latent column names belonging to ``block`` (``<block>_<i>``).

    The integer suffix is required so that metadata columns sharing a prefix
    with a block (``emt_proteomics``, ``model_type_true``) are not picked up
    when the latent table has been joined with ``obs``.
    """
    prefix = f"{block}_"
    return [c for c in columns if c.startswith(prefix) and c[len(prefix) :].isdigit()]


@dataclass
class Predictions:
    """Latent coordinates and original-scale reconstructions for a dataset."""

    latent: pd.DataFrame
    recon: dict[str, np.ndarray]
    head_outputs: dict[str, np.ndarray]
    source_ids: np.ndarray

    def block(self, name: str) -> np.ndarray:
        return self.latent[block_columns(self.latent.columns, name)].to_numpy()


def inference_loader(run: FittedRun, data: MultiOmicDataset):
    """Data loader applying the run's fitted preprocessing to ``data``."""
    dm = FactorDataModule(data, None, run.data_cfg, run.model_cfg)
    dm.setup_inference(run.datamodule)
    return dm, dm.train_eval_dataloader()


def predict_dataset(
    run: FittedRun,
    data: MultiOmicDataset,
    force_source_id: int | None = None,
    block_overrides: dict[str, np.ndarray] | None = None,
) -> Predictions:
    """Encode and decode ``data``, optionally under a counterfactual source."""
    dm, loader = inference_loader(run, data)
    overrides = None
    if block_overrides:
        overrides = {
            k: torch.as_tensor(v, dtype=torch.float32)
            for k, v in block_overrides.items()
        }
    out = run.model.predict(loader, force_source_id, overrides)
    source_ids = out["source_ids"]
    if force_source_id is not None:
        source_ids = np.full_like(source_ids, force_source_id)
    recon = {
        view: dm.inverse_transform_view(view, values, source_ids=source_ids)
        for view, values in out["x_hat"].items()
    }
    latent = pd.DataFrame(
        out["mu"], index=out["sample_names"], columns=run.model.layout.columns()
    )
    return Predictions(latent, recon, out["head_outputs"], out["source_ids"])


def block_prototype(latent: pd.DataFrame, block: str, rows: np.ndarray) -> np.ndarray:
    """Mean block vector over ``rows`` (e.g. the reference-source training samples)."""
    cols = block_columns(latent.columns, block)
    return latent.loc[rows, cols].mean(axis=0).to_numpy()


# Persistence -----------------------------------------------------------------


def save_run(run: FittedRun, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": run.model.state_dict(),
            "datamodule": run.datamodule.state_dict(),
            "hparams": dict(run.model.hparams),
        },
        out_dir / MODEL_FILE,
    )
    run.history.to_csv(out_dir / HISTORY_FILE, index=False)
    with open(out_dir / CONFIG_FILE, "w") as handle:
        yaml.safe_dump(json.loads(json.dumps(run.model.export_config())), handle)


def load_run(
    out_dir: str | Path,
    train: MultiOmicDataset,
    data_cfg: DataConfig,
    model_cfg: DisentangleConfig,
) -> FittedRun:
    """Rebuild a saved run; ``train`` only supplies the preprocessing fit."""
    out_dir = Path(out_dir)
    saved = torch.load(out_dir / MODEL_FILE, map_location="cpu", weights_only=False)
    dm = FactorDataModule(train, None, data_cfg, model_cfg)
    dm.setup()
    dm.load_state_dict(saved["datamodule"])
    model = build_model(dm, data_cfg, model_cfg)
    model.load_state_dict(saved["state_dict"])
    model.model_type_names = dm.batch_categories
    history = pd.read_csv(out_dir / HISTORY_FILE)
    return FittedRun(model, dm, data_cfg, model_cfg, history)
