"""Data module adding factor targets to the MOSA minibatches.

``FactorDataModule`` reuses the MOSA preprocessing (scaling, conditionals,
missing-value handling) and wraps the resulting datasets so that every batch
also carries, for each configured factor, an encoded target and an observed
mask. Categorical factors are integer codes fitted on the training samples;
continuous factors are z-scored with training statistics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from mosa.config import DataConfig
from mosa.data.dataset import MultiOmicDataset
from mosa.models.mosa.datamodule import MOSADataModule
from torch.utils.data import Dataset

from mosa_disentangle.config import DisentangleConfig, FactorConfig


class FactorEncoder:
    """Fit-on-train encoder for one factor column."""

    def __init__(self, factor: FactorConfig):
        self.factor = factor
        self.categories: list[str] = []
        self.mean = 0.0
        self.std = 1.0

    @property
    def n_classes(self) -> int:
        return len(self.categories) if self.factor.kind == "categorical" else 1

    def fit(self, obs: pd.DataFrame) -> FactorEncoder:
        values = obs[self.factor.column]
        if self.factor.kind == "categorical":
            self.categories = sorted(values.dropna().astype(str).unique())
        else:
            numeric = pd.to_numeric(values, errors="coerce")
            self.mean = float(numeric.mean())
            self.std = float(numeric.std(ddof=0)) or 1.0
        return self

    def transform(self, obs: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (target, mask); unseen categories and NaNs are masked out."""
        values = obs[self.factor.column]
        if self.factor.kind == "categorical":
            codes = pd.Categorical(
                values.astype(str), categories=self.categories
            ).codes.astype(np.int64)
            mask = codes >= 0
            return torch.from_numpy(codes), torch.from_numpy(mask)
        numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float32)
        mask = ~np.isnan(numeric)
        scaled = np.where(mask, (numeric - self.mean) / self.std, 0.0)
        return torch.from_numpy(scaled.astype(np.float32)), torch.from_numpy(mask)

    def state_dict(self) -> dict:
        return {"categories": self.categories, "mean": self.mean, "std": self.std}

    def load_state_dict(self, state: dict) -> None:
        self.categories = list(state["categories"])
        self.mean = float(state["mean"])
        self.std = float(state["std"])


class _WithFactors(Dataset):
    """Wrap a MOSA dataset so each item also carries factor targets and masks."""

    def __init__(
        self,
        base: Dataset,
        targets: dict[str, torch.Tensor],
        masks: dict[str, torch.Tensor],
    ):
        self.base = base
        self.targets = targets
        self.masks = masks

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict:
        item = self.base[idx]
        item["factors"] = {k: v[idx] for k, v in self.targets.items()}
        item["factor_masks"] = {k: v[idx] for k, v in self.masks.items()}
        return item


class FactorDataModule(MOSADataModule):
    """MOSA data module whose batches include encoded factor targets."""

    def __init__(
        self,
        train_data: MultiOmicDataset | None,
        val_data: MultiOmicDataset | None,
        data_cfg: DataConfig,
        model_cfg: DisentangleConfig,
    ):
        super().__init__(train_data, val_data, data_cfg, model_cfg)
        self.factor_encoders = {
            name: FactorEncoder(f) for name, f in model_cfg.factors.items()
        }

    @property
    def factor_info(self) -> dict[str, dict]:
        return {
            name: {"kind": enc.factor.kind, "n_classes": enc.n_classes}
            for name, enc in self.factor_encoders.items()
        }

    @property
    def n_tissues(self) -> int:
        return len(self.tissue_categories)

    def setup(self, stage: str | None = None) -> None:
        if self.train_dataset is not None:
            return
        super().setup(stage)
        for enc in self.factor_encoders.values():
            enc.fit(self.train_data.metadata)
        self._wrap_datasets()

    def setup_inference(self, source: MOSADataModule) -> None:
        super().setup_inference(source)
        if isinstance(source, FactorDataModule):
            for name, enc in source.factor_encoders.items():
                self.factor_encoders[name].load_state_dict(enc.state_dict())
        self._wrap_datasets()

    def _wrap_datasets(self) -> None:
        self.train_dataset = self._wrap(self.train_dataset, self.train_data)
        if self.val_dataset is not None and self.val_data is not None:
            self.val_dataset = self._wrap(self.val_dataset, self.val_data)

    def _wrap(self, dataset: Dataset, data: MultiOmicDataset) -> Dataset:
        if isinstance(dataset, _WithFactors):
            dataset = dataset.base
        targets, masks = {}, {}
        for name, enc in self.factor_encoders.items():
            targets[name], masks[name] = enc.transform(data.metadata)
        return _WithFactors(dataset, targets, masks)

    def _get_weighted_sampler(self, dataset: Dataset):
        base = dataset.base if isinstance(dataset, _WithFactors) else dataset
        return super()._get_weighted_sampler(base)

    def state_dict(self) -> dict:
        state = super().state_dict()
        state["factor_encoders"] = {
            name: enc.state_dict() for name, enc in self.factor_encoders.items()
        }
        return state

    def load_state_dict(self, state: dict) -> None:
        super().load_state_dict(state)
        for name, enc_state in state.get("factor_encoders", {}).items():
            self.factor_encoders[name].load_state_dict(enc_state)
