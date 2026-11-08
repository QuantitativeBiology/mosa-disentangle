"""Disentangled MOSA VAE.

``DisentangledVAE`` keeps the MOSA encoders, fusion and conditional decoders
and adds a latent layout ``[block_1 | ... | block_k | free]``. Each factor block
can be tied to its factor with a supervised head, protected from leaking into
the other blocks with a discriminator, and decorrelated from them with a CKA
penalty. The free block carries whatever variation the factors do not explain.
"""

from __future__ import annotations

import dataclasses
import logging

import torch
import torch.nn as nn
from mosa.config import DataConfig
from mosa.models.mosa.vae.layers import MLP
from mosa.models.mosa.vae.vae_module import VAE, _batch_to_device

from mosa_disentangle.config import DisentangleConfig
from mosa_disentangle.losses import (
    cka,
    confusion_loss,
    dip_vae_regulariser,
    factor_kernel_input,
    supervised_loss,
)

logger = logging.getLogger(__name__)


class BlockLayout:
    """Slices of the joint latent vector owned by each block."""

    def __init__(self, block_dims: dict[str, int]):
        self.dims = dict(block_dims)
        self.slices: dict[str, slice] = {}
        start = 0
        for name, dim in self.dims.items():
            self.slices[name] = slice(start, start + dim)
            start += dim
        self.total = start

    @property
    def names(self) -> list[str]:
        return [n for n, d in self.dims.items() if d > 0]

    def block(self, z: torch.Tensor, name: str) -> torch.Tensor:
        return z[:, self.slices[name]]

    def complement(self, z: torch.Tensor, name: str) -> torch.Tensor:
        """Everything except ``name``; the whole vector if there is no block."""
        if name not in self.slices or self.dims[name] == 0:
            return z
        sl = self.slices[name]
        return torch.cat([z[:, : sl.start], z[:, sl.stop :]], dim=1)

    def complement_dim(self, name: str) -> int:
        return self.total - self.dims.get(name, 0)

    def split(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        return {n: self.block(z, n) for n in self.names}

    def columns(self) -> list[str]:
        return [f"{n}_{i}" for n in self.names for i in range(self.dims[n])]


def _head(in_dim: int, out_dim: int, hidden: int = 0) -> nn.Module:
    sizes = [in_dim] + ([hidden] if hidden > 0 else []) + [out_dim]
    return MLP(sizes, dropout_p=0.0, use_batch_norm=hidden > 0, activation=nn.ReLU)


class DisentangledVAE(VAE):
    """MOSA VAE with factor-specific latent blocks and leakage penalties."""

    def __init__(
        self,
        config: DisentangleConfig,
        view_input_dims: dict[str, int],
        conditional_dim: int,
        n_batches: int,
        n_tissues: int,
        factor_info: dict[str, dict],
        data_cfg: DataConfig | None = None,
    ):
        super().__init__(config, view_input_dims, conditional_dim, n_batches, data_cfg)
        self.hparams["n_tissues"] = n_tissues
        self.hparams["factor_info"] = factor_info
        self.config: DisentangleConfig = config
        self.factor_info = factor_info
        self.layout = BlockLayout(config.block_dims)
        self.register_buffer(
            "cond_mask",
            self._conditional_mask(conditional_dim, n_batches, n_tissues),
            persistent=False,
        )

        self.heads = nn.ModuleDict()
        self.leak_discriminators = nn.ModuleDict()
        for name, factor in config.factors.items():
            out_dim = factor_info[name]["n_classes"]
            if factor.supervised_weight > 0:
                self.heads[name] = _head(factor.block_dim, out_dim)
            if factor.adversarial_weight > 0:
                in_dim = self.layout.complement_dim(name)
                self.leak_discriminators[name] = _head(in_dim, out_dim, in_dim)
        logger.debug("Latent layout: %s", self.layout.dims)

    def _conditional_mask(
        self, conditional_dim: int, n_batches: int, n_tissues: int
    ) -> torch.Tensor:
        """Mask selecting the conditional parts the decoders may read."""
        allowed = set(self.config.decoder_conditionals)
        parts = {
            "model_type": (0, n_batches),
            "tissue": (n_batches, n_batches + n_tissues),
            "mutations": (n_batches + n_tissues, conditional_dim),
        }
        mask = torch.zeros(conditional_dim)
        for part, (start, stop) in parts.items():
            if part in allowed:
                mask[start:stop] = 1.0
        return mask

    # Forward pass -----------------------------------------------------------

    def encode(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        view_embeddings, sample_masks = {}, {}
        for name in self.view_order:
            sample_mask = batch["missing_masks"][name].any(dim=1)
            emb = self.encoders[name](batch["encoder_inputs"][name])
            emb[~sample_mask] = 0.0
            view_embeddings[name] = emb
            sample_masks[name] = sample_mask
        return self.latent_space(view_embeddings, self.view_order, sample_masks)

    def decode(self, z: torch.Tensor, conditionals: torch.Tensor) -> dict:
        cond = conditionals * self.cond_mask
        return {name: self.decoders[name](z, cond) for name in self.view_order}

    def forward(self, batch: dict) -> dict:
        mu, logvar, z = self.encode(batch)
        out = {"x_hat": self.decode(z, batch["conditionals"]), "z": z}
        out.update(mu=mu, logvar=logvar, blocks=self.layout.split(z))
        out["head_outputs"] = {
            name: head(out["blocks"][name]) for name, head in self.heads.items()
        }
        return out

    # Losses -----------------------------------------------------------------

    def _factor_losses(self, batch: dict, out: dict) -> dict[str, torch.Tensor]:
        """Supervised head and CKA leakage penalties per factor."""
        zero = out["z"].new_tensor(0.0)
        losses = {"supervised": zero.clone(), "hsic": zero.clone()}
        for name, factor in self.config.factors.items():
            target = batch["factors"][name]
            mask = batch["factor_masks"][name]
            kind = self.factor_info[name]["kind"]
            n_classes = self.factor_info[name]["n_classes"]
            if factor.supervised_weight > 0:
                losses["supervised"] = losses["supervised"] + (
                    factor.supervised_weight
                    * supervised_loss(out["head_outputs"][name], target, mask, kind)
                )
            if factor.hsic_weight > 0:
                encoded, rows = factor_kernel_input(target, mask, kind, n_classes)
                other = self.layout.complement(out["z"], name)[rows]
                losses["hsic"] = losses["hsic"] + factor.hsic_weight * cka(
                    other, encoded, categorical_y=kind == "categorical"
                )
        return losses

    def _block_independence(self, z: torch.Tensor) -> torch.Tensor:
        names = self.layout.names
        total = z.new_tensor(0.0)
        if self.config.block_hsic_weight <= 0 or len(names) < 2:
            return total
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                total = total + cka(self.layout.block(z, a), self.layout.block(z, b))
        return self.config.block_hsic_weight * total

    def _dip(self, out: dict) -> torch.Tensor:
        if self.config.dip_lambda_od <= 0 and self.config.dip_lambda_d <= 0:
            return out["mu"].new_tensor(0.0)
        return dip_vae_regulariser(
            out["mu"],
            out["logvar"],
            self.config.dip_lambda_od,
            self.config.dip_lambda_d,
        )

    def _compute_losses(self, batch: dict, out: dict) -> dict:
        losses = super()._compute_losses(batch, out)
        losses.update(self._factor_losses(batch, out))
        losses["block_hsic"] = self._block_independence(out["z"])
        losses["dip"] = self._dip(out)
        return losses

    def _leakage_losses(self, batch: dict, z: torch.Tensor, detach: bool) -> dict:
        """Discriminator losses; ``detach`` gives the discriminator-side view."""
        result = {}
        for name, disc in self.leak_discriminators.items():
            inputs = self.layout.complement(z, name)
            logits = disc(inputs.detach() if detach else inputs)
            target, mask = batch["factors"][name], batch["factor_masks"][name]
            if detach or self.config.factors[name].adversarial_mode == "flip":
                result[name] = supervised_loss(logits, target, mask, "categorical")
            else:
                result[name] = confusion_loss(logits, mask)
        return result

    def _encoder_adversarial_term(self, batch: dict, z: torch.Tensor) -> torch.Tensor:
        """Signed leakage term added to the VAE objective."""
        total = z.new_tensor(0.0)
        if self.current_epoch < self.config.adversarial_warmup_epochs:
            return total
        for name, loss in self._leakage_losses(batch, z, detach=False).items():
            factor = self.config.factors[name]
            sign = -1.0 if factor.adversarial_mode == "flip" else 1.0
            total = total + sign * factor.adversarial_weight * loss
        return total

    def _objective(self, losses: dict, kl_weight: float) -> torch.Tensor:
        return (
            losses["recon"]
            + kl_weight * losses["kl"]
            + self.config.contrastive_weight * losses["contrastive"]
            + losses["supervised"]
            + losses["hsic"]
            + losses["block_hsic"]
            + losses["dip"]
        )

    # Lightning hooks ----------------------------------------------------------

    def training_step(self, batch: dict, batch_idx: int):
        optimizers = self.optimizers()
        opt_vae, opt_disc = (
            optimizers if isinstance(optimizers, list) else (optimizers, None)
        )
        out = self.forward(batch)
        losses = self._compute_losses(batch, out)
        kl_weight = self._current_kl_weight()

        disc_total = out["z"].new_tensor(0.0)
        if opt_disc is not None:
            disc_losses = self._leakage_losses(batch, out["z"], detach=True)
            disc_total = sum(disc_losses.values())
            opt_disc.zero_grad()
            self.manual_backward(disc_total)
            self._clip_grads(opt_disc)
            opt_disc.step()

        adv_term = self._encoder_adversarial_term(batch, out["z"])
        total = self._objective(losses, kl_weight) + adv_term
        opt_vae.zero_grad()
        self.manual_backward(total)
        self._clip_grads(opt_vae)
        opt_vae.step()

        self._log_losses("train", total, losses, batch)
        self.log("train/adversarial", adv_term, **self._log_kw(batch))
        self.log("train/disc", disc_total, **self._log_kw(batch))

    def validation_step(self, batch: dict, batch_idx: int):
        out = self.forward(batch)
        losses = self._compute_losses(batch, out)
        total = self._objective(losses, self._current_kl_weight())
        self._log_losses("val", total, losses, batch)

    @staticmethod
    def _log_kw(batch: dict) -> dict:
        return dict(
            on_step=False,
            on_epoch=True,
            batch_size=batch["conditionals"].shape[0],
            sync_dist=True,
        )

    def _log_losses(self, stage: str, total, losses: dict, batch: dict) -> None:
        kw = self._log_kw(batch)
        self.log(f"{stage}/loss", total, prog_bar=True, **kw)
        for key in ("recon", "kl", "supervised", "hsic", "block_hsic", "dip"):
            self.log(f"{stage}/{key}", losses[key], **kw)
        for omic, loss in losses["recon_metrics"]["omic_losses"].items():
            self.log(f"{stage}/recon_{omic}", loss, **kw)

    def configure_optimizers(self):
        vae_params = (
            list(self.encoders.parameters())
            + list(self.decoders.parameters())
            + list(self.latent_space.parameters())
            + list(self.heads.parameters())
        )
        optimizers = [torch.optim.Adam(vae_params, lr=self.config.learning_rate)]
        if len(self.leak_discriminators) > 0:
            optimizers.append(
                torch.optim.Adam(
                    self.leak_discriminators.parameters(),
                    lr=self.config.adv_learning_rate,
                )
            )
        if self.config.lr_scheduler == "step":
            schedulers = [
                torch.optim.lr_scheduler.StepLR(
                    opt, step_size=self.config.lr_step_size, gamma=self.config.lr_gamma
                )
                for opt in optimizers
            ]
            return optimizers, schedulers
        return optimizers

    # Inference ----------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        loader: torch.utils.data.DataLoader,
        force_source_id: int | None = None,
        block_overrides: dict[str, torch.Tensor] | None = None,
    ) -> dict:
        """Encode a loader and decode it, optionally under a counterfactual.

        ``force_source_id`` rewrites the source one-hot seen by the decoders
        (the MOSA conditional-reconstruction mechanism). ``block_overrides``
        replaces whole latent blocks with fixed vectors before decoding, e.g.
        the mean source block of a reference source.
        """
        self.eval()
        collected: dict[str, list] = {"mu": [], "logvar": [], "source_ids": []}
        x_hat: dict[str, list[torch.Tensor]] = {}
        heads: dict[str, list[torch.Tensor]] = {}
        names: list[str] = []
        for batch in loader:
            batch = _batch_to_device(batch, self.device)
            mu, logvar, z = self.encode(batch)
            z = self._apply_overrides(z, block_overrides)
            cond = self._force_source(batch["conditionals"], force_source_id)
            recon = self.decode(z, cond)
            collected["mu"].append(mu.cpu())
            collected["logvar"].append(logvar.cpu())
            collected["source_ids"].append(batch["source_ids"].cpu())
            names.extend(batch["sample_name"])
            for omic, values in recon.items():
                x_hat.setdefault(omic, []).append(values.cpu())
            for name, head in self.heads.items():
                block = self.layout.block(z, name)
                heads.setdefault(name, []).append(head(block).cpu())
        result = {k: torch.cat(v).numpy() for k, v in collected.items()}
        result["z"] = result["mu"]
        result["x_hat"] = {k: torch.cat(v).numpy() for k, v in x_hat.items()}
        result["head_outputs"] = {k: torch.cat(v).numpy() for k, v in heads.items()}
        result["sample_names"] = names
        return result

    def _apply_overrides(self, z, overrides: dict[str, torch.Tensor] | None):
        if not overrides:
            return z
        z = z.clone()
        for name, vector in overrides.items():
            z[:, self.layout.slices[name]] = vector.to(z).expand(z.shape[0], -1)
        return z

    def _force_source(self, conditionals: torch.Tensor, source_id: int | None):
        if source_id is None:
            return conditionals
        if not 0 <= source_id < self.n_batches:
            raise ValueError(f"source_id {source_id} outside [0, {self.n_batches})")
        cond = conditionals.clone()
        cond[:, : self.n_batches] = 0.0
        cond[:, source_id] = 1.0
        return cond

    def export_config(self) -> dict:
        return dataclasses.asdict(self.config)
