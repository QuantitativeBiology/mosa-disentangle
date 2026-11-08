"""Regularisers used to organise the MOSA latent space into factor blocks.

All functions operate on minibatch tensors and return scalar losses.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def _center(kernel: Tensor) -> Tensor:
    """Double-centre a kernel matrix (H K H)."""
    n = kernel.shape[0]
    h = torch.eye(n, device=kernel.device) - 1.0 / n
    return h @ kernel @ h


CKA_EPS = 1e-6


def gaussian_kernel(x: Tensor, bandwidth: float | None = None) -> Tensor:
    """RBF kernel with the median-distance heuristic when no bandwidth is given.

    The heuristic bandwidth is treated as a constant (detached) so the encoder
    cannot lower the penalty by rescaling a block instead of decorrelating it.
    """
    d2 = torch.cdist(x, x).pow(2)
    if bandwidth is None:
        frozen = d2.detach()
        positive = frozen[frozen > 0]
        median = positive.median() if positive.numel() > 0 else frozen.new_tensor(1.0)
        bandwidth = torch.sqrt(0.5 * median).clamp(min=1e-6)
    return torch.exp(-d2 / (2.0 * bandwidth**2))


def linear_kernel(x: Tensor) -> Tensor:
    return x @ x.T


def hsic(kernel_x: Tensor, kernel_y: Tensor) -> Tensor:
    """Biased HSIC estimator tr(K H L H) / (n - 1)^2 (Gretton et al., 2005)."""
    n = kernel_x.shape[0]
    return torch.trace(_center(kernel_x) @ _center(kernel_y)) / (n - 1) ** 2


def cka(x: Tensor, y: Tensor, categorical_y: bool = False) -> Tensor:
    """Normalised HSIC (centred kernel alignment) in [0, 1].

    ``x`` is a latent block (RBF kernel). ``y`` is either another latent block
    (RBF kernel) or a one-hot / continuous factor encoding, for which a linear
    kernel is used when ``categorical_y`` is True.
    """
    if x.shape[0] < 4:
        return x.new_tensor(0.0)
    kx = gaussian_kernel(x)
    ky = linear_kernel(y) if categorical_y else gaussian_kernel(y)
    hxy = hsic(kx, ky)
    # The epsilon keeps the ratio and its gradient finite when a block has
    # collapsed to a near-constant (HSIC(K, K) -> 0), which otherwise makes
    # strongly weighted penalties diverge.
    denom = torch.sqrt((hsic(kx, kx) + CKA_EPS) * (hsic(ky, ky) + CKA_EPS))
    return hxy / denom


def dip_vae_regulariser(
    mu: Tensor, logvar: Tensor, lambda_od: float, lambda_d: float
) -> Tensor:
    """DIP-VAE-II penalty on the covariance of the aggregate posterior.

    Off-diagonal covariances are pushed to zero and diagonal variances to one
    (Kumar et al., ICLR 2018). This is the regulariser previously explored in
    the PhenPred repository of the MOSA paper.
    """
    if mu.shape[0] < 2:
        return mu.new_tensor(0.0)
    cov_mu = torch.cov(mu.T)
    cov_z = cov_mu + torch.diag(logvar.exp().mean(dim=0))
    diag = torch.diagonal(cov_z)
    off_diag = cov_z - torch.diag(diag)
    return lambda_od * off_diag.pow(2).sum() + lambda_d * (diag - 1.0).pow(2).sum()


def supervised_loss(pred: Tensor, target: Tensor, mask: Tensor, kind: str) -> Tensor:
    """Masked cross-entropy (categorical) or MSE (continuous) for a factor head."""
    if mask.sum() == 0:
        return pred.new_tensor(0.0)
    pred, target = pred[mask], target[mask]
    if kind == "categorical":
        return F.cross_entropy(pred, target.long())
    return F.mse_loss(pred.squeeze(-1), target.float())


def confusion_loss(logits: Tensor, mask: Tensor) -> Tensor:
    """Cross-entropy of the discriminator output against the uniform distribution.

    Minimising it from the encoder side makes every class equally likely, a
    bounded alternative to maximising the discriminator loss.
    """
    if mask.sum() == 0:
        return logits.new_tensor(0.0)
    log_probs = F.log_softmax(logits[mask], dim=1)
    return -log_probs.mean()


def factor_kernel_input(target: Tensor, mask: Tensor, kind: str, n_classes: int):
    """Encode a factor as the matrix fed to the CKA penalty, with its row mask."""
    if kind == "categorical":
        codes = target.long().clamp(min=0)
        return F.one_hot(codes, n_classes).float()[mask], mask
    return target.float().unsqueeze(1)[mask], mask
