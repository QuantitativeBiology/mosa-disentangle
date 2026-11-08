import torch

from mosa_disentangle.losses import cka, confusion_loss, dip_vae_regulariser


def test_cka_detects_dependence():
    torch.manual_seed(0)
    x = torch.randn(200, 4)
    dependent = x[:, :2] + 0.05 * torch.randn(200, 2)
    independent = torch.randn(200, 2)
    assert cka(x, dependent) > 0.5
    assert cka(x, independent) < 0.2


def test_cka_finite_for_collapsed_block():
    """A near-constant block must give a finite penalty and gradient."""
    x = torch.zeros(64, 3, requires_grad=True)
    y = torch.randn(64, 2)
    value = cka(x + 1e-9 * torch.randn(64, 3), y)
    value.backward()
    assert torch.isfinite(value) and value < 0.5
    assert torch.isfinite(x.grad).all()


def test_cka_categorical_bounds():
    torch.manual_seed(0)
    labels = torch.randint(0, 3, (100,))
    onehot = torch.nn.functional.one_hot(labels, 3).float()
    x = onehot @ torch.randn(3, 5) + 0.1 * torch.randn(100, 5)
    value = cka(x, onehot, categorical_y=True)
    assert 0.0 <= value <= 1.0 and value > 0.5


def test_confusion_loss_minimal_at_uniform():
    uniform = torch.zeros(10, 4)
    peaked = torch.tensor([[10.0, 0, 0, 0]] * 10)
    mask = torch.ones(10, dtype=torch.bool)
    assert confusion_loss(uniform, mask) < confusion_loss(peaked, mask)


def test_dip_zero_for_whitened_posterior():
    mu = torch.randn(4000, 3)
    mu = (mu - mu.mean(0)) / mu.std(0)
    logvar = torch.full_like(mu, -20.0)
    assert dip_vae_regulariser(mu, logvar, 1.0, 1.0) < 0.05
