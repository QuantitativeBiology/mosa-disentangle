import pytest
from mosa.errors import ConfigError

from mosa_disentangle.config import DisentangleConfig, FactorConfig
from mosa_disentangle.training import build_configs, deep_update


def test_latent_dim_is_derived(configs):
    _, model_cfg = configs
    assert model_cfg.joint_latent_dim == 4 + 2 + 3 + 1
    assert list(model_cfg.block_dims) == ["model_type", "tissue", "emt", "free"]


def test_adv_weight_rejected():
    with pytest.raises(ConfigError):
        DisentangleConfig(adv_weight=0.1)


def test_continuous_adversary_rejected():
    with pytest.raises(ConfigError):
        FactorConfig(
            name="emt",
            column="emt",
            kind="continuous",
            block_dim=1,
            adversarial_weight=1.0,
        )


def test_unknown_factor_key_rejected(raw_config):
    raw = deep_update(raw_config, {"model": {"factors": {"emt": {"bogus": 1}}}})
    with pytest.raises(ConfigError):
        build_configs(raw)


def test_deep_update_does_not_mutate():
    base = {"a": {"b": 1, "c": 2}}
    merged = deep_update(base, {"a": {"b": 5}})
    assert merged == {"a": {"b": 5, "c": 2}}
    assert base["a"]["b"] == 1
