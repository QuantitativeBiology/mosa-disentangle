"""Configuration for the disentangled MOSA variant.

``DisentangleConfig`` extends ``MOSAConfig`` with a description of the factors
of variation to isolate. Each factor may own a latent block and may be tied to
that block by a supervised head, pushed out of the other blocks by a leakage
discriminator, or decorrelated from them with an HSIC/CKA penalty.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mosa.config import check_unknown_keys
from mosa.errors import ConfigError
from mosa.models.mosa.config import MOSAConfig

FACTOR_KINDS = ("categorical", "continuous")
ADVERSARIAL_MODES = ("flip", "confusion")
CONDITIONAL_PARTS = ("model_type", "tissue", "mutations")
FREE_BLOCK = "free"


@dataclass
class FactorConfig:
    """How one factor of variation is handled in the latent space.

    Attributes
    ----------
    name
        Factor identifier, also used as the latent block name.
    column
        ``.obs`` column holding the factor labels or values.
    kind
        ``categorical`` (classification head) or ``continuous`` (regression).
    block_dim
        Width of the dedicated latent block; 0 means no block, in which case
        the factor can only be regularised against the whole latent space.
    supervised_weight
        Weight of the head predicting the factor from its own block.
    adversarial_weight
        Weight of the leakage discriminator predicting the factor from the
        other blocks (whole latent space when ``block_dim == 0``).
    adversarial_mode
        ``flip`` maximises the discriminator loss (as the MOSA adversary);
        ``confusion`` pushes the discriminator output towards uniform.
    hsic_weight
        Weight of the CKA penalty between the factor and the other blocks.
    """

    name: str
    column: str
    kind: str = "categorical"
    block_dim: int = 0
    supervised_weight: float = 0.0
    adversarial_weight: float = 0.0
    adversarial_mode: str = "flip"
    hsic_weight: float = 0.0

    def __post_init__(self):
        for attr in ("supervised_weight", "adversarial_weight", "hsic_weight"):
            value = getattr(self, attr)
            if not isinstance(value, float):
                object.__setattr__(self, attr, float(value))
        if self.kind not in FACTOR_KINDS:
            raise ConfigError(
                f"Factor '{self.name}': kind must be one of {FACTOR_KINDS}, "
                f"got '{self.kind}'"
            )
        if self.block_dim < 0:
            raise ConfigError(f"Factor '{self.name}': block_dim must be >= 0")
        if self.adversarial_mode not in ADVERSARIAL_MODES:
            raise ConfigError(
                f"Factor '{self.name}': adversarial_mode must be one of "
                f"{ADVERSARIAL_MODES}, got '{self.adversarial_mode}'"
            )
        if self.kind == "continuous" and self.adversarial_weight > 0:
            raise ConfigError(
                f"Factor '{self.name}': adversarial leakage penalties are only "
                "defined for categorical factors; use hsic_weight instead"
            )
        if self.supervised_weight > 0 and self.block_dim == 0:
            raise ConfigError(
                f"Factor '{self.name}': a supervised head needs block_dim > 0"
            )
        if self.name == FREE_BLOCK:
            raise ConfigError(f"'{FREE_BLOCK}' is reserved for the free block")

    @property
    def has_block(self) -> bool:
        return self.block_dim > 0


@dataclass
class DisentangleConfig(MOSAConfig):
    """MOSA hyperparameters plus the latent block layout and its regularisers.

    ``joint_latent_dim`` is derived as ``free_dim`` plus the block widths, so
    the free block absorbs whatever the factors do not explain.
    """

    factors: dict[str, FactorConfig] = field(default_factory=dict)
    free_dim: int = 32
    # CKA penalty between every pair of latent blocks (including the free one).
    block_hsic_weight: float = 0.0
    # DIP-VAE-II regulariser on the aggregate posterior (0 disables).
    dip_lambda_od: float = 0.0
    dip_lambda_d: float = 0.0
    # Which parts of the MOSA conditional vector the decoders may see.
    decoder_conditionals: list[str] = field(
        default_factory=lambda: list(CONDITIONAL_PARTS)
    )
    # Epochs before the leakage penalties act on the encoder.
    adversarial_warmup_epochs: int = 0

    @classmethod
    def from_yaml_dict(cls, raw: dict) -> DisentangleConfig:
        raw = dict(raw)
        factors = raw.pop("factors", {}) or {}
        parsed = {}
        for name, fcfg in factors.items():
            check_unknown_keys(
                FactorConfig, fcfg, f"model.factors.{name}", ignore={"name"}
            )
            parsed[name] = FactorConfig(name=name, **fcfg)
        raw["factors"] = parsed
        return super().from_yaml_dict(raw)

    def __post_init__(self):
        self.joint_latent_dim = self.free_dim + sum(
            f.block_dim for f in self.factors.values()
        )
        super().__post_init__()
        for attr in ("block_hsic_weight", "dip_lambda_od", "dip_lambda_d"):
            value = getattr(self, attr)
            if not isinstance(value, float):
                object.__setattr__(self, attr, float(value))
        if self.free_dim < 0:
            raise ConfigError(f"free_dim must be >= 0, got {self.free_dim}")
        if self.adv_weight > 0:
            raise ConfigError(
                "adv_weight is not used by the disentangled model; express the "
                "source adversary as a factor with adversarial_weight > 0"
            )
        unknown = set(self.decoder_conditionals) - set(CONDITIONAL_PARTS)
        if unknown:
            raise ConfigError(
                f"decoder_conditionals must be a subset of {CONDITIONAL_PARTS}, "
                f"got {sorted(unknown)}"
            )
        for name, factor in self.factors.items():
            if factor.name != name:
                raise ConfigError(
                    f"Factor key '{name}' does not match its name '{factor.name}'"
                )

    @property
    def block_dims(self) -> dict[str, int]:
        """Ordered block widths, factors first and the free block last."""
        dims = {n: f.block_dim for n, f in self.factors.items() if f.has_block}
        dims[FREE_BLOCK] = self.free_dim
        return dims
