import numpy as np
import pandas as pd

from mosa_disentangle.metrics import knn_label_entropy, nmse, paired_retrieval
from mosa_disentangle.training import block_columns


def test_block_columns_ignore_metadata_with_shared_prefix():
    columns = [
        "emt_0",
        "emt_1",
        "emt_proteomics",
        "model_type_0",
        "model_type_true",
        "tissue",
    ]
    assert block_columns(columns, "emt") == ["emt_0", "emt_1"]
    assert block_columns(columns, "model_type") == ["model_type_0"]


def test_nmse_of_mean_prediction_is_one():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(50, 10))
    mask = np.ones_like(x, dtype=bool)
    assert abs(nmse(x, np.full_like(x, x.mean()), mask) - 1.0) < 1e-6
    assert nmse(x, x, mask) == 0.0


def test_knn_entropy_extremes():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(200, 3))
    mixed = pd.Series(rng.choice(["a", "b"], 200))
    separated = pd.Series(np.where(z[:, 0] > 0, "a", "b"))
    z = z + np.where(z[:, [0]] > 0, 5.0, -5.0)  # move the two groups apart
    assert knn_label_entropy(z, mixed, k=15) > 0.8
    assert knn_label_entropy(z, separated, k=15) < 0.3


def test_paired_retrieval_perfect_pairs():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(30, 4))
    z = np.vstack([base, base + 0.01 * rng.normal(size=base.shape)])
    groups = pd.Series([f"g{i}" for i in range(30)] * 2)
    batches = pd.Series(["a"] * 30 + ["b"] * 30)
    result = paired_retrieval(z, groups, batches)
    assert result["paired_top1"] == 1.0 and result["paired_n"] == 30
