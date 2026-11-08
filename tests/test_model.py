import numpy as np
import torch

from mosa_disentangle.batch_sim import BatchSimConfig, inject_source_effects
from mosa_disentangle.data import read_dataset, write_dataset
from mosa_disentangle.evaluation import evaluate_run
from mosa_disentangle.training import (
    fit_model,
    load_run,
    make_folds,
    predict_dataset,
    save_run,
    split_dataset,
)


def test_fit_predict_evaluate_roundtrip(toy_dataset, configs, tmp_path):
    data_cfg, model_cfg = configs
    val_idx = make_folds(toy_dataset.metadata, n_folds=3, seed=0)[0]
    train, val = split_dataset(toy_dataset, val_idx)
    run = fit_model(train, val, data_cfg, model_cfg, tmp_path)
    assert len(run.history) == model_cfg.num_epochs
    assert np.isfinite(run.history["train/loss"]).all()

    pred = predict_dataset(run, val)
    assert pred.latent.shape == (val.n_samples, model_cfg.joint_latent_dim)
    assert pred.block("tissue").shape[1] == 3
    assert set(pred.recon) == set(val.views)

    ref = run.datamodule.batch_categories.index("cell_line")
    proto = np.zeros(2, dtype=np.float32)
    cf = predict_dataset(run, val, ref, {"model_type": proto})
    assert cf.recon["rna"].shape == val.views["rna"].shape

    scores, matrix, _, _ = evaluate_run(run, train, val)
    assert scores["diverged"] == 0
    assert "nmse_rna" in scores and np.isfinite(scores["nmse_rna"])
    assert set(matrix.index) == {"model_type", "tissue", "emt", "free", "all"}

    save_run(run, tmp_path / "saved")
    loaded = load_run(tmp_path / "saved", train, data_cfg, model_cfg)
    again = predict_dataset(loaded, val)
    np.testing.assert_allclose(
        again.latent.to_numpy(), pred.latent.to_numpy(), atol=1e-4
    )

    with torch.no_grad():  # a blown-up encoder must not crash the evaluation
        for p in run.model.encoders.parameters():
            p.fill_(float("nan"))
    scores, matrix, _, _ = evaluate_run(run, train, val)
    assert scores["diverged"] == 1 and matrix.empty


def test_decoder_conditional_mask(toy_dataset, configs, tmp_path):
    data_cfg, model_cfg = configs
    run = fit_model(toy_dataset, None, data_cfg, model_cfg, tmp_path)
    mask = run.model.cond_mask
    n_batches = run.datamodule.n_batches
    assert torch.all(mask[:n_batches] == 0)  # model_type hidden from decoders
    assert torch.all(mask[n_batches:] == 1)  # tissue visible


def test_batch_simulation_roundtrip(toy_dataset, tmp_path):
    cfg = BatchSimConfig(
        sources=["cell_line", "organoid"],
        confounding=0.3,
        missing_views={"organoid": ["prot"]},
        seed=1,
    )
    observed, truth = inject_source_effects(toy_dataset, cfg)
    organoid = observed.metadata["model_type"] == "organoid"
    assert organoid.any() and not observed.masks["prot"][organoid.to_numpy()].any()
    assert truth.masks["prot"].sum() == toy_dataset.masks["prot"].sum()
    ref = (observed.metadata["model_type"] == "cell_line").to_numpy()
    np.testing.assert_allclose(
        observed.views["rna"][ref], toy_dataset.views["rna"][ref], equal_nan=True
    )
    path = tmp_path / "obs.h5mu"
    write_dataset(observed, path)
    loaded = read_dataset(path)
    assert loaded.view_names == sorted(observed.view_names)
    assert list(loaded.metadata["model_type"]) == list(observed.metadata["model_type"])
