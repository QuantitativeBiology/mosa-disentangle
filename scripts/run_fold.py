"""Train and evaluate one model variant on one cross-validation fold.

Outputs in ``--out-dir``: metrics.json, probe_matrix.csv, latent_train.parquet,
latent_val.parquet, history.csv and (optionally) model.pt.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import limit_threads, parser, setup_logging  # noqa: E402

limit_threads()

import torch  # noqa: E402

from mosa_disentangle.data import read_dataset  # noqa: E402
from mosa_disentangle.evaluation import evaluate_run  # noqa: E402
from mosa_disentangle.training import (  # noqa: E402
    build_configs,
    fit_model,
    load_experiment,
    make_folds,
    save_run,
    split_dataset,
)


def main() -> None:
    setup_logging()
    torch.set_num_threads(4)
    p = parser(__doc__)
    p.add_argument("--dataset", required=True, help="h5mu with observed data")
    p.add_argument("--truth", default=None, help="h5mu with batch-free ground truth")
    p.add_argument("--base-config", required=True)
    p.add_argument("--variant-config", required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--n-folds", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--reference-source", default=None)
    p.add_argument("--group-col", default=None, help="obs column for grouped folds")
    p.add_argument("--max-epochs", type=int, default=None, help="override")
    p.add_argument("--save-model", action="store_true")
    p.add_argument("--out-dir", required=True)
    opts = p.parse_args()

    raw = load_experiment(opts.base_config, opts.variant_config)
    raw["data"]["path"] = opts.dataset
    raw["model"]["random_seed"] = opts.seed + opts.fold
    if opts.max_epochs is not None:
        raw["model"]["num_epochs"] = opts.max_epochs
    data_cfg, model_cfg = build_configs(raw)

    dataset = read_dataset(opts.dataset, data_cfg.views)
    truth = read_dataset(opts.truth, data_cfg.views) if opts.truth else None
    folds = make_folds(dataset.metadata, opts.n_folds, opts.seed, group=opts.group_col)
    train, val = split_dataset(dataset, folds[opts.fold])
    truth_val = truth.subset(folds[opts.fold]) if truth is not None else None

    out_dir = Path(opts.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    run = fit_model(train, val, data_cfg, model_cfg, out_dir)
    fit_seconds = time.perf_counter() - started

    scores, matrix, pred_train, pred_val = evaluate_run(
        run, train, val, truth_val, opts.reference_source
    )
    scores.update(
        {
            "dataset": Path(opts.dataset).stem,
            "variant": Path(opts.variant_config).stem,
            "fold": opts.fold,
            "n_train": train.n_samples,
            "n_val": val.n_samples,
            "joint_latent_dim": model_cfg.joint_latent_dim,
            "fit_seconds": fit_seconds,
        }
    )
    with open(out_dir / "metrics.json", "w") as handle:
        json.dump(scores, handle, indent=2)
    matrix.to_csv(out_dir / "probe_matrix.csv")
    pred_train.latent.join(train.metadata).to_parquet(out_dir / "latent_train.parquet")
    pred_val.latent.join(val.metadata).to_parquet(out_dir / "latent_val.parquet")
    run.history.to_csv(out_dir / "history.csv", index=False)
    if opts.save_model:
        save_run(run, out_dir)


if __name__ == "__main__":
    main()
