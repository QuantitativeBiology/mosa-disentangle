# mosa-disentangle

Disentangled representation learning for [MOSA](https://github.com/QuantitativeBiology/mosa),
the conditional multi-view VAE for bulk cancer multi-omics. The latent space is
organised into blocks, one per factor of variation (technical source, tissue,
EMT) plus a free block, and several strategies to keep each factor inside its
block are benchmarked against the released MOSA on held-out reconstruction,
block-wise probing, batch mixing and counterfactual batch-removed
reconstruction.

This repository is the starting basis for an MSc thesis; `docs/RESEARCH_NOTES.md`
records what was tested, what worked, what did not, open problems and the
literature to read first.

## Layout

```
config/            pipeline, dataset and model-variant YAML files
  datasets/        base configuration per dataset (views, block layout, training)
  variants/        one file per disentanglement strategy (overrides the base)
src/mosa_disentangle/
  config.py        DisentangleConfig / FactorConfig (extends MOSAConfig)
  model.py         DisentangledVAE: block layout, heads, discriminators, counterfactual decoding
  losses.py        HSIC/CKA, DIP-VAE, supervised and confusion losses
  datamodule.py    MOSA data module + encoded factor targets in every batch
  data.py          dataset construction from the figshare tables, MuData I/O
  batch_sim.py     semi-synthetic technical factor with ground truth
  training.py      fit / predict / save / load harness, cross-validation splits
  evaluation.py    all metrics for one fold (reconstruction, probes, mixing, counterfactuals)
  metrics.py       metric implementations
  plotting.py      figure helpers for the notebooks
scripts/           thin command-line entry points used by the Snakefile
notebooks/         01 data, 02 benchmark, 03 latent interpretation, 04 counterfactuals
tests/             unit and smoke tests (pytest)
Snakefile          download -> datasets -> simulations -> runs -> tables -> notebooks
results/           tables, executed notebooks (run outputs are git-ignored)
data/              raw and processed data (git-ignored; provenance in data/README.md)
```

## Setup

The host GPUs run driver 550 (CUDA 12.4); `mosalib` pins `torch==2.14.0`, whose
default wheel targets CUDA 13, so torch is taken from the CUDA 12.6 index.

```bash
conda create -p .venv python=3.12
.venv/bin/python -m pip install mosalib snakemake papermill jupyter ipykernel ruff pytest
.venv/bin/python -m pip install "torch==2.14.0+cu126" --index-url https://download.pytorch.org/whl/cu126
.venv/bin/python -m pip install -e .
.venv/bin/python -m pytest -q            # ~1 min on CPU
```

## Running

```bash
scripts/download_data.sh                                  # ~0.4 GB from figshare
.venv/bin/snakemake --cores 8 --resources gpu_slots=2      # full benchmark
.venv/bin/snakemake runs --cores 8 --resources gpu_slots=2 # training only
.venv/bin/snakemake report --cores 2                       # tables + notebooks
```

A single experiment without Snakemake:

```bash
.venv/bin/python scripts/run_fold.py --dataset data/processed/sim_c05.h5mu \
    --truth data/processed/sim_c05_truth.h5mu \
    --base-config config/datasets/cell_lines.yaml \
    --variant-config config/variants/full.yaml --fold 0 \
    --reference-source cell_line --out-dir results/runs/sim_c05/full/fold0
```

Each fold directory receives `metrics.json` (one flat dictionary of held-out
scores, plus `diverged = 1` and no other score when the optimisation produced
non-finite latent codes), `probe_matrix.csv`, the latent codes of both folds
as parquet, `history.csv` and, for the runs listed under `save_models`,
`model.pt`. `scripts/aggregate_results.py` turns these into
`results/tables/metrics_long.csv` (one row per fold) and
`metrics_summary.csv` (mean and standard deviation per dataset and variant).

## Datasets

The public MOSA data (figshare 10.6084/m9.figshare.24420580) contains cell
lines only, so the thesis' technical factor (cell line / organoid / tumour) is
emulated with a controlled simulation whose batch-free truth is known
(`sim_c00`, `sim_c05`, `sim_c09`, increasing confounding between pseudo-source
and tissue). A real two-laboratory proteomics dataset (`proteomics_paired`,
Sanger vs CCLE) is the sanity check on a genuine technical effect. See
`data/README.md`. Swapping in the internal cell line + organoid + tumour cohort
only requires an h5mu with `model_type`, `tissue` and `emt` in `.obs`.

## Model variants

| variant | latent | regularisers |
|---|---|---|
| `baseline` | one space | MOSA as released |
| `adversarial` | one space | MOSA source discriminator (loss flipped) |
| `adversarial_confusion` | one space | source discriminator, bounded confusion objective |
| `dip` | one space | DIP-VAE-II (earlier PhenPred attempt) |
| `supervised` | blocks | supervised heads per block |
| `supervised_nocond` | blocks | heads + source adversary, decoders blind to source |
| `hsic` | blocks | heads + CKA independence constraints |
| `midas` | blocks | heads + leakage discriminators (MIDAS-style) |
| `full` | blocks | heads + discriminators + CKA |
| `full_nocond` | blocks | as `full`, confusion mode, decoders blind to source |
| `sweep_adv*_hsic*` | blocks | `full_nocond` with stronger source penalties |

Each factor is declared once in the dataset config (`column`, `kind`,
`block_dim`); a variant only sets `supervised_weight`, `adversarial_weight`,
`hsic_weight` and `decoder_conditionals`. `joint_latent_dim` is derived.
On the public data `supervised_nocond` is the recommended starting point
(no reconstruction cost, best counterfactual and downstream scores); the
CKA variants cost reconstruction without reducing leakage. Results and the
reasoning are in `docs/RESEARCH_NOTES.md`, sections 7 and 8.

## Relationship to the MOSA package

`DisentangledVAE` subclasses `mosa.models.mosa.vae.vae_module.VAE` and
`FactorDataModule` subclasses `MOSADataModule`; MOSA's encoders, fusion,
conditional decoders, preprocessing and reconstruction loss are used
unchanged. Upstreaming the best variant means adding `FactorConfig` fields to
`MOSAConfig`, the factor targets to the data module and the block logic to the
VAE module, following `docs/development/modifying-vae.md` in the MOSA
repository.
