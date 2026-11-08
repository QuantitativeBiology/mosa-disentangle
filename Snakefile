"""Disentangled MOSA benchmark pipeline.

    snakemake --cores 8 --resources gpu_slots=2        # everything
    snakemake datasets --cores 4                        # only the data
    snakemake runs --cores 8 --resources gpu_slots=2    # train + evaluate
    snakemake report --cores 2                          # tables + notebooks

GPU jobs share the first visible GPU; ``gpu_slots`` bounds how many run at
once (two or three fit comfortably on a 16 GB card).
"""

from pathlib import Path

configfile: "config/pipeline.yaml"

RAW = config["raw_dir"]
PROCESSED = config["processed_dir"]
RESULTS = config["results_dir"]
DATASETS = config["datasets"]
SIMULATIONS = config["simulations"]
VARIANTS = config["variants"]
EXTRA_RUNS = [(e["dataset"], v) for e in config.get("extra_runs", []) for v in e["variants"]]
ALL_VARIANTS = sorted(set(VARIANTS) | {v for _, v in EXTRA_RUNS})
FOLDS = list(range(config["n_folds"]))
SAVE_MODELS = {(m["dataset"], m["variant"]) for m in config.get("save_models", [])}
PYTHON = ".venv/bin/python"
ENV = "OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4"
NOTEBOOKS = ["01_data_overview", "02_benchmark_results", "03_latent_interpretation",
             "04_counterfactual_reconstruction"]


def dataset_path(name):
    return f"{PROCESSED}/{name}.h5mu"


def truth_path(name):
    return f"{PROCESSED}/{name}_truth.h5mu" if DATASETS[name]["truth"] else []


def run_dir(name, variant, fold):
    return f"{RESULTS}/runs/{name}/{variant}/fold{fold}"


rule all:
    input:
        f"{RESULTS}/tables/metrics_summary.csv",
        expand(f"{RESULTS}/notebooks/{{nb}}.ipynb", nb=NOTEBOOKS),


rule datasets:
    input:
        [dataset_path(n) for n in DATASETS],


rule runs:
    input:
        [f"{run_dir(n, v, k)}/metrics.json" for n in DATASETS for v in VARIANTS for k in FOLDS],
        [f"{run_dir(n, v, k)}/metrics.json" for n, v in EXTRA_RUNS for k in FOLDS],


rule report:
    input:
        f"{RESULTS}/tables/metrics_summary.csv",
        expand(f"{RESULTS}/notebooks/{{nb}}.ipynb", nb=NOTEBOOKS),


rule build_cell_lines:
    input:
        f"{RAW}/transcriptomics.csv",
        f"{RAW}/samplesheet.csv",
    output:
        dataset_path("cell_lines"),
    params:
        top=" ".join(f"{k}={v}" for k, v in config["cell_lines"]["top_variance"].items()),
        min_views=config["cell_lines"]["min_views"],
    threads: 4
    shell:
        "{ENV} {PYTHON} scripts/build_dataset.py --name cell_lines --raw-dir {RAW} "
        "--output {output} --min-views {params.min_views} --top-variance {params.top}"


rule build_proteomics_paired:
    input:
        f"{RAW}/proteomics.csv",
        f"{RAW}/proteomics_ccle.csv",
    output:
        dataset_path("proteomics_paired"),
        f"{PROCESSED}/proteomics_paired_truth.h5mu",
    threads: 4
    shell:
        "{ENV} {PYTHON} scripts/build_dataset.py --name proteomics_paired "
        "--raw-dir {RAW} --output {output[0]} --top-variance proteomics=4000"


rule simulate_sources:
    input:
        dataset_path("cell_lines"),
    output:
        dataset_path("{sim}"),
        f"{PROCESSED}/{{sim}}_truth.h5mu",
    wildcard_constraints:
        sim="|".join(SIMULATIONS),
    params:
        args=lambda wc: " ".join(
            f"--{k.replace('_', '-')} {v}"
            for k, v in {**config["simulation_defaults"], **SIMULATIONS[wc.sim]}.items()
        ),
    threads: 4
    shell:
        "{ENV} {PYTHON} scripts/simulate_sources.py --input {input} "
        "--output {output[0]} {params.args}"


rule run_fold:
    input:
        data=lambda wc: dataset_path(wc.dataset),
        truth=lambda wc: truth_path(wc.dataset),
        base=lambda wc: DATASETS[wc.dataset]["base"],
        variant="config/variants/{variant}.yaml",
    output:
        f"{RESULTS}/runs/{{dataset}}/{{variant}}/fold{{fold}}/metrics.json",
        f"{RESULTS}/runs/{{dataset}}/{{variant}}/fold{{fold}}/probe_matrix.csv",
        f"{RESULTS}/runs/{{dataset}}/{{variant}}/fold{{fold}}/latent_val.parquet",
    wildcard_constraints:
        dataset="|".join(DATASETS),
        variant="|".join(ALL_VARIANTS),
        fold="[0-9]+",
    params:
        truth=lambda wc, input: f"--truth {input.truth}" if input.truth else "",
        reference=lambda wc: (
            f"--reference-source {DATASETS[wc.dataset]['reference']}"
            if DATASETS[wc.dataset]["truth"] else ""
        ),
        group=lambda wc: (
            f"--group-col {DATASETS[wc.dataset]['group']}"
            if DATASETS[wc.dataset].get("group") else ""
        ),
        out=lambda wc: run_dir(wc.dataset, wc.variant, wc.fold),
        save=lambda wc: "--save-model" if (wc.dataset, wc.variant) in SAVE_MODELS else "",
    threads: 4
    resources:
        gpu_slots=1,
    log:
        f"{RESULTS}/runs/{{dataset}}/{{variant}}/fold{{fold}}/run.log",
    shell:
        "{ENV} {PYTHON} scripts/run_fold.py --dataset {input.data} {params.truth} "
        "--base-config {input.base} --variant-config {input.variant} "
        "--fold {wildcards.fold} --n-folds {config[n_folds]} --seed {config[seed]} "
        "{params.reference} {params.group} {params.save} --out-dir {params.out} "
        "> {log} 2>&1"


rule aggregate:
    input:
        rules.runs.input,
    output:
        f"{RESULTS}/tables/metrics_long.csv",
        f"{RESULTS}/tables/metrics_summary.csv",
    shell:
        "{ENV} {PYTHON} scripts/aggregate_results.py --runs-dir {RESULTS}/runs "
        "--out-dir {RESULTS}/tables"


rule notebook:
    input:
        f"{RESULTS}/tables/metrics_summary.csv",
        nb="notebooks/{nb}.ipynb",
    output:
        f"{RESULTS}/notebooks/{{nb}}.ipynb",
    wildcard_constraints:
        nb="|".join(NOTEBOOKS),
    threads: 4
    shell:
        "{ENV} {PYTHON} -m papermill {input.nb} {output} --cwd . "
        "-k python3 --log-output 2> /dev/null"
