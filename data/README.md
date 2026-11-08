# Data provenance

Nothing under `data/` is versioned except this file. Everything is rebuilt by
the Snakemake pipeline from the public deposit of the MOSA paper.

## Raw tables (`data/raw/figshare_24420580/`)

Downloaded from the figshare deposit *Synthetic augmentation of cancer cell
line multi-omic datasets using unsupervised deep learning: Cancer Cell Lines
Data* (doi 10.6084/m9.figshare.24420580; Cai et al., Nat Commun 2024). The
download command is in `scripts/download_data.sh`. Files used:

| File | Content | Orientation |
|---|---|---|
| `transcriptomics.csv` | RNA-seq log expression, 15,320 genes x 1,072 lines | features x samples |
| `proteomics.csv` | Sanger DIA-MS protein intensities, 6,671 proteins x 949 lines | features x samples |
| `proteomics_ccle.csv` | CCLE TMT proteomics, 12,196 proteins x 375 lines (CCLE identifiers) | features x samples |
| `methylation.csv` | Promoter methylation beta values, 14,608 genes x 922 lines | features x samples |
| `metabolomics.csv` | 225 metabolites x 928 lines | features x samples |
| `drugresponse.csv` | GDSC1/GDSC2 ln(IC50), 819 drugs x 995 lines | features x samples |
| `crisprcas9.csv` | Integrated Sanger+Broad CRISPR gene effects, 17,486 genes x 537 lines | features x samples |
| `cnv_summary_20230303_matrix.csv` | Copy-number categories (-2..2), 1,398 lines x 777 genes | samples x features |
| `samplesheet.csv` | Cell Model Passports annotation (tissue, cancer type, MSI, ...) | |
| `samplesheet_cancercell.csv` | Proteomics cohort sheet with the published EMT score | |
| `mutations_summary_20230202.csv` | Driver mutations, long format | |
| `growth_rate_20220907.csv` | Growth rates | |
| `crisprcas9_institute.csv` | Which institute screened each line (Sanger/Broad) | |

Everything is a cell line (`samplesheet.csv` lists 74 organoids, but they are
present only in the copy-number table). Of the 375 CCLE proteomics lines, 362
carry Sanger identifiers and 290 of those were also measured by Sanger: the
only real paired technical factor in the public data.

## Processed datasets (`data/processed/`)

| File | Built by | Description |
|---|---|---|
| `cell_lines.h5mu` | `scripts/build_dataset.py --name cell_lines` | 7 views, lines with >= 2 views, `.obs` with `model_type`, `tissue`, `emt` (transcriptomic EMT score, missing for haematopoietic lines), `emt_all_lines`, `emt_proteomics` (published), `msi_status`, `mutation_*` |
| `sim_cXX.h5mu`, `sim_cXX_truth.h5mu` | `scripts/simulate_sources.py` | Cell lines with a simulated 3-level technical factor (confounding XX) and the batch-free ground truth |
| `proteomics_paired.h5mu`, `proteomics_paired_truth.h5mu` | `scripts/build_dataset.py --name proteomics_paired` | One proteomics view, two real laboratories (`model_type` = sanger / ccle); rows are (line, laboratory), `cell_line` groups the pairs; the truth file holds the other laboratory's measurement of the same line |

The simulated sources are called `cell_line` (reference, untouched), `organoid`
(no proteomics or metabolomics) and `tumour` (no CRISPR, drug response or
metabolomics) to mimic the view availability of the internal MOSA cohort.
