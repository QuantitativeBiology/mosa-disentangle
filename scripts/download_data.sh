#!/usr/bin/env bash
# Download the MOSA cell line tables from figshare (doi 10.6084/m9.figshare.24420580).
set -euo pipefail
dest=${1:-data/raw/figshare_24420580}
mkdir -p "$dest"
cd "$dest"
while IFS=: read -r name id; do
  [ -s "$name" ] || curl -sL -o "$name" "https://ndownloader.figshare.com/files/$id"
done <<'LIST'
samplesheet.csv:42859714
samplesheet_cancercell.csv:42859705
model_list_20230505.csv:42859702
crisprcas9_institute.csv:42859732
growth_rate_20220907.csv:42859693
drugresponse_drug_targets.csv:42859726
mutations_summary_20230202.csv:42859753
metabolomics.csv:42859717
drugresponse.csv:42859735
cnv_summary_20230303_matrix.csv:42859696
proteomics.csv:42859756
proteomics_ccle.csv:42859765
crisprcas9.csv:42859771
methylation.csv:42859777
transcriptomics.csv:42859780
LIST
ls -la
