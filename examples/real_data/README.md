# Real data examples

This directory contains example scripts for running SMODER on real spatial multi-omics datasets.

The original internal scripts used absolute server paths. In the public version, all input and output paths should be passed through command-line arguments.

## Available scripts

run_mousebrain_h3k27ac.py

This script runs SMODER on a Mousebrain H3K27ac RNA + peak dataset. It is a public wrapper around the packaged Mousebrain H3K27ac pipeline.

Example usage:

python examples/real_data/run_mousebrain_h3k27ac.py `
  --sc-rna path/to/sc_mousebrain_processed.h5ad `
  --st-rna path/to/RNA.h5ad `
  --st-modality2 path/to/peak.h5ad `
  --output-dir outputs/mousebrain_h3k27ac

run_hbc.py

This script runs SMODER on a real HBC RNA + ADT dataset.

Example usage:

python examples/real_data/run_hbc.py `
  --sc-rna path/to/HBC/sc.h5ad `
  --st-rna path/to/HBC/HBC_RNA.h5ad `
  --st-adt path/to/HBC/HBC_ADT.h5ad `
  --output-dir outputs/hbc

run_mousebrain_h3k27me3.py

This script runs SMODER on a real Mousebrain RNA + H3K27me3 peak dataset.

Example usage:

python examples/real_data/run_mousebrain_h3k27me3.py `
  --sc-rna path/to/MouseBrain/sc_mousebrain_processed.h5ad `
  --st-rna path/to/MouseBrain/MouseBrain_RNA_modified.h5ad `
  --st-atac path/to/MouseBrain/MouseBrain_peak_H3K27me3_modified.h5ad `
  --output-dir outputs/mousebrain_h3k27me3

run_mouse_embryo.py

This script runs SMODER on a real Mouse embryo RNA + ATAC dataset.

Example usage:

python examples/real_data/run_mouse_embryo.py `
  --sc-rna path/to/mouse_embryo/mouse_embryo_sc_filtered.h5ad `
  --st-rna path/to/mouse_embryo/MouseEmbryo25um_RNA_updated.h5ad `
  --st-atac path/to/mouse_embryo/MouseEmbryo_peak_ATAC_updated.h5ad `
  --output-dir outputs/mouse_embryo

## Required inputs

Each real-data example generally requires:

1. a single-cell RNA reference .h5ad file
2. a spatial RNA .h5ad file
3. a second spatial modality .h5ad file, such as ADT/protein or ATAC/peak data
4. an output directory

## Expected outputs

The scripts save SMODER outputs such as:

spatial_decon_result.h5ad
cell_type_proportions.csv
training_log.txt
trained_models/

Generated outputs should not be committed to GitHub.
