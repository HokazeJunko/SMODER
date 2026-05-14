# Paired bimodal simulation examples

This folder contains paired RNA+ADT and RNA+ATAC simulation examples.

## Available scripts

generate_adt_rna.py

Generate a paired RNA+ADT simulated spatial multi-omics dataset from aligned single-cell RNA and ADT reference files.

Example usage:

python examples/simulations/paired_bimodal/generate_adt_rna.py `
  --input-dir path/to/reference_data `
  --rna-file humanBrain_rna_top10_celltypes_f2ab.h5ad `
  --adt-file humanBrain_adt_top10_celltypes_f2ab.h5ad `
  --cell-type-col celltype `
  --output-dir outputs/simulated_adt_rna

generate_atac_rna.py

Generate a paired RNA+ATAC simulated spatial multi-omics dataset from aligned single-cell RNA and ATAC reference files.

Example usage:

python examples/simulations/paired_bimodal/generate_atac_rna.py `
  --input-dir path/to/reference_data `
  --rna-file human_melanoma_RNA_f2ab.h5ad `
  --atac-file human_melanoma_ATAC_f2ab.h5ad `
  --cell-type-col cell_type `
  --output-dir outputs/simulated_atac_rna

The generation scripts save files such as:

simulation_multiomics.h5mu
simulation_rna.h5ad
simulation_adt.h5ad or simulation_atac.h5ad
sampled_cells_info.csv
cell_type_hvgs_info.csv

## Planned scripts

run_adt_rna.py
run_atac_rna.py
