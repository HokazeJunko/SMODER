# Multi-reference simulation examples

This folder contains simulation examples that start from paired multi-modal single-cell references, such as RNA+ADT or RNA+ATAC.

Compared with the simpler paired_bimodal examples, these scripts include an additional spatial-disruption step. The workflow is:

1. Load paired multi-modal single-cell reference data
2. Generate structured spatial spots
3. Re-sample selected target regions to create stronger spatial heterogeneity
4. Add spatial gradient noise
5. Optionally dilute or shuffle highly variable features
6. Save simulated spatial RNA and second-modality AnnData files
7. Run SMODER on the generated simulated data

## Available scripts

generate_adt_rna.py

Generate a simulated RNA+ADT spatial multi-omics dataset from paired single-cell RNA and ADT references.

Example usage:

python examples/simulations/multi_reference/generate_adt_rna.py `
  --input-dir path/to/reference_data `
  --rna-file humanBrain_rna_top10_celltypes.h5ad `
  --adt-file humanBrain_adt_top10_celltypes.h5ad `
  --cell-type-col celltype `
  --output-dir outputs/multi_reference_adt_rna

generate_atac_rna.py

Generate a simulated RNA+ATAC spatial multi-omics dataset from paired single-cell RNA and ATAC references.

Example usage:

python examples/simulations/multi_reference/generate_atac_rna.py `
  --input-dir path/to/reference_data `
  --rna-file human_melanoma_RNA_f2ab.h5ad `
  --atac-file human_melanoma_ATAC_f2ab.h5ad `
  --cell-type-col cell_type `
  --output-dir outputs/multi_reference_atac_rna

## Planned scripts

run_adt_rna.py

Run SMODER on a simulated RNA+ADT dataset.

run_atac_rna.py

Run SMODER on a simulated RNA+ATAC dataset.

## Expected generated files

The generation scripts save files such as:

simulation_multiomics.h5mu
simulation_rna.h5ad
simulation_adt.h5ad or simulation_atac.h5ad
sampled_cells_info.csv
cell_type_hvgs_info.csv

## Expected SMODER output files

The run scripts save files such as:

spatial_decon_result.h5ad
cell_type_proportions.csv
training_log.txt
trained_models/

Large generated files and model outputs should not be committed to GitHub.
