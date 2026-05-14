# Single-reference simulation examples

This folder contains simulation examples that start from a single-cell RNA reference.

Unlike the paired_bimodal and multi_reference examples, this workflow does not assume that paired single-cell second-modality data are already available. Instead, the workflow first generates a synthetic second-modality reference from the single-cell RNA reference, and then uses the generated multi-modal reference to simulate spatial multi-omics data.

## Available scripts

generate_rna_atac.py

Generate a synthetic ATAC/peak-like reference from a single-cell RNA reference, then simulate spatial RNA+ATAC data.

Example usage:

python examples/simulations/single_reference/generate_rna_atac.py `
  --rna-ref path/to/sc_reference.h5ad `
  --cell-type-col cell_type `
  --output-dir outputs/single_reference_rna_atac

## Planned generation scripts

generate_hbc.py
generate_hgm.py
generate_lymph_node.py
generate_mousebrain.py
generate_spleen.py

These scripts may be added later as dataset-specific wrappers around the generic generate_rna_atac.py workflow.

## Planned SMODER run scripts

run_hbc.py
run_hgm.py
run_lymph_node.py
run_mousebrain.py
run_spleen.py

The run scripts will take the generated simulated spatial RNA and second-modality files as input and run SMODER.

## Expected generated files

The generation scripts save files such as:

ref_RNA.h5ad
ref_ATAC.h5ad
simulation_multiomics.h5mu
simulation_rna.h5ad
simulation_atac.h5ad
sampled_cells_info.csv
cell_type_hvgs_info.csv

## Expected SMODER output files

The run scripts save files such as:

spatial_decon_result.h5ad
cell_type_proportions.csv
training_log.txt
trained_models/

Large generated files and model outputs should not be committed to GitHub.
