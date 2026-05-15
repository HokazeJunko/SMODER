# Simulation examples

This directory contains scripts for generating simulated spatial multi-omics datasets and running SMODER on the simulated data.

The simulation workflow usually has two stages:

1. Generate simulated spatial RNA and second-modality data
2. Run SMODER on the generated simulated data

## Important note about simulation helper modules

Each simulation subdirectory keeps its own helper module because the original simulation workflows use different implementations.

- `paired_bimodal/` uses its local `spatial_sim_function.py`
- `multi_reference/` uses its local `spatial_sim_function.py`
- `single_reference/` uses its local `simulate_new.py`

Although some helper files have similar names and overlapping function names, they should not be treated as one shared file.

## Subdirectories

### `single_reference/`

Simulation examples that start from a single-cell RNA reference and generate simulated spatial RNA plus a simulated second modality.

### `multi_reference/`

Simulation examples that start from paired multi-modal single-cell references, such as RNA+ADT or RNA+ATAC.

### `paired_bimodal/`

Paired bimodal simulation examples where both modalities are treated as paired references.

## Important notes

Large generated files are not included in the GitHub repository. Generated `.h5ad`, `.h5mu`, model checkpoints, logs, and result folders should be stored locally and excluded from Git.
