# Simulation examples

This directory contains scripts for generating simulated spatial multi-omics datasets and running SMODER on the simulated data.

The simulation workflow usually has two stages:

1. Generate simulated spatial RNA and second-modality data
2. Run SMODER on the generated simulated data

## Shared utility module

spatial_simulation_utils.py

This file contains shared simulation utilities, including:

- spatial pattern generation
- structured region assignment
- spot-level cell sampling
- simulated spatial coordinate generation
- noise and perturbation utilities
- helper functions for simulated multi-omics data

The dataset-specific generation scripts will import this utility module.

## Subdirectories

single_reference/

Simulation examples that start from a single-cell RNA reference and generate simulated spatial RNA plus a simulated second modality.

multi_reference/

Simulation examples that start from paired multi-modal single-cell references, such as RNA+ADT or RNA+ATAC.

paired_bimodal/

Paired bimodal simulation examples where both modalities are treated as paired references.

## Important notes

Large generated files are not included in the GitHub repository. Generated .h5ad, .h5mu, model checkpoints, logs, and result folders should be stored locally and excluded from Git.
