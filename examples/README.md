# SMODER Examples

This directory contains example scripts for running SMODER on real datasets and simulated datasets.

The example scripts are provided for reproducibility and adaptation. Large input datasets are not included in this repository. Users should download or prepare the required .h5ad / .h5mu files and pass their own file paths to the scripts.

## Directory structure

examples/
©À©¤©¤ real_data/
©¦   ©¸©¤©¤ scripts for real spatial multi-omics datasets
©¸©¤©¤ simulations/
    ©À©¤©¤ single_reference/
    ©¦   ©¸©¤©¤ simulation scripts using a single-cell RNA reference
    ©À©¤©¤ multi_reference/
    ©¦   ©¸©¤©¤ simulation scripts using paired multi-modal single-cell references
    ©¸©¤©¤ paired_bimodal/
        ©¸©¤©¤ paired RNA+ADT or RNA+ATAC simulation examples

## Notes

- The scripts are examples, not required for importing the smoder Python package.
- Input data paths should be provided by users.
- Output files such as spatial_decon_result.h5ad, trained models, logs, and figures should not be committed to GitHub.
- For installation and tutorials, see the Read the Docs documentation.
