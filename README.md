# SMODER

**SMODER** is a spatial multi-omics deconvolution toolkit under active development.

SMODER is designed to support spatial multi-omics deconvolution workflows in a modular and reusable Python package structure. The project is currently being refactored from research scripts into a cleaner package-based implementation, with runnable pipelines, reusable preprocessing modules, and extensible model components.

At the current stage, the package has been validated on a mouse brain RNA + peak dataset, and the end-to-end workflow can successfully complete:

- data loading
- preprocessing
- feature engineering
- graph construction
- model training
- result saving

---

## Highlights

- Modular Python package structure
- End-to-end runnable spatial multi-omics pipeline
- Support for RNA + second-modality workflows
- Reusable preprocessing, model, and pipeline components
- Editable installation for active development
- Planned support for tutorials and documentation website deployment

---

## Project Status

SMODER is currently in an active refactoring stage.

### What is already working

- the `smoder` package can be imported successfully
- the current mouse brain H3K27ac pipeline can run end-to-end
- the package can be installed in editable mode
- output files can be generated successfully, including:
  - `cell_type_proportions.csv`
  - `spatial_decon_result.h5ad`
  - `training_log.txt`

### Current recommended execution method

```bash
python -m smoder.pipelines.mousebrain_h3k27ac
```

---

## Repository Structure

```text
SMODER/
├── smoder/
│   ├── __init__.py
│   ├── config/
│   ├── models/
│   ├── pipelines/
│   └── preprocessing/
├── scripts/
├── tutorials/
├── docs/
├── outputs/
├── pyproject.toml
└── README.md
```

### Main package components

- `smoder/models/gnn.py`  
  Neural network building blocks, including GCN encoders and attention-based fusion modules.

- `smoder/models/deconvolution.py`  
  Main deconvolution model implementation, including `SpaMultiDecon_two_modals`.

- `smoder/preprocessing/rna.py`  
  RNA preprocessing utilities for single-cell and spatial transcriptomics data.

- `smoder/preprocessing/modality2.py`  
  Preprocessing utilities for the second modality, including ADT CLR normalization and peak TF-IDF processing.

- `smoder/pipelines/mousebrain_h3k27ac.py`  
  End-to-end pipeline for the current validated mouse brain test dataset.

- `smoder/config/defaults.py`  
  Default configuration utilities for the current pipeline.

---

## Installation

### 1. Clone the repository

```bash
git clone <repository_url>
cd SMODER
```

### 2. Create a Python environment

A dedicated Python environment is recommended.

For example, using conda:

```bash
conda create -n smoder python=3.12
conda activate smoder
```

### 3. Install the package in editable mode

```bash
pip install -e .
```

This installs SMODER in editable mode so that package imports always reflect the latest source code during development.

---

## Input Data

The current validated workflow expects three input files:

- a single-cell reference RNA dataset
- a spatial RNA dataset
- a second spatial modality dataset

A typical dataset layout may look like this:

```text
<data_root>/mousebrain_H3K27ac/
├── sc_mousebrain_processed.h5ad
├── RNA.h5ad
└── peak.h5ad
```

The current validated example uses:

- single-cell reference RNA
- spatial RNA
- spatial peak data

---

## Quick Start

### Run the current validated pipeline

```bash
python -m smoder.pipelines.mousebrain_h3k27ac
```

### Typical outputs

A typical output directory may contain:

```text
<output_dir>/
├── cell_type_proportions.csv
├── spatial_decon_result.h5ad
└── training_log.txt
```

---

## Method Overview

SMODER currently adopts a modular workflow for spatial multi-omics deconvolution, including:

1. preprocessing of single-cell reference RNA data
2. preprocessing of spatial RNA data
3. preprocessing of a second spatial modality
4. feature engineering and dimensionality reduction
5. graph construction for spatial and feature relationships
6. model training and latent representation learning
7. deconvolution result generation and result export

The codebase is structured so that preprocessing, model definition, configuration, and end-to-end pipelines are separated into different modules for easier maintenance and extension.

---

## Development Principles

The current refactoring follows several principles:

- keep the runnable pipeline working during reorganization
- gradually replace legacy top-level scripts with package modules
- separate model code, preprocessing code, pipeline code, and configuration code
- keep execution reproducible while improving project structure
- prepare the codebase for future tutorials and documentation deployment

At this stage, module-based execution is the recommended entry point:

```bash
python -m smoder.pipelines.mousebrain_h3k27ac
```

---

## Documentation Plan

Planned documentation components include:

- tutorial notebooks
- package usage examples
- pipeline-specific instructions
- API-oriented documentation pages
- Sphinx documentation
- Read the Docs deployment

---

## Roadmap

Planned next steps include:

- cleaner package interfaces
- additional runnable pipelines for other datasets
- improved script entry points
- tutorial notebooks
- more complete installation and usage instructions
- documentation website deployment
- further cleanup of legacy top-level scripts

---

## Citation

If you use SMODER in your work, please cite the corresponding paper or project release once the formal citation information becomes available.

*Currently: citation information is not yet finalized.*

---

## Contact

Project contact information can be added here after the repository is prepared for public release.

Example:

- maintainer: `<name>`
- email: `<email>`
- lab / group: `<lab_or_group_name>`

---

## Disclaimer

SMODER is still under active development. Interfaces, file layout, configuration details, and documentation may continue to change during the refactoring process.