# SMODER

**SMODER** is a Python toolkit for spatial multi-omics deconvolution.

It is designed to integrate spatial transcriptomics with an additional spatial modality, such as protein or chromatin accessibility signals, in order to infer cell-type composition across spatial locations.

The current implementation provides a modular workflow for:

- data preprocessing
- feature engineering
- graph construction
- model training
- deconvolution result generation

The package is organized around reusable components for preprocessing, modeling, configuration, and end-to-end pipeline execution.

## Highlights

- Modular Python package structure
- End-to-end runnable spatial multi-omics pipeline
- Support for RNA + second-modality workflows
- Reusable preprocessing, model, and pipeline components
- Tutorial notebook and documentation website support

## Project Status

What is currently available:

- the `smoder` package can be installed and imported
- the current mouse brain H3K27ac pipeline can run end-to-end
- tutorial and API documentation are available through Read the Docs
- output files can be generated successfully, including:
  - `cell_type_proportions.csv`
  - `spatial_decon_result.h5ad`
  - `training_log.txt`

Current recommended execution method:

```bash
python -m smoder.pipelines.mousebrain_h3k27ac
```

## Repository Structure

```text
SMODER/
├── smoder/
│   ├── __init__.py
│   ├── config/
│   ├── models/
│   ├── pipelines/
│   └── preprocessing/
├── data/
├── docs/
├── scripts/
├── tutorials/
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Installation

### Requirements

The current validated environment is:

- Linux
- conda-based Python environment
- Python 3.12

The main tutorial and recommended workflow are currently oriented toward this environment.

### Clone the repository

```bash
git clone <repository_url>
cd SMODER
```

### Create a Python environment

A dedicated environment is recommended.

For example:

```bash
conda create -n smoder python=3.12
conda activate smoder
```

### Install the package

```bash
pip install -e .
```

## Data

The current validated example uses a mouse brain RNA + peak dataset.

Required input files:

- `sc_mousebrain_processed.h5ad`: single-cell reference RNA
- `RNA.h5ad`: spatial RNA
- `peak.h5ad`: spatial peak data

Example directory layout:

```text
data/
└── mousebrain_H3K27ac/
   ├── sc_mousebrain_processed.h5ad
   ├── RNA.h5ad
   └── peak.h5ad
```

Please prepare the required dataset files separately and place them into a local data directory matching the expected structure.

Additional notes on data preparation are provided in `data/README.md` and in the documentation site.

## Quick Start

Run the current validated pipeline:

```bash
python -m smoder.pipelines.mousebrain_h3k27ac
```

Typical outputs include:

- `cell_type_proportions.csv`
- `spatial_decon_result.h5ad`
- `training_log.txt`

## Documentation

Documentation is available through Read the Docs.

It currently includes:

- installation guide
- data preparation guide
- tutorial notebook
- API reference

## Roadmap

Planned next steps include:

- additional runnable pipelines for other datasets
- improved package interfaces
- more complete dataset instructions
- formal PyPI release
- continued documentation improvements

## Citation

If you use SMODER in your work, please cite the corresponding paper or project release once formal citation information becomes available.

## Contact

Maintainer: Shucun Xiong