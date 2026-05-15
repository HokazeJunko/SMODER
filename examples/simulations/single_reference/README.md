# Single-reference simulation examples

This folder contains simulation examples that start from a single-cell RNA reference.

This workflow uses the local helper module `simulate_new.py`.

Unlike the `paired_bimodal` and `multi_reference` examples, this workflow does not assume that paired single-cell second-modality data are already available. Instead, the workflow first generates a synthetic second-modality reference from the single-cell RNA reference, and then uses the generated multi-modal reference to simulate spatial multi-omics data.

## Recommended scripts for reproducing original experiments

The dataset-specific scripts below are migrated from the original single-reference simulation scripts. They are preferred when the goal is to reproduce the original experiments.

### HBC

Generate simulated data:

```bash
python examples/simulations/single_reference/generate_hbc.py \
  --raw-data-path path/to/HBC/reference_data \
  --output-dir outputs/single_reference_hbc
```

Run SMODER:

```bash
python examples/simulations/single_reference/run_hbc.py \
  --sc-rna path/to/HBC/sc.h5ad \
  --st-rna outputs/single_reference_hbc/simulation_rna.h5ad \
  --st-modality2 outputs/single_reference_hbc/simulation_adt.h5ad \
  --output-dir outputs/single_reference_hbc/SMODER
```

### HGM

Generate simulated data:

```bash
python examples/simulations/single_reference/generate_hgm.py \
  --raw-data-path path/to/HGM/reference_data \
  --output-dir outputs/single_reference_hgm
```

Run SMODER:

```bash
python examples/simulations/single_reference/run_hgm.py \
  --sc-rna path/to/HGM/sc.h5ad \
  --st-rna outputs/single_reference_hgm/simulation_rna.h5ad \
  --st-modality2 outputs/single_reference_hgm/simulation_adt.h5ad \
  --output-dir outputs/single_reference_hgm/SMODER
```

### Lymph node

Generate simulated data:

```bash
python examples/simulations/single_reference/generate_lymph_node.py \
  --raw-data-path path/to/lymph_node/reference_data \
  --output-dir outputs/single_reference_lymph_node
```

Run SMODER:

```bash
python examples/simulations/single_reference/run_lymph_node.py \
  --sc-rna path/to/lymph_node/sc.h5ad \
  --st-rna outputs/single_reference_lymph_node/simulation_rna.h5ad \
  --st-modality2 outputs/single_reference_lymph_node/simulation_adt.h5ad \
  --output-dir outputs/single_reference_lymph_node/SMODER
```

### Mouse brain

Generate simulated data:

```bash
python examples/simulations/single_reference/generate_mousebrain.py \
  --raw-data-path path/to/mousebrain/reference_data \
  --output-dir outputs/single_reference_mousebrain
```

Run SMODER:

```bash
python examples/simulations/single_reference/run_mousebrain.py \
  --sc-rna path/to/mousebrain/sc_mousebrain_processed.h5ad \
  --st-rna outputs/single_reference_mousebrain/simulation_rna.h5ad \
  --st-modality2 outputs/single_reference_mousebrain/simulation_atac.h5ad \
  --output-dir outputs/single_reference_mousebrain/SMODER
```

### Spleen

Generate simulated data:

```bash
python examples/simulations/single_reference/generate_spleen.py \
  --raw-data-path path/to/spleen/reference_data \
  --output-dir outputs/single_reference_spleen
```

Run SMODER:

```bash
python examples/simulations/single_reference/run_spleen.py \
  --sc-rna path/to/spleen/sc.h5ad \
  --st-rna outputs/single_reference_spleen/simulation_rna.h5ad \
  --st-modality2 outputs/single_reference_spleen/simulation_adt.h5ad \
  --output-dir outputs/single_reference_spleen/SMODER
```

## Generic template scripts

The following scripts are kept as simplified templates:

```text
generate_rna_atac.py
run_rna_atac.py
```

For reproducing the original experiments, prefer the dataset-specific scripts listed above.

## Expected generated files

The generation scripts save files such as:

```text
ref_RNA.h5ad
ref_ATAC.h5ad or ref_ADT.h5ad
simulation_multiomics.h5mu
simulation_rna.h5ad
simulation_atac.h5ad or simulation_adt.h5ad
sampled_cells_info.csv
cell_type_hvgs_info.csv
```

## Expected SMODER output files

The run scripts save files such as:

```text
spatial_decon_result.h5ad
cell_type_proportions.csv
training_log.txt
trained_models/
```

Large generated files and model outputs should not be committed to GitHub.