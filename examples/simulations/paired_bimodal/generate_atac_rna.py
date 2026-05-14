"""Generate a paired RNA+ATAC simulated spatial multi-omics dataset.

This script uses paired single-cell RNA and ATAC reference data to generate
simulated spatial RNA and ATAC data.

Large input files are not included in this repository. Users should provide
their own local paths.

Example
-------
python examples/simulations/paired_bimodal/generate_atac_rna.py `
  --input-dir path/to/reference_data `
  --rna-file human_melanoma_RNA_f2ab.h5ad `
  --atac-file human_melanoma_ATAC_f2ab.h5ad `
  --cell-type-col cell_type `
  --output-dir outputs/simulated_atac_rna
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a paired RNA+ATAC simulated spatial multi-omics dataset."
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing the paired single-cell RNA and ATAC reference .h5ad files.",
    )
    parser.add_argument(
        "--rna-file",
        default="human_melanoma_RNA_f2ab.h5ad",
        help="RNA reference .h5ad filename inside --input-dir.",
    )
    parser.add_argument(
        "--atac-file",
        default="human_melanoma_ATAC_f2ab.h5ad",
        help="ATAC reference .h5ad filename inside --input-dir.",
    )
    parser.add_argument(
        "--cell-type-col",
        default="cell_type",
        help="Column in .obs that stores cell-type labels.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for generated simulated data.",
    )

    parser.add_argument(
        "--base-size",
        type=int,
        default=12,
        help="Base size of each structured spatial quadrant. Total spots will be (2 * base_size)^2.",
    )
    parser.add_argument(
        "--structured-shapes",
        default="squares,rings,corners,checkers",
        help="Comma-separated spatial pattern names for the four quadrants.",
    )
    parser.add_argument(
        "--cell-number-mean",
        type=int,
        default=10,
        help="Mean cell number per spot for each of the 8 regions.",
    )
    parser.add_argument(
        "--cell-type-number",
        type=int,
        default=2,
        help="Number of cell types assigned to each of the 8 regions.",
    )
    parser.add_argument(
        "--cell-number-nu",
        type=float,
        default=25.0,
        help="Dispersion parameter for Conway-Maxwell-Poisson cell-count sampling.",
    )
    parser.add_argument(
        "--balance",
        choices=["balanced", "unbalanced"],
        default="unbalanced",
        help="Cell sampling mode.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for spatial simulation.",
    )
    parser.add_argument(
        "--noise-seed",
        type=int,
        default=1234,
        help="Random seed for spatial noise.",
    )
    parser.add_argument(
        "--rna-poisson-scale",
        type=float,
        default=15.0,
        help="Poisson noise scale for RNA.",
    )
    parser.add_argument(
        "--atac-poisson-scale",
        type=float,
        default=0.0,
        help="Poisson noise scale for ATAC.",
    )
    parser.add_argument(
        "--enable-atac-noise",
        action="store_true",
        help="Enable spatial gradient Poisson noise for ATAC. By default ATAC noise is disabled.",
    )

    parser.add_argument(
        "--enable-hvg-shuffle",
        action="store_true",
        help="Enable cell-type HVG dilution/shuffling for RNA.",
    )
    parser.add_argument(
        "--hvg-top-n",
        type=int,
        default=200,
        help="Number of HVGs per cell type when HVG shuffling is enabled.",
    )
    parser.add_argument(
        "--hvg-min-disp",
        type=float,
        default=0.5,
        help="Minimum dispersion threshold for HVG selection.",
    )
    parser.add_argument(
        "--hvg-dilution-factor",
        type=float,
        default=0.2,
        help="Dilution factor used for HVG shuffling.",
    )

    return parser.parse_args()


def add_simulation_utils_to_path() -> None:
    simulation_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(simulation_root))


def build_noise_config(args: argparse.Namespace) -> dict:
    return {
        "rna": {
            "enable_noise": True,
            "poisson_scale": args.rna_poisson_scale,
            "zero_protection": True,
            "spatial_gradient_coeff": 3.0,
        },
        "atac": {
            "enable_noise": args.enable_atac_noise,
            "poisson_scale": args.atac_poisson_scale,
            "zero_protection": True,
            "spatial_gradient_coeff": 3.0,
        },
        "noise_seed": args.noise_seed,
    }


def build_hvg_shuffle_config(args: argparse.Namespace) -> dict:
    return {
        "enable_shuffle": args.enable_hvg_shuffle,
        "target_modality": ["rna"],
        "spot_proportion": 1.0,
        "hvg_top_n": args.hvg_top_n,
        "hvg_min_disp": args.hvg_min_disp,
        "hvg_dilution_factor": args.hvg_dilution_factor,
        "shuffle_seed": args.noise_seed,
    }


def main() -> None:
    args = parse_args()

    add_simulation_utils_to_path()

    import muon as mu
    import numpy as np
    import pandas as pd
    import scanpy as sc

    import spatial_simulation_utils as ssf

    random.seed(args.seed)
    np.random.seed(args.seed)

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    modality_config = {
        "rna": args.rna_file,
        "atac": args.atac_file,
    }

    feature_shuffle_config = build_hvg_shuffle_config(args)
    noise_config = build_noise_config(args)

    print("Loading paired single-cell RNA and ATAC reference data...")
    modal_data = {}
    cell_type_hvgs_dict = {}

    first_modality = list(modality_config.keys())[0]
    ref_cell_types = None
    all_cell_types = None

    for mod_name, mod_file in modality_config.items():
        file_path = input_dir / mod_file
        adata = sc.read_h5ad(file_path)

        if "counts" in adata.layers:
            adata.X = adata.layers["counts"].copy()
        else:
            print(f"Warning: {mod_name} has no counts layer; using X matrix.")
            adata.X = adata.X.copy()

        if mod_name == first_modality:
            adata.obs[args.cell_type_col] = adata.obs[args.cell_type_col].astype("category")
            ref_cell_types = adata.obs[args.cell_type_col].values
            all_cell_types = adata.obs[args.cell_type_col].cat.categories
        else:
            if adata.n_obs != len(ref_cell_types):
                raise ValueError(
                    f"{mod_name} has {adata.n_obs} cells, but the first modality has {len(ref_cell_types)} cells. "
                    "This paired-bimodal simulation expects aligned cells across modalities."
                )
            adata.obs[args.cell_type_col] = ref_cell_types

        modal_data[mod_name] = adata

        if args.enable_hvg_shuffle and mod_name in feature_shuffle_config["target_modality"]:
            print(f"Computing cell-type HVGs for {mod_name}...")
            cell_type_hvgs_dict[mod_name] = ssf.compute_cell_type_hvgs(
                adata.copy(),
                args.cell_type_col,
                top_n=feature_shuffle_config["hvg_top_n"],
                min_disp=feature_shuffle_config["hvg_min_disp"],
            )

    print(f"RNA: {modal_data['rna'].n_obs} cells x {modal_data['rna'].n_vars} features")
    print(f"ATAC: {modal_data['atac'].n_obs} cells x {modal_data['atac'].n_vars} features")

    mdata = mu.MuData(modal_data)
    print("Reference MuData:", mdata)

    structured_shapes = [x.strip() for x in args.structured_shapes.split(",") if x.strip()]

    sim_params = {
        "cell_type_key": args.cell_type_col,
        "num_spots": (args.base_size * 2) ** 2,
        "balance": args.balance,
        "cell_number_mean": [args.cell_number_mean] * 8,
        "cell_number_nu": args.cell_number_nu,
        "cell_type_number": [args.cell_type_number] * 8,
        "poisson_noise_scale": 0.0,
        "structured_shapes": structured_shapes,
        "structured_base_size": args.base_size,
        "random_seed": args.seed,
    }

    print("Generating structured spatial simulation...")
    simulation_mdata, sampled_cells_df = ssf.generate_spatial_data(
        reference=mdata,
        **sim_params,
    )
    sampled_cells_df["sampling_round"] = 1
    print(f"Generated {simulation_mdata.n_obs} simulated spots.")

    print("Computing dominant cell type for each spot...")
    ref_mod = first_modality
    proportions = simulation_mdata[ref_mod].obsm["proportions"]
    dominant_indices = np.argmax(proportions, axis=1)
    dominant_types = [all_cell_types[i] for i in dominant_indices]
    simulation_mdata[ref_mod].obs["dominant_cell_type"] = dominant_types

    print("Adding spatial gradient Poisson noise...")
    simulation_mdata = ssf.add_spatial_gradient_poisson_noise(simulation_mdata, noise_config)

    if args.enable_hvg_shuffle:
        print("Applying HVG dilution/shuffling...")
        for mod_name in feature_shuffle_config["target_modality"]:
            if mod_name in cell_type_hvgs_dict:
                simulation_mdata = ssf.shuffle_cell_type_hvgs(
                    simulation_mdata,
                    feature_shuffle_config,
                    cell_type_hvgs_dict[mod_name],
                    dominant_cell_type_col="dominant_cell_type",
                )
            else:
                print(f"Warning: no precomputed HVG information for {mod_name}; skipping.")

    print("Saving simulated data...")

    h5mu_file = output_dir / "simulation_multiomics.h5mu"
    mu.write_h5mu(h5mu_file, simulation_mdata)
    print(f"Saved MuData: {h5mu_file}")

    sampled_cells_file = output_dir / "sampled_cells_info.csv"
    sampled_cells_df.to_csv(sampled_cells_file, index=False)
    print(f"Saved sampled-cell information: {sampled_cells_file}")

    for mod_name in modality_config.keys():
        mod_file = output_dir / f"simulation_{mod_name}.h5ad"
        simulation_mdata[mod_name].write_h5ad(mod_file)
        print(f"Saved {mod_name} AnnData: {mod_file}")

    hvgs_info_file = output_dir / "cell_type_hvgs_info.csv"
    hvgs_rows = [
        {"cell_type": ct, "modality": mod, "hvgs": ",".join(hvgs)}
        for mod, ct_hvgs in cell_type_hvgs_dict.items()
        for ct, hvgs in ct_hvgs.items()
    ]
    pd.DataFrame(hvgs_rows, columns=["cell_type", "modality", "hvgs"]).to_csv(
        hvgs_info_file,
        index=False,
    )
    print(f"Saved HVG information: {hvgs_info_file}")

    print("=" * 80)
    print("Paired RNA+ATAC spatial simulation finished.")
    print(f"Output directory: {output_dir}")
    print(f"Total spots: {simulation_mdata.n_obs}")
    print(f"RNA noise scale: {noise_config['rna']['poisson_scale']}")
    print(f"ATAC noise scale: {noise_config['atac']['poisson_scale']}")
    print(f"HVG shuffling enabled: {feature_shuffle_config['enable_shuffle']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
