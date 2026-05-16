"""Generate a multi-reference RNA+ADT simulated spatial dataset.

This script starts from paired single-cell RNA and ADT reference files,
generates structured spatial spots, and optionally re-samples selected
target regions to create stronger spatial heterogeneity.

Large input files are not included in this repository. Users should provide
their own local paths.

Example
-------
python analysis_scripts/simulations/multi_reference/generate_adt_rna.py `
  --input-dir path/to/reference_data `
  --rna-file humanBrain_rna_top10_celltypes.h5ad `
  --adt-file humanBrain_adt_top10_celltypes.h5ad `
  --cell-type-col celltype `
  --output-dir outputs/multi_reference_adt_rna
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a multi-reference RNA+ADT simulated spatial dataset."
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing paired single-cell RNA and ADT reference .h5ad files.",
    )
    parser.add_argument(
        "--rna-file",
        default="humanBrain_rna_top10_celltypes.h5ad",
        help="RNA reference .h5ad filename inside --input-dir.",
    )
    parser.add_argument(
        "--adt-file",
        default="humanBrain_adt_top10_celltypes.h5ad",
        help="ADT reference .h5ad filename inside --input-dir.",
    )
    parser.add_argument(
        "--cell-type-col",
        default="celltype",
        help="Column in .obs that stores cell-type labels.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for generated simulated data.",
    )

    parser.add_argument("--base-size", type=int, default=12)
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
        "--cell-type-number-pattern",
        default="2,3,2,3,2,3,2,3",
        help="Comma-separated number of cell types assigned to each of the 8 regions.",
    )
    parser.add_argument("--cell-number-nu", type=float, default=25.0)
    parser.add_argument(
        "--balance",
        choices=["balanced", "unbalanced"],
        default="unbalanced",
    )

    parser.add_argument("--seed", type=int, default=221)
    parser.add_argument("--noise-seed", type=int, default=221)
    parser.add_argument("--rna-poisson-scale", type=float, default=15.0)
    parser.add_argument("--adt-poisson-scale", type=float, default=0.0)
    parser.add_argument(
        "--enable-adt-noise",
        action="store_true",
        help="Enable spatial gradient Poisson noise for ADT. By default ADT noise is disabled.",
    )

    parser.add_argument(
        "--target-regions",
        default="40,40,60;200,40,60;40,200,60;200,200,60;120,120,60",
        help="Semicolon-separated target regions formatted as cx,cy,size.",
    )
    parser.add_argument(
        "--extreme-proportion",
        type=float,
        default=0.8,
        help="Dominant cell-type proportion used in second-round target-region sampling.",
    )
    parser.add_argument("--cell-count-min", type=int, default=10)
    parser.add_argument("--cell-count-max", type=int, default=20)
    parser.add_argument(
        "--no-shuffle-target-types",
        action="store_true",
        help="Disable random shuffling of target cell-type order during second-round sampling.",
    )
    parser.add_argument(
        "--disable-second-sampling",
        action="store_true",
        help="Disable target-region second-round sampling.",
    )

    parser.add_argument(
        "--disable-hvg-shuffle",
        action="store_true",
        help="Disable cell-type HVG dilution/shuffling. Enabled by default for this example.",
    )
    parser.add_argument("--hvg-top-n", type=int, default=200)
    parser.add_argument("--hvg-min-disp", type=float, default=0.5)
    parser.add_argument("--hvg-dilution-factor", type=float, default=0.2)

    return parser.parse_args()


def add_simulation_utils_to_path() -> None:
    simulation_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(simulation_root))


def parse_int_pattern(text: str, expected_len: int = 8) -> list[int]:
    values = [int(x.strip()) for x in text.split(",") if x.strip()]
    if len(values) != expected_len:
        raise ValueError(f"Expected {expected_len} integers, got {len(values)}: {text}")
    return values


def parse_target_regions(text: str) -> list[tuple[float, float, float]]:
    regions = []
    if not text.strip():
        return regions

    for item in text.split(";"):
        item = item.strip()
        if not item:
            continue
        parts = [float(x.strip()) for x in item.split(",")]
        if len(parts) != 3:
            raise ValueError(
                f"Invalid target region '{item}'. Expected format: cx,cy,size"
            )
        regions.append((parts[0], parts[1], parts[2]))
    return regions


def build_noise_config(args: argparse.Namespace) -> dict:
    return {
        "rna": {
            "enable_noise": True,
            "poisson_scale": args.rna_poisson_scale,
            "zero_protection": True,
            "spatial_gradient_coeff": 3.0,
        },
        "adt": {
            "enable_noise": args.enable_adt_noise,
            "poisson_scale": args.adt_poisson_scale,
            "zero_protection": True,
            "spatial_gradient_coeff": 3.0,
        },
        "noise_seed": args.noise_seed,
    }


def build_hvg_shuffle_config(args: argparse.Namespace) -> dict:
    return {
        "enable_shuffle": not args.disable_hvg_shuffle,
        "target_modality": ["rna"],
        "spot_proportion": 1.0,
        "hvg_top_n": args.hvg_top_n,
        "hvg_min_disp": args.hvg_min_disp,
        "hvg_dilution_factor": args.hvg_dilution_factor,
        "shuffle_seed": args.noise_seed,
    }


def run_second_sampling(
    simulation_mdata,
    sampled_cells_df_first,
    ref_adata_dict: dict,
    modality_config: dict,
    cell_type_col: str,
    target_regions: list[tuple[float, float, float]],
    extreme_proportion: float,
    cell_count_min: int,
    cell_count_max: int,
    shuffle_target_types: bool,
    seed: int,
):
    import numpy as np
    import pandas as pd
    import scipy.sparse

    import spatial_sim_function as ssf

    ref_mod = list(modality_config.keys())[0]
    coords = simulation_mdata[ref_mod].obsm["spatial"]
    spot_names = list(simulation_mdata[ref_mod].obs_names)

    target_spots = ssf.get_spots_in_regions(coords, target_regions, spot_names)

    simulation_mdata[ref_mod].obs["second_sampling"] = False

    if len(target_spots) == 0:
        print("No target spots found. Falling back to first-round simulation only.")

        all_cell_types = ref_adata_dict[ref_mod].obs[cell_type_col].cat.categories
        proportions = simulation_mdata[ref_mod].obsm["proportions"]
        dominant_indices = np.argmax(proportions, axis=1)
        dominant_types = [all_cell_types[i] for i in dominant_indices]
        simulation_mdata[ref_mod].obs["dominant_cell_type"] = dominant_types

        return simulation_mdata, sampled_cells_df_first

    print(f"Second-round sampling will update {len(target_spots)} target spots.")

    target_indices = [spot_names.index(spot) for spot in target_spots]

    all_cell_types = ref_adata_dict[ref_mod].obs[cell_type_col].cat.categories
    n_cell_types = len(all_cell_types)

    cell_type_to_cells = {
        ct: ref_adata_dict[ref_mod].obs_names[
            ref_adata_dict[ref_mod].obs[cell_type_col] == ct
        ].tolist()
        for ct in all_cell_types
    }

    if shuffle_target_types:
        np.random.seed(seed + 500)
        random_cell_types = np.random.permutation(all_cell_types)
    else:
        random_cell_types = all_cell_types

    sampled_cells_df_second = []

    for spot_order_idx, spot_idx in enumerate(target_indices):
        spot_name = target_spots[spot_order_idx]
        sub_seed = seed + spot_order_idx + 1000
        np.random.seed(sub_seed)
        random.seed(sub_seed)

        cell_count = np.random.randint(cell_count_min, cell_count_max + 1)

        dominant_ct = np.random.choice(random_cell_types)
        dominant_count = int(cell_count * extreme_proportion)
        other_count = cell_count - dominant_count

        other_cts = [ct for ct in random_cell_types if ct != dominant_ct]
        other_selected = np.random.choice(other_cts, size=other_count, replace=True)

        selected_types = [dominant_ct] * dominant_count + list(other_selected)
        np.random.shuffle(selected_types)

        sampled_cells = []
        for ct in set(selected_types):
            ct_count = selected_types.count(ct)
            available_cells = cell_type_to_cells[ct]
            sampled_ct_cells = np.random.choice(
                available_cells,
                size=ct_count,
                replace=len(available_cells) < ct_count,
            )
            sampled_cells.extend(sampled_ct_cells)

        for mod_name in modality_config.keys():
            adata_sim = simulation_mdata[mod_name]
            adata_ref = ref_adata_dict[mod_name]
            exp_matrix = adata_ref[sampled_cells, :].X
            exp_sum = (
                exp_matrix.sum(axis=0).A1
                if scipy.sparse.issparse(exp_matrix)
                else exp_matrix.sum(axis=0)
            )
            exp_sum_2d = exp_sum.reshape(1, -1)

            adata_x_lil = adata_sim.X.tolil()
            adata_x_lil[spot_idx] = exp_sum_2d
            adata_sim.X = adata_x_lil.tocsr()

        cell_type_counts = pd.Series(selected_types).value_counts()

        proportions = np.zeros(n_cell_types)
        for ct in cell_type_counts.index:
            if ct in all_cell_types:
                proportions[all_cell_types.get_loc(ct)] = cell_type_counts[ct] / cell_count

        dominant_type = cell_type_counts.index[0] if not cell_type_counts.empty else "Unknown"

        simulation_mdata[ref_mod].obs.loc[spot_name, "cell_count"] = cell_count
        simulation_mdata[ref_mod].obs.loc[spot_name, "dominant_cell_type"] = dominant_type

        if "proportions" in simulation_mdata[ref_mod].obsm:
            simulation_mdata[ref_mod].obsm["proportions"][spot_idx] = proportions

        sampled_cells_df_second.append(
            {
                "spot_name": spot_name,
                "sampling_round": 2,
                "main_seed": seed,
                "sub_seed": sub_seed,
                "cell_count": cell_count,
                "dominant_cell_type": dominant_type,
                "selected_types": list(set(selected_types)),
                "cell_type_proportions": proportions.tolist(),
                "sampled_cells": sampled_cells,
            }
        )

    simulation_mdata[ref_mod].obs.loc[target_spots, "second_sampling"] = True

    sampled_cells_df_second = pd.DataFrame(sampled_cells_df_second)
    sampled_cells_df = pd.concat(
        [sampled_cells_df_first, sampled_cells_df_second],
        ignore_index=True,
    )

    return simulation_mdata, sampled_cells_df


def main() -> None:
    args = parse_args()

    add_simulation_utils_to_path()

    import muon as mu
    import numpy as np
    import pandas as pd
    import scanpy as sc

    import spatial_sim_function as ssf

    random.seed(args.seed)
    np.random.seed(args.seed)

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    modality_config = {
        "rna": args.rna_file,
        "adt": args.adt_file,
    }

    feature_shuffle_config = build_hvg_shuffle_config(args)
    noise_config = build_noise_config(args)

    print("Loading paired single-cell RNA and ADT reference data...")
    modal_data = {}
    ref_adata_dict = {}
    cell_type_hvgs_dict = {}

    first_modality = list(modality_config.keys())[0]
    ref_cell_types = None

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
        else:
            if adata.n_obs != len(ref_cell_types):
                raise ValueError(
                    f"{mod_name} has {adata.n_obs} cells, but the first modality has {len(ref_cell_types)} cells. "
                    "This multi-reference simulation expects aligned cells across modalities."
                )
            adata.obs[args.cell_type_col] = ref_cell_types

        modal_data[mod_name] = adata
        ref_adata_dict[mod_name] = adata

        if feature_shuffle_config["enable_shuffle"] and mod_name == "rna":
            print(f"Computing cell-type HVGs for {mod_name}...")
            cell_type_hvgs_dict[mod_name] = ssf.compute_cell_type_hvgs(
                adata.copy(),
                args.cell_type_col,
                top_n=feature_shuffle_config["hvg_top_n"],
                min_disp=feature_shuffle_config["hvg_min_disp"],
            )

    print(f"RNA: {modal_data['rna'].n_obs} cells x {modal_data['rna'].n_vars} features")
    print(f"ADT: {modal_data['adt'].n_obs} cells x {modal_data['adt'].n_vars} features")

    mdata = mu.MuData(modal_data)
    print("Reference MuData:", mdata)

    structured_shapes = [x.strip() for x in args.structured_shapes.split(",") if x.strip()]
    cell_type_number = parse_int_pattern(args.cell_type_number_pattern, expected_len=8)

    sim_params = {
        "cell_type_key": args.cell_type_col,
        "num_spots": (args.base_size * 2) ** 2,
        "balance": args.balance,
        "cell_number_mean": [args.cell_number_mean] * 8,
        "cell_number_nu": args.cell_number_nu,
        "cell_type_number": cell_type_number,
        "poisson_noise_scale": 0.0,
        "structured_shapes": structured_shapes,
        "structured_base_size": args.base_size,
        "random_seed": args.seed,
    }

    print("Generating first-round structured spatial simulation...")
    simulation_mdata, sampled_cells_df_first = ssf.generate_spatial_data(
        reference=mdata,
        **sim_params,
    )
    sampled_cells_df_first["sampling_round"] = 1
    print(f"Generated {simulation_mdata.n_obs} simulated spots.")

    if args.disable_second_sampling:
        print("Second-round sampling is disabled.")

        ref_mod = first_modality
        all_cell_types = ref_adata_dict[ref_mod].obs[args.cell_type_col].cat.categories
        proportions = simulation_mdata[ref_mod].obsm["proportions"]
        dominant_indices = np.argmax(proportions, axis=1)
        dominant_types = [all_cell_types[i] for i in dominant_indices]
        simulation_mdata[ref_mod].obs["dominant_cell_type"] = dominant_types
        simulation_mdata[ref_mod].obs["second_sampling"] = False
        sampled_cells_df = sampled_cells_df_first
    else:
        target_regions = parse_target_regions(args.target_regions)
        simulation_mdata, sampled_cells_df = run_second_sampling(
            simulation_mdata=simulation_mdata,
            sampled_cells_df_first=sampled_cells_df_first,
            ref_adata_dict=ref_adata_dict,
            modality_config=modality_config,
            cell_type_col=args.cell_type_col,
            target_regions=target_regions,
            extreme_proportion=args.extreme_proportion,
            cell_count_min=args.cell_count_min,
            cell_count_max=args.cell_count_max,
            shuffle_target_types=not args.no_shuffle_target_types,
            seed=args.seed,
        )

    print("Adding spatial gradient Poisson noise...")
    simulation_mdata = ssf.add_spatial_gradient_poisson_noise(simulation_mdata, noise_config)

    if feature_shuffle_config["enable_shuffle"]:
        print("Applying HVG dilution/shuffling...")
        for mod_name in ["rna"]:
            if mod_name in cell_type_hvgs_dict:
                simulation_mdata = ssf.shuffle_cell_type_hvgs(
                    simulation_mdata,
                    feature_shuffle_config,
                    cell_type_hvgs_dict[mod_name],
                )
            else:
                print(f"Warning: no precomputed HVG information for {mod_name}; skipping.")

    print("Saving simulated data...")

    h5mu_file = output_dir / "simulation_multiomics.h5mu"
    try:
        mu.write_h5mu(h5mu_file, simulation_mdata)
        print(f"Saved MuData: {h5mu_file}")
    except Exception as exc:
        print(f"Warning: failed to save MuData: {exc}")

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
    print("Multi-reference RNA+ADT spatial simulation finished.")
    print(f"Output directory: {output_dir}")
    print(f"Total spots: {simulation_mdata.n_obs}")
    print(f"Second-round sampling disabled: {args.disable_second_sampling}")
    print(f"HVG shuffling enabled: {feature_shuffle_config['enable_shuffle']}")
    print("=" * 80)


if __name__ == "__main__":
    main()


