"""Run SMODER on the Mousebrain H3K27ac example dataset.

This script is a public example wrapper around the packaged Mousebrain H3K27ac
pipeline. Large input data files are not included in this repository. Users
should provide their own local paths to the required .h5ad files.

Example
-------
python examples/real_data/run_mousebrain_h3k27ac.py `
  --sc-rna path/to/sc_mousebrain_processed.h5ad `
  --st-rna path/to/RNA.h5ad `
  --st-modality2 path/to/peak.h5ad `
  --output-dir outputs/mousebrain_h3k27ac
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SMODER on a Mousebrain H3K27ac RNA + peak dataset."
    )

    parser.add_argument(
        "--sc-rna",
        required=True,
        help="Path to the single-cell RNA reference .h5ad file.",
    )
    parser.add_argument(
        "--st-rna",
        required=True,
        help="Path to the spatial RNA .h5ad file.",
    )
    parser.add_argument(
        "--st-modality2",
        required=True,
        help="Path to the second spatial modality .h5ad file, such as H3K27ac peak data.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for SMODER output files.",
    )

    parser.add_argument(
        "--ref-celltype-col",
        default=None,
        help="Column name in scRNA obs that stores cell-type labels. If omitted, the packaged default is used.",
    )
    parser.add_argument(
        "--sample-id-col",
        default=None,
        help="Column name in scRNA obs that stores sample IDs. If omitted, the packaged default is used.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs. If omitted, the packaged default is used.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate. If omitted, the packaged default is used.",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=None,
        help="Hidden dimension of the GCN model. If omitted, the packaged default is used.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device for training, for example 'cuda:0' or 'cpu'. If omitted, CUDA is used when available.",
    )
    parser.add_argument(
        "--no-model-save",
        action="store_true",
        help="Disable saving trained model checkpoints.",
    )

    return parser.parse_args()


def set_random_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def build_base_config(args: argparse.Namespace) -> dict:
    output_dir = Path(args.output_dir).expanduser().resolve()
    model_save_dir = output_dir / "trained_models"

    output_dir.mkdir(parents=True, exist_ok=True)
    model_save_dir.mkdir(parents=True, exist_ok=True)

    return {
        "sc_rna_path": str(Path(args.sc_rna).expanduser().resolve()),
        "st_rna_path": str(Path(args.st_rna).expanduser().resolve()),
        "st_adt_path": str(Path(args.st_modality2).expanduser().resolve()),
        "output_dir": str(output_dir),
        "model_save_dir": str(model_save_dir),
    }


def build_params(args: argparse.Namespace) -> dict:
    import torch

    from smoder.pipelines.mousebrain_h3k27ac import set_analysis_params

    params = set_analysis_params()

    # Mousebrain H3K27ac uses peak data as the second modality.
    params["modal2_type"] = "peak"

    params["seed"] = args.seed
    params["device"] = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    params["model_save"] = not args.no_model_save

    if args.ref_celltype_col is not None:
        params["ref_celltype_col"] = args.ref_celltype_col
    if args.sample_id_col is not None:
        params["sample_id_col"] = args.sample_id_col
    if args.epochs is not None:
        params["epochs"] = args.epochs
    if args.learning_rate is not None:
        params["learning_rate"] = args.learning_rate
    if args.hidden_dim is not None:
        params["hidden_dim"] = args.hidden_dim

    return params


def main() -> None:
    args = parse_args()

    from smoder.pipelines.mousebrain_h3k27ac import (
        load_and_show_data,
        init_model_and_preprocess,
        feature_engineering_and_graph_build,
        train_model,
        save_results,
    )

    set_random_seed(args.seed)
    base_config = build_base_config(args)
    params = build_params(args)

    print("Running SMODER Mousebrain H3K27ac example")
    print(f"scRNA reference: {base_config['sc_rna_path']}")
    print(f"spatial RNA:     {base_config['st_rna_path']}")
    print(f"second modality: {base_config['st_adt_path']}")
    print(f"output dir:      {base_config['output_dir']}")
    print(f"device:          {params['device']}")

    ref_dict, smo_dict, params = load_and_show_data(base_config, params)
    model = init_model_and_preprocess(ref_dict, smo_dict, params)
    model, dim_rna, dim_modal2 = feature_engineering_and_graph_build(model, params)
    adata_result = train_model(model, params, dim_rna, dim_modal2, base_config)
    save_results(adata_result, model, params, base_config, dim_rna, dim_modal2)

    print("SMODER Mousebrain H3K27ac example finished successfully.")


if __name__ == "__main__":
    main()
