"""Run SMODER on the Mousebrain H3K27me3 RNA+peak real dataset.

This script runs SMODER on a real Mousebrain RNA+H3K27me3 peak dataset.

Large input files are not included in this repository. Users should provide
their own local paths.

Example
-------
python examples/simulations/paired_bimodal/run_atac_rna.py
  --sc-rna path/to/MouseBrain/sc_mousebrain_processed.h5ad
  --st-rna path/to/MouseBrain/MouseBrain_RNA_modified.h5ad
  --st-atac path/to/MouseBrain/MouseBrain_peak_H3K27me3_modified.h5ad
  --output-dir outputs/mousebrain_h3k27me3
"""

from __future__ import annotations

import argparse
import os
import random
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SMODER on the Mousebrain H3K27me3 RNA+peak real dataset."
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
        "--st-atac",
        required=True,
        help="Path to the spatial H3K27me3 peak .h5ad file.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for SMODER output files.",
    )

    parser.add_argument(
        "--ref-celltype-col",
        default="annotation_1",
        help="Column in scRNA obs that stores cell-type labels.",
    )
    parser.add_argument(
        "--sample-id-col",
        default="sample",
        help="Column in scRNA obs that stores sample IDs.",
    )

    parser.add_argument("--log-fc", type=float, default=1.25)
    parser.add_argument(
        "--top-n-filter",
        action="store_true",
        help="Enable top-n marker gene filtering. By default, top-n filtering is disabled for this example.",
    )
    parser.add_argument("--top-n", type=int, default=200)
    parser.add_argument("--no-select-info-genes", action="store_true")

    parser.add_argument("--k-spatial", type=int, default=8)
    parser.add_argument("--k-feature", type=int, default=8)
    parser.add_argument("--pca-n-components-rna", type=int, default=256)
    parser.add_argument("--modal2-target-dim", type=int, default=256)

    parser.add_argument("--method", type=int, default=2)
    parser.add_argument("--embd-dim", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--weight-nb-loss", type=float, default=1.0)
    parser.add_argument("--weight-recon-loss", type=float, default=0.0001)
    parser.add_argument("--weight-consistency", type=float, default=1.0)
    parser.add_argument("--weight-spatial", type=float, default=1e-5)

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--device",
        default=None,
        help="Device for training, for example 'cuda:0' or 'cpu'. If omitted, CUDA is used when available.",
    )
    parser.add_argument("--no-model-save", action="store_true")

    return parser.parse_args()


def set_random_seed(seed: int) -> None:
    import numpy as np
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
        "st_adt_path": str(Path(args.st_atac).expanduser().resolve()),
        "output_dir": str(output_dir),
        "model_save_dir": str(model_save_dir),
    }


def build_params(args: argparse.Namespace) -> dict:
    import torch

    return {
        "ref_celltype_col": args.ref_celltype_col,
        "sample_id_col": args.sample_id_col,
        "log_FC": args.log_fc,
        "top_n_filter": args.top_n_filter,
        "top_n": args.top_n,
        "do_select_info_genes": not args.no_select_info_genes,
        "ct_select": None,
        "modal2_type": "peak",
        "K_spatial": args.k_spatial,
        "K_feature": args.k_feature,
        "pca_n_components_rna": args.pca_n_components_rna,
        "modal2_target_dim": args.modal2_target_dim,
        "obsm_name_rna": "X_feat_rna",
        "obsm_name_modal2": "X_feat_modal2",
        "method": args.method,
        "embd_dim": args.embd_dim,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "hidden_dim": args.hidden_dim,
        "weight_loss": [args.weight_nb_loss, args.weight_recon_loss],
        "weight_consistency": args.weight_consistency,
        "weight_spatial": args.weight_spatial,
        "seed": args.seed,
        "model_save": not args.no_model_save,
        "device": args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"),
    }


def load_data(base_config: dict, params: dict):
    import anndata as ad
    import scipy.sparse as sparse

    print(f"=== Data loading ({datetime.now().strftime('%H:%M:%S')}) ===")

    print("Loading single-cell RNA reference...")
    adata_sc = ad.read_h5ad(base_config["sc_rna_path"])
    print(f"  scRNA shape: {adata_sc.shape}")
    print(f"  scRNA matrix type: {type(adata_sc.X)}, sparse={sparse.issparse(adata_sc.X)}")

    print("Loading spatial RNA...")
    adata_st_rna = ad.read_h5ad(base_config["st_rna_path"])
    if not adata_st_rna.var_names.is_unique:
        adata_st_rna.var_names_make_unique()
    print(f"  spatial RNA shape: {adata_st_rna.shape}")

    print("Loading spatial H3K27me3 peak...")
    adata_st_atac = ad.read_h5ad(base_config["st_adt_path"])
    print(f"  spatial ATAC shape: {adata_st_atac.shape}")

    all_celltypes = adata_sc.obs[params["ref_celltype_col"]].unique().tolist()
    params["ct_select"] = all_celltypes if params["ct_select"] is None else params["ct_select"]
    print(f"  selected cell types: {len(params['ct_select'])}")

    ref_dict = {"modal1": adata_sc}
    smo_dict = {"modal1": adata_st_rna, "modal2": adata_st_atac}

    return ref_dict, smo_dict, params


def init_model_and_preprocess(ref_dict: dict, smo_dict: dict, params: dict):
    from smoder.models.deconvolution import SpaMultiDecon_two_modals

    print(f"=== Model initialization and preprocessing ({datetime.now().strftime('%H:%M:%S')}) ===")

    model = SpaMultiDecon_two_modals(
        ref_adata_dict=ref_dict,
        smo_adata_dict=smo_dict,
        nn_para_dict={
            "epochs": params["epochs"],
            "learning_rate": params["learning_rate"],
            "device": params["device"],
            "hidden_dim": params["hidden_dim"],
            "weight_loss": params["weight_loss"],
            "weight_consistency": params["weight_consistency"],
            "weight_spatial": params["weight_spatial"],
            "seed": params["seed"],
            "pca_n_components_rna": params["pca_n_components_rna"],
            "modal2_target_dim": params["modal2_target_dim"],
        },
        modal2_type=params["modal2_type"],
    )

    preprocess_kwargs = {
        "ref_celltype_col": params["ref_celltype_col"],
        "sample_id_col": params["sample_id_col"],
        "ct_select": params["ct_select"],
        "log_FC": params["log_FC"],
        "do_select_info_genes": params["do_select_info_genes"],
    }

    try:
        model.preprocess(
            **preprocess_kwargs,
            top_n_filter=params["top_n_filter"],
            top_n=params["top_n"],
        )
    except TypeError:
        model.preprocess(**preprocess_kwargs)

    model.obsm_name_rna = params["obsm_name_rna"]
    model.obsm_name_modal2 = params["obsm_name_modal2"]

    print("Preprocessing finished.")
    print(f"  basis matrix shape: {model.ref_adata_dict['basis_matrix'].shape}")
    print(f"  spatial RNA shape after preprocessing: {model.smo_adata_dict['modal1'].shape}")
    print(f"  spatial ATAC shape after preprocessing: {model.smo_adata_dict['modal2'].shape}")

    return model


def feature_engineering_and_graph_build(model, params: dict):
    print(f"=== Feature engineering and graph construction ({datetime.now().strftime('%H:%M:%S')}) ===")

    obsm_name_rna = params["obsm_name_rna"]
    obsm_name_modal2 = params["obsm_name_modal2"]

    model.run_feature_engineering_and_mapping(
        obsm_name_rna=obsm_name_rna,
        obsm_name_modal2=obsm_name_modal2,
    )
    model.feature_engineering_done = True

    adata_rna = model.smo_adata_dict["modal1"]
    adata_atac = model.smo_adata_dict["modal2"]

    print("Constructing spatial graph...")
    model.create_spatialgraph(
        obsm_spatial="spatial",
        K_neighbors=params["K_spatial"],
    )
    model.smo_adata_dict["modal2"].uns["spatial_graph"] = model.smo_adata_dict["modal1"].uns["spatial_graph"]

    print("Constructing feature graphs...")
    model.create_featuregraph(
        obsm_name=obsm_name_rna,
        K_neighbors=params["K_feature"],
        name="rna",
    )
    model.create_featuregraph(
        obsm_name=obsm_name_modal2,
        K_neighbors=params["K_feature"],
        name="adt",
    )

    dim_rna = adata_rna.obsm[obsm_name_rna].shape[1]
    dim_modal2 = adata_atac.obsm[obsm_name_modal2].shape[1]

    print(f"  RNA feature dimension: {dim_rna}")
    print(f"  ATAC feature dimension: {dim_modal2}")

    return model, dim_rna, dim_modal2


def train_model(model, params: dict, base_config: dict):
    import torch

    print(f"=== Model training ({datetime.now().strftime('%H:%M:%S')}, device={params['device']}) ===")

    obsm_name_rna = params["obsm_name_rna"]
    obsm_name_modal2 = params["obsm_name_modal2"]

    adata_result, cross_fusion = model.train(
        embd_dim=params["embd_dim"],
        method=params["method"],
        name1="rna",
        name2="adt",
        obsm_name_rna=obsm_name_rna,
        obsm_name_adt=obsm_name_modal2,
        plot=True,
        model_save=params["model_save"],
        model_save_dir=base_config["model_save_dir"],
    )

    with torch.no_grad():
        cross_fusion.eval()

        device = params["device"]
        rna_feat = torch.tensor(
            model.smo_adata_dict["modal1"].obsm[obsm_name_rna].copy(),
            dtype=torch.float32,
            device=device,
        )
        atac_feat = torch.tensor(
            model.smo_adata_dict["modal2"].obsm[obsm_name_modal2].copy(),
            dtype=torch.float32,
            device=device,
        )
        spatial_graph = torch.tensor(
            model.smo_adata_dict["modal1"].uns["spatial_graph"],
            dtype=torch.long,
            device=device,
        )

        rna_feature_graph = (
            torch.tensor(
                model.smo_adata_dict["modal1"].uns["rna"],
                dtype=torch.long,
                device=device,
            )
            if params["method"] == 2
            else None
        )
        modal2_feature_graph = (
            torch.tensor(
                model.smo_adata_dict["modal2"].uns["adt"],
                dtype=torch.long,
                device=device,
            )
            if params["method"] == 2
            else None
        )

        if params["method"] == 1:
            rna_encoder = cross_fusion.encode_modal1(spatial_graph, None, rna_feat).cpu().numpy()
            peak_encoder = cross_fusion.encode_modal2(spatial_graph, None, atac_feat).cpu().numpy()
        else:
            rna_encoder = cross_fusion.encode_modal1(spatial_graph, rna_feature_graph, rna_feat).cpu().numpy()
            peak_encoder = cross_fusion.encode_modal2(spatial_graph, modal2_feature_graph, atac_feat).cpu().numpy()

    adata_result.obsm["rna_encoder"] = rna_encoder
    adata_result.obsm["peak_encoder"] = peak_encoder

    return adata_result


def clean_anndata_for_save(adata):
    import pandas as pd

    illegal_chars = {
        "/": "_",
        "\\": "_",
        ":": "_",
        "*": "_",
        "?": "_",
        '"': "_",
        "<": "_",
        ">": "_",
        "|": "_",
    }

    def replace_illegal(text):
        if pd.isna(text):
            return text
        text_str = str(text)
        for illegal, legal in illegal_chars.items():
            text_str = text_str.replace(illegal, legal)
        return text_str

    adata.obs.columns = [replace_illegal(col) for col in adata.obs.columns]
    adata.obs.index = [replace_illegal(idx) for idx in adata.obs.index]
    adata.var.columns = [replace_illegal(col) for col in adata.var.columns]
    adata.var.index = [replace_illegal(idx) for idx in adata.var.index]
    adata.obsm = {replace_illegal(k): v for k, v in adata.obsm.items()}
    adata.uns = {replace_illegal(k): v for k, v in adata.uns.items()}

    return adata


def save_results(adata_result, model, params: dict, base_config: dict, dim_rna: int, dim_modal2: int) -> None:
    import pandas as pd

    print(f"=== Saving results ({datetime.now().strftime('%H:%M:%S')}) ===")

    adata_to_save = adata_result.copy()

    adata_rna = model.smo_adata_dict["modal1"]
    adata_atac = model.smo_adata_dict["modal2"]

    if "spatial" in adata_rna.obsm:
        adata_to_save.obsm["spatial"] = adata_rna.obsm["spatial"].copy()
    if params["obsm_name_rna"] in adata_rna.obsm:
        adata_to_save.obsm[f"rna_{params['obsm_name_rna']}"] = adata_rna.obsm[params["obsm_name_rna"]].copy()
    if "X_pca_rna" in adata_rna.obsm:
        adata_to_save.obsm["rna_X_pca"] = adata_rna.obsm["X_pca_rna"].copy()

    if params["obsm_name_modal2"] in adata_atac.obsm:
        adata_to_save.obsm[f"peak_{params['obsm_name_modal2']}"] = adata_atac.obsm[params["obsm_name_modal2"]].copy()
    if "X_lsi_peak" in adata_atac.obsm:
        adata_to_save.obsm["peak_X_lsi"] = adata_atac.obsm["X_lsi_peak"].copy()

    if "spatial_graph" in adata_rna.uns:
        adata_to_save.uns["spatial_graph"] = adata_rna.uns["spatial_graph"].copy()
    if "rna" in adata_rna.uns:
        adata_to_save.uns["rna_feature_graph"] = adata_rna.uns["rna"].copy()
    if "adt" in adata_atac.uns:
        adata_to_save.uns["peak_feature_graph"] = adata_atac.uns["adt"].copy()
    if "lsi" in adata_atac.uns:
        adata_to_save.uns["peak_lsi_params"] = adata_atac.uns["lsi"].copy()

    adata_to_save.uns["bimodal_integration_info"] = {
        "modal1_type": "rna",
        "modal2_type": "peak",
        "integration_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "obsm_keys": list(adata_to_save.obsm.keys()),
        "uns_keys": list(adata_to_save.uns.keys()),
    }

    adata_to_save = clean_anndata_for_save(adata_to_save)

    cell_type_names = model.ref_adata_dict["basis_matrix"].index.tolist()
    proportion_df = pd.DataFrame(
        adata_to_save.obsm["cell_type_proportions"],
        index=adata_to_save.obs.index,
        columns=cell_type_names,
    ).reset_index().rename(columns={"index": "spot_name"})

    proportion_path = Path(base_config["output_dir"]) / "cell_type_proportions.csv"
    proportion_df.to_csv(proportion_path, index=False)

    adata_path = Path(base_config["output_dir"]) / "spatial_decon_result.h5ad"
    adata_to_save.write_h5ad(adata_path, compression="gzip")

    log_path = Path(base_config["output_dir"]) / "training_log.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("# SMODER Mousebrain H3K27me3 RNA+peak real-data run log\n")
        f.write(f"Analysis time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\n## Input files\n")
        f.write(f"- scRNA reference: {Path(base_config['sc_rna_path']).name}\n")
        f.write(f"- spatial RNA: {Path(base_config['st_rna_path']).name}\n")
        f.write(f"- spatial ATAC: {Path(base_config['st_adt_path']).name}\n")
        f.write("\n## Model parameters\n")
        for key, val in params.items():
            f.write(f"- {key}: {val}\n")
        f.write("\n## Output dimensions\n")
        f.write(f"- RNA feature dimension: {dim_rna}\n")
        f.write(f"- ATAC feature dimension: {dim_modal2}\n")
        f.write(f"- number of spots: {adata_to_save.n_obs}\n")
        f.write(f"- number of cell types: {len(cell_type_names)}\n")

    print(f"Saved cell-type proportions: {proportion_path}")
    print(f"Saved integrated AnnData result: {adata_path}")
    print(f"Saved training log: {log_path}")


def main() -> None:
    args = parse_args()

    set_random_seed(args.seed)
    base_config = build_base_config(args)
    params = build_params(args)

    print("Running SMODER on the Mousebrain H3K27me3 RNA+peak real dataset")
    print(f"scRNA reference: {base_config['sc_rna_path']}")
    print(f"spatial RNA:     {base_config['st_rna_path']}")
    print(f"spatial ATAC:    {base_config['st_adt_path']}")
    print(f"output dir:      {base_config['output_dir']}")
    print(f"device:          {params['device']}")

    ref_dict, smo_dict, params = load_data(base_config, params)
    model = init_model_and_preprocess(ref_dict, smo_dict, params)
    model, dim_rna, dim_modal2 = feature_engineering_and_graph_build(model, params)
    adata_result = train_model(model, params, base_config)
    save_results(adata_result, model, params, base_config, dim_rna, dim_modal2)

    print("SMODER Mousebrain H3K27me3 RNA+peak run finished successfully.")


if __name__ == "__main__":
    main()




