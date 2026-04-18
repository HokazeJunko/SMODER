import os


def get_mousebrain_h3k27ac_base_config(run_name="mousebrain_H3K27ac_run4"):
    output_dir = f"/data/xiongsc/projects/SMODER/outputs/{run_name}"
    model_save_dir = os.path.join(output_dir, "trained_models")

    base_config = {
        "sc_rna_path": "/data/xiongsc/data/SMODER/mousebrain_H3K27ac/sc_mousebrain_processed.h5ad",
        "st_rna_path": "/data/xiongsc/data/SMODER/mousebrain_H3K27ac/RNA.h5ad",
        "st_adt_path": "/data/xiongsc/data/SMODER/mousebrain_H3K27ac/peak.h5ad",
        "output_dir": output_dir,
        "model_save_dir": model_save_dir,
    }

    os.makedirs(base_config["output_dir"], exist_ok=True)
    os.makedirs(base_config["model_save_dir"], exist_ok=True)
    return base_config


def get_mousebrain_h3k27ac_params():
    params = {
        "ref_celltype_col": "annotation_1",
        "sample_id_col": "sample",
        "log_FC": 1.25,
        "do_select_info_genes": True,
        "ct_select": None,
        "modal2_type": "peak",
        "K_spatial": 8,
        "K_feature": 8,
        "pca_n_components_rna": 256,
        "modal2_target_dim": 256,
        "obsm_name_rna": "X_feat_rna",
        "obsm_name_modal2": "X_feat_modal2",
        "method": 2,
        "embd_dim": 50,
        "epochs": 10,
        "learning_rate": 2e-3,
        "hidden_dim": 512,
        "weight_loss": [1, 0.001],
        "weight_consistency": 1,
        "weight_spatial": 1e-5,
        "seed": 1,
        "model_save": False,
        "device": "cuda:0",
    }
    return params