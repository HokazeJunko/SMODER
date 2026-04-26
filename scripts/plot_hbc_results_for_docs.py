import os

import scanpy as sc

from smoder.visualization import (
    plot_all_cell_type_proportions,
    plot_cell_type_proportion_panel,
    plot_embedding_spatial_clustering,
    plot_individual_cell_type_heatmaps,
    plot_reconstruction_heatmaps,
)

PROJECT_ROOT = "/data/xiongsc/projects/SMODER"

SPATIAL_RESULT = "/data/baiyh/keyan/SpaMultiDecon/HBC_SMODER_f3/spatial_decon_result.h5ad"

RECON_DIR = os.path.join(PROJECT_ROOT, "outputs", "hbc_reconstruction_for_docs")
RNA_RECON = os.path.join(RECON_DIR, "RNA_recon.h5ad")
ADT_RECON = os.path.join(RECON_DIR, "ADT_recon.h5ad")

OUT_DIR = os.path.join(PROJECT_ROOT, "docs", "_static", "results", "hbc")
CELLTYPE_DIR = os.path.join(OUT_DIR, "cell_type_proportions")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CELLTYPE_DIR, exist_ok=True)


def plot_reconstruction_if_available(recon_path, prefix, title_prefix):
    if not os.path.exists(recon_path):
        print(f"Skip reconstruction plots because file not found: {recon_path}")
        return

    recon_adata = sc.read_h5ad(recon_path)
    plot_reconstruction_heatmaps(
        recon_adata,
        out_dir=OUT_DIR,
        prefix=prefix,
        title_prefix=title_prefix,
    )
    print(f"Saved reconstruction heatmaps from: {recon_path}")


def main():
    print("Reading:", SPATIAL_RESULT)
    adata = sc.read_h5ad(SPATIAL_RESULT)

    print("adata shape:", adata.shape)
    print("obsm keys:", list(adata.obsm.keys()))

    # HBC metadata columns are the first 8 obs columns.
    obs_start_col = 8

    plot_all_cell_type_proportions(
        adata,
        out_path=os.path.join(OUT_DIR, "hbc_cell_type_proportion_all9.png"),
        obs_start_col=obs_start_col,
        ncols=3,
        title="HBC spatial heatmaps of cell-type proportions",
    )
    print("Saved all-cell-type proportion panel.")

    plot_cell_type_proportion_panel(
        adata,
        out_path=os.path.join(OUT_DIR, "hbc_cell_type_proportion_top9.png"),
        obs_start_col=obs_start_col,
        top_n=9,
        ncols=3,
        title="HBC spatial heatmaps of selected cell-type proportions",
    )
    print("Saved selected-cell-type proportion panel.")

    plot_individual_cell_type_heatmaps(
        adata,
        out_dir=CELLTYPE_DIR,
        obs_start_col=obs_start_col,
    )
    print(f"Saved individual cell-type heatmaps to: {CELLTYPE_DIR}")

    plot_embedding_spatial_clustering(
        adata,
        out_path=os.path.join(OUT_DIR, "hbc_embedding_spatial_clustering.png"),
        embedding_key="embedding",
        method="leiden",
        resolution=0.6,
        n_neighbors=15,
    )
    print("Saved embedding-based spatial clustering.")

    plot_reconstruction_if_available(
        RNA_RECON,
        prefix="hbc_rna",
        title_prefix="HBC denoised RNA expression",
    )

    plot_reconstruction_if_available(
        ADT_RECON,
        prefix="hbc_adt",
        title_prefix="HBC denoised ADT signal",
    )

    print("All available HBC figures have been generated.")


if __name__ == "__main__":
    main()
