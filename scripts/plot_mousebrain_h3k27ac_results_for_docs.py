import os
import shutil

import scanpy as sc

from smoder.visualization import (
    plot_all_cell_type_proportions,
    plot_cell_type_proportion_panel,
    plot_embedding_spatial_clustering,
    plot_individual_cell_type_heatmaps,
    plot_reconstruction_heatmaps,
)

PROJECT_ROOT = "/data/xiongsc/projects/SMODER"

SPATIAL_RESULT = "/data/baiyh/keyan/SpaMultiDecon/Mousebrain_H3K27ac_result/SMODER_f3/spatial_decon_result.h5ad"

RECON_DIR = os.path.join(PROJECT_ROOT, "outputs", "mousebrain_H3K27ac_reconstruction_for_docs")
RNA_RECON = os.path.join(RECON_DIR, "RNA_recon.h5ad")
EPIGENOMICS_RECON = os.path.join(RECON_DIR, "ATAC_recon.h5ad")

OUT_DIR = os.path.join(PROJECT_ROOT, "docs", "_static", "results", "mousebrain_h3k27ac")
CELLTYPE_DIR = os.path.join(OUT_DIR, "cell_type_proportions")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CELLTYPE_DIR, exist_ok=True)


def copy_training_loss_if_available():
    candidates = [
        os.path.join(PROJECT_ROOT, "outputs", "mousebrain_H3K27ac_run2", "trained_models", "training_losses.png"),
        os.path.join(PROJECT_ROOT, "outputs", "mousebrain_H3K27ac", "trained_models", "training_losses.png"),
        os.path.join(PROJECT_ROOT, "outputs", "mousebrain_H3K27ac_run3", "trained_models", "training_losses.png"),
    ]

    for src in candidates:
        if os.path.exists(src):
            dst = os.path.join(OUT_DIR, "mousebrain_training_losses.png")
            shutil.copy2(src, dst)
            print(f"Copied: {src} -> {dst}")
            return

    print("No training_losses.png found.")


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

    # Mousebrain metadata columns are the first 9 obs columns.
    obs_start_col = 9

    plot_all_cell_type_proportions(
        adata,
        out_path=os.path.join(OUT_DIR, "mousebrain_cell_type_proportion_all59.png"),
        obs_start_col=obs_start_col,
        ncols=6,
        title="Spatial heatmaps of cell-type proportions",
    )
    print("Saved all-cell-type proportion panel.")

    plot_cell_type_proportion_panel(
        adata,
        out_path=os.path.join(OUT_DIR, "mousebrain_cell_type_proportion_top12.png"),
        obs_start_col=obs_start_col,
        top_n=12,
        ncols=4,
        title="Spatial heatmaps of selected cell-type proportions",
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
        out_path=os.path.join(OUT_DIR, "mousebrain_embedding_spatial_clustering.png"),
        embedding_key="embedding",
        method="leiden",
        resolution=0.6,
        n_neighbors=15,
    )
    print("Saved embedding-based spatial clustering.")

    copy_training_loss_if_available()

    plot_reconstruction_if_available(
        RNA_RECON,
        prefix="mousebrain_rna",
        title_prefix="Denoised RNA expression",
    )

    plot_reconstruction_if_available(
        EPIGENOMICS_RECON,
        prefix="mousebrain_epigenomics",
        title_prefix="Denoised gene-level epigenomic signal",
    )

    print("All available Mousebrain H3K27ac figures have been generated.")


if __name__ == "__main__":
    main()
