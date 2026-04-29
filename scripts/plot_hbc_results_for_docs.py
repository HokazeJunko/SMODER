from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from matplotlib.lines import Line2D

from smoder.visualization import (
    plot_all_cell_type_proportions,
    plot_cell_type_proportion_panel,
    plot_individual_cell_type_heatmaps,
    plot_reconstruction_heatmaps,
)


PROJECT_ROOT = Path("/data/xiongsc/projects/SMODER")

SPATIAL_RESULT = Path("/data/baiyh/keyan/SpaMultiDecon/HBC_SMODER_f3/spatial_decon_result.h5ad")

RECON_DIR = PROJECT_ROOT / "outputs" / "hbc_reconstruction_for_docs"
RNA_RECON = RECON_DIR / "RNA_recon.h5ad"
ADT_RECON = RECON_DIR / "ADT_recon.h5ad"

OUT_DIR = PROJECT_ROOT / "docs" / "_static" / "results" / "hbc"
CELLTYPE_DIR = OUT_DIR / "cell_type_proportions"

OUT_DIR.mkdir(parents=True, exist_ok=True)
CELLTYPE_DIR.mkdir(parents=True, exist_ok=True)


def plot_embedding_clustering_with_dot_legend(
    adata,
    out_path,
    embedding_key="embedding",
    cluster_key="smoder_cluster",
    method="leiden",
    resolution=0.6,
    n_neighbors=15,
    point_size=8,
):
    if embedding_key not in adata.obsm:
        raise KeyError(f"adata.obsm['{embedding_key}'] not found.")

    adata_tmp = adata.copy()
    sc.pp.neighbors(adata_tmp, use_rep=embedding_key, n_neighbors=n_neighbors)

    if method.lower() == "leiden":
        sc.tl.leiden(adata_tmp, resolution=resolution, key_added=cluster_key)
        method_label = "Leiden"
    elif method.lower() == "louvain":
        sc.tl.louvain(adata_tmp, resolution=resolution, key_added=cluster_key)
        method_label = "Louvain"
    else:
        raise ValueError("method must be 'leiden' or 'louvain'.")

    labels = adata_tmp.obs[cluster_key].astype(str)

    def sort_key(x):
        try:
            return int(x)
        except ValueError:
            return x

    categories = sorted(labels.unique(), key=sort_key)
    cmap = plt.get_cmap("tab20", len(categories))
    color_map = {cat: cmap(i) for i, cat in enumerate(categories)}
    colors = labels.map(color_map).values

    coords = np.asarray(adata_tmp.obsm["spatial"])
    x = coords[:, 0]
    y = -coords[:, 1]

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    ax.scatter(x, y, c=list(colors), s=point_size, edgecolors="none")

    ax.set_title(f"Spatial clustering based on learned embeddings ({method_label})")
    ax.set_aspect("equal")
    ax.axis("off")

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=color_map[cat],
            markeredgecolor=color_map[cat],
            markersize=7,
            label=str(cat),
        )
        for cat in categories
    ]

    ax.legend(
        handles=handles,
        title="Cluster",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=False,
        borderaxespad=0.0,
        handlelength=0.8,
        handletextpad=0.4,
    )

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return adata_tmp


def plot_reconstruction_if_available(recon_path, prefix, title_prefix):
    if not recon_path.exists():
        print(f"Skip reconstruction plots because file not found: {recon_path}")
        return

    recon_adata = sc.read_h5ad(recon_path)
    plot_reconstruction_heatmaps(
        recon_adata,
        out_dir=str(OUT_DIR),
        prefix=prefix,
        title_prefix=title_prefix,
    )
    print(f"Saved reconstruction heatmaps from: {recon_path}")


def main():
    print("Reading:", SPATIAL_RESULT)
    adata = sc.read_h5ad(SPATIAL_RESULT)

    print("adata shape:", adata.shape)
    print("obsm keys:", list(adata.obsm.keys()))

    obs_start_col = 8

    plot_all_cell_type_proportions(
        adata,
        out_path=str(OUT_DIR / "hbc_cell_type_proportion_all9.png"),
        obs_start_col=obs_start_col,
        ncols=3,
        title="HBC spatial heatmaps of cell-type proportions",
    )
    print("Saved all-cell-type proportion panel.")

    plot_cell_type_proportion_panel(
        adata,
        out_path=str(OUT_DIR / "hbc_cell_type_proportion_top9.png"),
        obs_start_col=obs_start_col,
        top_n=9,
        ncols=3,
        title="HBC spatial heatmaps of selected cell-type proportions",
    )
    print("Saved selected-cell-type proportion panel.")

    plot_individual_cell_type_heatmaps(
        adata,
        out_dir=str(CELLTYPE_DIR),
        obs_start_col=obs_start_col,
    )
    print(f"Saved individual cell-type heatmaps to: {CELLTYPE_DIR}")

    plot_embedding_clustering_with_dot_legend(
        adata,
        out_path=str(OUT_DIR / "hbc_embedding_spatial_clustering.png"),
        embedding_key="embedding",
        method="leiden",
        resolution=0.6,
        n_neighbors=15,
    )
    print("Saved embedding-based spatial clustering with dot legend.")

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
