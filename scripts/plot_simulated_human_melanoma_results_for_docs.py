import os

import scanpy as sc

from smoder.visualization import (
    plot_all_cell_type_proportions,
    plot_cell_type_proportion_panel,
    plot_individual_cell_type_heatmaps,
)

PROJECT_ROOT = "/data/xiongsc/projects/SMODER"

SPATIAL_RESULT = "/data/baiyh/keyan/模拟/数据/new/atac_rna/bench/seed=1/SMODER/spatial_decon_result.h5ad"

OUT_DIR = os.path.join(PROJECT_ROOT, "docs", "_static", "results", "simulated_human_melanoma")
CELLTYPE_DIR = os.path.join(OUT_DIR, "cell_type_proportions")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CELLTYPE_DIR, exist_ok=True)


def main():
    print("Reading:", SPATIAL_RESULT)
    adata = sc.read_h5ad(SPATIAL_RESULT)

    print("adata shape:", adata.shape)
    print("obs columns:", list(adata.obs.columns))
    print("obsm keys:", list(adata.obsm.keys()))

    # 前四列是 metadata，后面 10 列是细胞类型名称
    cell_type_names = list(adata.obs.columns[4:])

    # 更显眼的配色：turbo
    plot_all_cell_type_proportions(
        adata,
        out_path=os.path.join(OUT_DIR, "simulated_cell_type_proportion_all10.png"),
        obsm_key="cell_type_proportions",
        cell_type_names=cell_type_names,
        ncols=5,
        point_size=12,
        vmin=None,
        vmax=None,
        cmap="turbo",
        title="Simulated human melanoma spatial heatmaps of cell-type proportions",
    )
    print("Saved all-cell-type proportion panel.")

    plot_cell_type_proportion_panel(
        adata,
        out_path=os.path.join(OUT_DIR, "simulated_cell_type_proportion_top10.png"),
        obsm_key="cell_type_proportions",
        cell_type_names=cell_type_names,
        top_n=10,
        ncols=5,
        point_size=12,
        vmin=None,
        vmax=None,
        cmap="turbo",
        title="Simulated human melanoma selected cell-type proportions",
    )
    print("Saved selected-cell-type proportion panel.")

    plot_individual_cell_type_heatmaps(
        adata,
        out_dir=CELLTYPE_DIR,
        obsm_key="cell_type_proportions",
        cell_type_names=cell_type_names,
        point_size=12,
        cmap="turbo",
    )
    print(f"Saved individual cell-type heatmaps to: {CELLTYPE_DIR}")

    print("All simulated human melanoma figures have been generated.")


if __name__ == "__main__":
    main()
