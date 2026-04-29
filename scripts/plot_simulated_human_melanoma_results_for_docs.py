from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc


INPUT_H5AD = Path("/data/baiyh/keyan/模拟/数据/new/atac_rna/bench/seed=1/SMODER/spatial_decon_result.h5ad")
OUTDIR = Path("/data/xiongsc/projects/SMODER/docs/_static/results/simulated_human_melanoma")
INDIVIDUAL_DIR = OUTDIR / "cell_type_proportions"

PROPORTION_CMAP = "YlOrRd"
POINT_SIZE = 18
TITLE_FONTSIZE = 18
SUBTITLE_FONTSIZE = 12


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def safe_name(name: str) -> str:
    return (
        str(name)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace(":", "_")
    )


def get_spatial_coords(adata):
    if "spatial" not in adata.obsm:
        raise ValueError("Spatial coordinates not found in adata.obsm['spatial'].")
    coords = np.asarray(adata.obsm["spatial"])
    return coords[:, 0], coords[:, 1]


def get_cell_type_matrix_and_names(adata):
    if "cell_type_proportions" in adata.obsm:
        matrix = np.asarray(adata.obsm["cell_type_proportions"])
        names = list(adata.obs.columns[4:])
        if len(names) != matrix.shape[1]:
            names = [f"cell_type_{i}" for i in range(matrix.shape[1])]
        return matrix, names

    if "proportions" in adata.obsm:
        matrix = np.asarray(adata.obsm["proportions"])
        if "proportion_names" in adata.uns:
            names = list(adata.uns["proportion_names"])
        else:
            names = [f"cell_type_{i}" for i in range(matrix.shape[1])]
        return matrix, names

    raise ValueError("No cell-type proportion matrix found in adata.obsm.")


def get_vmax(values):
    values = np.asarray(values, dtype=float)
    vmax = np.quantile(values, 0.98)
    if vmax <= 0:
        vmax = values.max()
    if vmax <= 0:
        vmax = 1.0
    return vmax


def plot_single_heatmap(x, y, values, title, outpath):
    values = np.asarray(values, dtype=float)

    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    sca = ax.scatter(
        x,
        y,
        c=values,
        s=POINT_SIZE,
        cmap=PROPORTION_CMAP,
        vmin=0,
        vmax=get_vmax(values),
        edgecolors="none",
    )

    ax.set_title(title, fontsize=SUBTITLE_FONTSIZE)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.invert_yaxis()

    cbar = fig.colorbar(sca, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)

    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_panel(x, y, matrix, cell_types, outpath, title, ncols=5):
    n = len(cell_types)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.5 * ncols, 4.1 * nrows),
        squeeze=False,
    )

    for i, ct in enumerate(cell_types):
        ax = axes[i // ncols][i % ncols]
        values = np.asarray(matrix[:, i], dtype=float)

        sca = ax.scatter(
            x,
            y,
            c=values,
            s=POINT_SIZE,
            cmap=PROPORTION_CMAP,
            vmin=0,
            vmax=get_vmax(values),
            edgecolors="none",
        )

        ax.set_title(ct, fontsize=SUBTITLE_FONTSIZE)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.invert_yaxis()

        cbar = fig.colorbar(sca, ax=ax, fraction=0.046, pad=0.03)
        cbar.ax.tick_params(labelsize=8)

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle(title, fontsize=TITLE_FONTSIZE, y=0.995)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ensure_dir(OUTDIR)
    ensure_dir(INDIVIDUAL_DIR)

    print(f"Reading: {INPUT_H5AD}")
    adata = sc.read_h5ad(INPUT_H5AD)

    print("adata shape:", adata.shape)
    print("obs columns:", list(adata.obs.columns))
    print("obsm keys:", list(adata.obsm.keys()))

    x, y = get_spatial_coords(adata)
    matrix, cell_types = get_cell_type_matrix_and_names(adata)

    plot_panel(
        x=x,
        y=y,
        matrix=matrix,
        cell_types=cell_types,
        outpath=OUTDIR / "simulated_cell_type_proportion_all10.png",
        title="Simulated human melanoma spatial heatmaps of cell-type proportions",
        ncols=5,
    )
    print("Saved all-cell-type proportion panel.")

    preferred_order = [
        "mDC",
        "tumour_1",
        "tumour_2",
        "T_CD4",
        "T_CD8",
        "mono-mac",
        "plasma",
        "myeloid",
        "pDC",
        "T_reg",
    ]
    selected = [ct for ct in preferred_order if ct in cell_types]
    if len(selected) == 0:
        selected = cell_types

    selected_idx = [cell_types.index(ct) for ct in selected]

    plot_panel(
        x=x,
        y=y,
        matrix=matrix[:, selected_idx],
        cell_types=selected,
        outpath=OUTDIR / "simulated_cell_type_proportion_top10.png",
        title="Simulated human melanoma selected cell-type proportions",
        ncols=5,
    )
    print("Saved selected-cell-type proportion panel.")

    for i, ct in enumerate(cell_types):
        plot_single_heatmap(
            x=x,
            y=y,
            values=matrix[:, i],
            title=ct,
            outpath=INDIVIDUAL_DIR / f"{safe_name(ct)}.png",
        )

    print(f"Saved individual cell-type heatmaps to: {INDIVIDUAL_DIR}")
    print("All simulated human melanoma figures have been generated.")


if __name__ == "__main__":
    main()
