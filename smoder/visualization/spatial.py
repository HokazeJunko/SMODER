"""
Spatial visualization utilities for SMODER results.
"""

from __future__ import annotations

import math
import os
from typing import Optional, Sequence

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt


def get_spatial_xy(adata: ad.AnnData, spatial_key: str = "spatial", flip_y: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Return x and y coordinates from ``adata.obsm[spatial_key]``."""
    if spatial_key not in adata.obsm:
        raise KeyError(f"adata.obsm['{spatial_key}'] not found.")

    coords = np.asarray(adata.obsm[spatial_key])
    x = coords[:, 0]
    y = coords[:, 1]

    if flip_y:
        y = -y

    return x, y


def save_spatial_continuous(
    adata: ad.AnnData,
    values: Sequence[float],
    out_path: str,
    title: str = "",
    spatial_key: str = "spatial",
    flip_y: bool = True,
    point_size: float = 6,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    figsize: tuple[float, float] = (5, 5),
    dpi: int = 300,
) -> None:
    """Save a spatial heatmap for continuous values."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    x, y = get_spatial_xy(adata, spatial_key=spatial_key, flip_y=flip_y)

    plt.figure(figsize=figsize)
    sca = plt.scatter(
        x,
        y,
        c=np.asarray(values),
        s=point_size,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidths=0,
    )
    plt.gca().set_aspect("equal")
    plt.axis("off")
    plt.title(title)
    plt.colorbar(sca, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def save_spatial_categorical(
    adata: ad.AnnData,
    labels: Sequence[str],
    out_path: str,
    title: str = "",
    spatial_key: str = "spatial",
    flip_y: bool = True,
    point_size: float = 6,
    cmap: str = "tab20",
    figsize: tuple[float, float] = (6, 5),
    dpi: int = 300,
) -> None:
    """Save a spatial map for categorical labels."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    x, y = get_spatial_xy(adata, spatial_key=spatial_key, flip_y=flip_y)

    labels = pd.Series(labels).astype("category")
    codes = labels.cat.codes.values
    categories = list(labels.cat.categories)

    plt.figure(figsize=figsize)
    sca = plt.scatter(x, y, c=codes, s=point_size, cmap=cmap, linewidths=0)
    plt.gca().set_aspect("equal")
    plt.axis("off")
    plt.title(title)

    cbar = plt.colorbar(sca, fraction=0.046, pad=0.04)
    cbar.set_ticks(range(len(categories)))
    cbar.set_ticklabels(categories)

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def get_cell_type_proportions(
    adata: ad.AnnData,
    obsm_key: str = "cell_type_proportions",
    obs_start_col: Optional[int] = None,
    cell_type_names: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Extract cell-type proportion matrix from a SMODER result AnnData object.

    Priority:
    1. use ``adata.obsm[obsm_key]`` if available;
    2. otherwise use ``adata.obs.iloc[:, obs_start_col:]``.
    """
    if obsm_key in adata.obsm:
        prop = np.asarray(adata.obsm[obsm_key])

        if cell_type_names is None:
            if obs_start_col is not None and len(adata.obs.columns[obs_start_col:]) == prop.shape[1]:
                cell_type_names = list(adata.obs.columns[obs_start_col:])
            else:
                cell_type_names = [f"CellType_{i}" for i in range(prop.shape[1])]

        return pd.DataFrame(prop, index=adata.obs_names, columns=list(cell_type_names))

    if obs_start_col is None:
        raise ValueError(
            f"adata.obsm['{obsm_key}'] not found. Please provide obs_start_col "
            "to extract proportions from adata.obs."
        )

    prop_df = adata.obs.iloc[:, obs_start_col:].copy()
    return prop_df.apply(pd.to_numeric)


def sanitize_filename(name: str) -> str:
    """Return a safe filename."""
    bad_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|', ' ']
    name = str(name)
    for ch in bad_chars:
        name = name.replace(ch, "_")
    return name


def plot_cell_type_proportion_panel(
    adata: ad.AnnData,
    out_path: str,
    obsm_key: str = "cell_type_proportions",
    obs_start_col: Optional[int] = None,
    cell_type_names: Optional[Sequence[str]] = None,
    selected_cell_types: Optional[Sequence[str]] = None,
    top_n: Optional[int] = 12,
    ncols: int = 4,
    point_size: float = 4,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    title: str = "Spatial heatmaps of cell-type proportions",
    spatial_key: str = "spatial",
    flip_y: bool = True,
    dpi: int = 300,
) -> pd.DataFrame:
    """Save a multi-panel figure of selected cell-type proportion heatmaps."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    prop_df = get_cell_type_proportions(
        adata,
        obsm_key=obsm_key,
        obs_start_col=obs_start_col,
        cell_type_names=cell_type_names,
    )

    if selected_cell_types is not None:
        selected = [ct for ct in selected_cell_types if ct in prop_df.columns]
    elif top_n is not None:
        selected = list(prop_df.mean(axis=0).sort_values(ascending=False).head(top_n).index)
    else:
        selected = list(prop_df.columns)

    if len(selected) == 0:
        raise ValueError("No valid cell types selected for plotting.")

    x, y = get_spatial_xy(adata, spatial_key=spatial_key, flip_y=flip_y)

    nrows = math.ceil(len(selected) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.asarray(axes).reshape(-1)

    for ax, ct in zip(axes, selected):
        sca = ax.scatter(
            x,
            y,
            c=prop_df[ct].values,
            s=point_size,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            linewidths=0,
        )
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(str(ct), fontsize=10)
        fig.colorbar(sca, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes[len(selected):]:
        ax.axis("off")

    fig.suptitle(title, y=0.995)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()

    return prop_df


def plot_all_cell_type_proportions(
    adata: ad.AnnData,
    out_path: str,
    obsm_key: str = "cell_type_proportions",
    obs_start_col: Optional[int] = None,
    cell_type_names: Optional[Sequence[str]] = None,
    ncols: int = 6,
    point_size: float = 2,
    cmap: str = "viridis",
    vmin: float = 0,
    vmax: float = 1,
    title: str = "Spatial heatmaps of cell-type proportions",
    spatial_key: str = "spatial",
    flip_y: bool = True,
    dpi: int = 200,
) -> pd.DataFrame:
    """Save one panel containing all cell-type proportion heatmaps."""
    prop_df = get_cell_type_proportions(
        adata,
        obsm_key=obsm_key,
        obs_start_col=obs_start_col,
        cell_type_names=cell_type_names,
    )

    selected = list(prop_df.columns)
    x, y = get_spatial_xy(adata, spatial_key=spatial_key, flip_y=flip_y)

    nrows = math.ceil(len(selected) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.2 * nrows))
    axes = np.asarray(axes).reshape(-1)

    last_sca = None
    for ax, ct in zip(axes, selected):
        last_sca = ax.scatter(
            x,
            y,
            c=prop_df[ct].values,
            s=point_size,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            linewidths=0,
        )
        ax.set_title(str(ct), fontsize=8)
        ax.set_aspect("equal")
        ax.axis("off")

    for ax in axes[len(selected):]:
        ax.axis("off")

    fig.suptitle(title, fontsize=18, y=0.995)
    if last_sca is not None:
        fig.colorbar(last_sca, ax=axes.tolist(), fraction=0.015, pad=0.01, label="Proportion")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    return prop_df


def plot_individual_cell_type_heatmaps(
    adata: ad.AnnData,
    out_dir: str,
    obsm_key: str = "cell_type_proportions",
    obs_start_col: Optional[int] = None,
    cell_type_names: Optional[Sequence[str]] = None,
    point_size: float = 6,
    cmap: str = "viridis",
    spatial_key: str = "spatial",
    flip_y: bool = True,
    dpi: int = 300,
) -> pd.DataFrame:
    """Save one spatial heatmap for each cell type."""
    os.makedirs(out_dir, exist_ok=True)

    prop_df = get_cell_type_proportions(
        adata,
        obsm_key=obsm_key,
        obs_start_col=obs_start_col,
        cell_type_names=cell_type_names,
    )

    for ct in prop_df.columns:
        out_path = os.path.join(out_dir, f"{sanitize_filename(ct)}.png")
        save_spatial_continuous(
            adata,
            prop_df[ct].values,
            out_path=out_path,
            title=f"Cell-type proportion: {ct}",
            spatial_key=spatial_key,
            flip_y=flip_y,
            point_size=point_size,
            cmap=cmap,
            dpi=dpi,
        )

    return prop_df


def plot_embedding_spatial_clustering(
    adata: ad.AnnData,
    out_path: str,
    embedding_key: str = "embedding",
    cluster_key: str = "smoder_cluster",
    method: str = "leiden",
    resolution: float = 0.6,
    n_neighbors: int = 15,
    spatial_key: str = "spatial",
    flip_y: bool = True,
    point_size: float = 6,
    dpi: int = 300,
) -> ad.AnnData:
    """Cluster a learned embedding and save a spatial cluster map."""
    if embedding_key not in adata.obsm:
        raise KeyError(f"adata.obsm['{embedding_key}'] not found.")

    adata_tmp = adata.copy()
    sc.pp.neighbors(adata_tmp, use_rep=embedding_key, n_neighbors=n_neighbors)

    method_lower = method.lower()
    if method_lower == "leiden":
        sc.tl.leiden(adata_tmp, resolution=resolution, key_added=cluster_key)
        label = "Leiden"
    elif method_lower == "louvain":
        sc.tl.louvain(adata_tmp, resolution=resolution, key_added=cluster_key)
        label = "Louvain"
    else:
        raise ValueError("method must be 'leiden' or 'louvain'.")

    save_spatial_categorical(
        adata_tmp,
        adata_tmp.obs[cluster_key],
        out_path=out_path,
        title=f"Spatial clustering based on learned embeddings ({label})",
        spatial_key=spatial_key,
        flip_y=flip_y,
        point_size=point_size,
        dpi=dpi,
    )

    return adata_tmp


def plot_reconstruction_heatmaps(
    recon_adata: ad.AnnData,
    out_dir: str,
    prefix: str,
    title_prefix: str,
    spatial_key: str = "spatial",
    flip_y: bool = True,
    point_size: float = 6,
    cmap: str = "viridis",
    dpi: int = 300,
) -> None:
    """Save spatial heatmaps for all reconstructed genes/features in recon_adata."""
    os.makedirs(out_dir, exist_ok=True)

    X = recon_adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X)

    for i, gene in enumerate(recon_adata.var_names):
        out_path = os.path.join(out_dir, f"{prefix}_{sanitize_filename(gene)}_denoised_heatmap.png")
        save_spatial_continuous(
            recon_adata,
            X[:, i],
            out_path=out_path,
            title=f"{title_prefix}: {gene}",
            spatial_key=spatial_key,
            flip_y=flip_y,
            point_size=point_size,
            cmap=cmap,
            dpi=dpi,
        )
