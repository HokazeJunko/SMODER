"""Postprocessing utilities for SMODER reconstruction.

This module provides ``omics_reconstruct`` for reconstructing selected RNA genes
or second-modality gene and feature signals from trained SMODER embeddings.

The input ``spatial_path`` should point to a SMODER result AnnData file that
contains spatial coordinates and encoder representations such as
``rna_encoder``, ``peak_encoder``, or ``adt_encoder``.
"""


from __future__ import annotations

import os
import warnings
from typing import Iterable, Optional, Sequence, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.sparse import issparse
from scipy.spatial import KDTree
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, random_split

warnings.filterwarnings("ignore")


def _get_device(device: Optional[str] = None) -> torch.device:
    """Return the requested torch device, or choose CUDA if available."""
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def set_random_seed(seed: int = 42) -> None:
    """Set NumPy and PyTorch random seeds."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def normalize_coords(coords: np.ndarray) -> np.ndarray:
    """Min-max normalize spatial coordinates to [0, 1]."""
    coords = np.asarray(coords)
    return (coords - coords.min(axis=0)) / (coords.max(axis=0) - coords.min(axis=0) + 1e-8)


def standardize_embedding(
    embedding_data: np.ndarray,
    train_size: int,
) -> Tuple[torch.Tensor, StandardScaler]:
    """Z-score standardize embeddings using the training subset only."""
    scaler = StandardScaler()
    scaler.fit(embedding_data[:train_size])
    embed_std = scaler.transform(embedding_data)
    return torch.tensor(embed_std, dtype=torch.float32), scaler


def lognorm_transform(expr_matrix: np.ndarray, target_sum: float = 1e4) -> np.ndarray:
    """Apply total-count normalization followed by log1p transformation."""
    expr_matrix = np.asarray(expr_matrix)
    total_counts = expr_matrix.sum(axis=1, keepdims=True)
    total_counts[total_counts == 0] = 1
    return np.log1p((expr_matrix / total_counts) * target_sum)


def filter_valid_genes(
    adata: ad.AnnData,
    target_genes: Optional[Sequence[str]],
) -> list[str]:
    """Return target genes/features that are present in ``adata.var_names``."""
    adata.var_names = adata.var_names.astype(str)

    if target_genes is None:
        valid_genes = adata.var_names.tolist()
        print(f"No target genes specified. Using all {len(valid_genes)} features.")
        return valid_genes

    target_genes = [str(g) for g in target_genes]
    valid_genes = list(set(target_genes) & set(adata.var_names))
    invalid_genes = list(set(target_genes) - set(adata.var_names))

    if invalid_genes:
        print(f"Warning: ignored {len(invalid_genes)} missing genes/features: {invalid_genes[:10]}")

    if not valid_genes:
        raise ValueError("No valid target genes/features found. Please check target gene names.")

    # Preserve the user-provided order whenever possible.
    ordered_valid = [g for g in target_genes if g in valid_genes]
    return ordered_valid


def load_align_data(expr_path: str, spatial_path: str) -> Tuple[ad.AnnData, ad.AnnData]:
    """
    Load expression and SMODER result AnnData files, then align common spots.
    """
    adata_expr = ad.read_h5ad(expr_path)
    adata_spatial = ad.read_h5ad(spatial_path)

    common_spots = list(set(adata_expr.obs_names) & set(adata_spatial.obs_names))
    if len(common_spots) == 0:
        raise ValueError("No common spots found between expr_path and spatial_path.")

    adata_expr = adata_expr[common_spots].copy()
    adata_spatial = adata_spatial[common_spots].copy()

    if "spatial" not in adata_spatial.obsm:
        raise KeyError("spatial_path must contain adata.obsm['spatial'].")

    print(f"Loaded and aligned data: {len(common_spots)} common spots; {adata_expr.n_vars} target features.")
    return adata_expr, adata_spatial


def project_coords(coords: np.ndarray, n_angles: int = 100) -> torch.Tensor:
    """Project 2D spatial coordinates under multiple rotation angles."""
    coords_norm = normalize_coords(coords)
    x, y = coords_norm[:, 0], coords_norm[:, 1]
    thetas = np.linspace(0, np.pi, n_angles, endpoint=False)

    proj_feats = []
    for theta in thetas:
        x_rot = x * np.cos(theta) - y * np.sin(theta)
        y_rot = x * np.sin(theta) + y * np.cos(theta)
        proj_feats.extend([x_rot, y_rot])

    return torch.tensor(np.stack(proj_feats, axis=1), dtype=torch.float32)


class RNAFCModel(nn.Module):
    """Fully connected model for RNA gene reconstruction."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 64, n_layers: int = 3):
        super().__init__()

        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def rna_run(
    adata_expr: ad.AnnData,
    adata_spatial: ad.AnnData,
    target_genes: Optional[Sequence[str]],
    hidden_dim: int,
    n_layers: int,
    epochs: int,
    lr: float,
    patience: int,
    encoder_key: str = "rna_encoder",
    n_angles: int = 100,
    batch_size: int = 64,
    print_interval: int = 10,
    do_lognorm: bool = True,
    target_sum: float = 1e4,
    device: Optional[str] = None,
    seed: int = 42,
) -> ad.AnnData:
    """Run RNA gene reconstruction."""
    set_random_seed(seed)
    torch_device = _get_device(device)

    coords = adata_spatial.obsm["spatial"]

    if encoder_key not in adata_spatial.obsm:
        raise KeyError(f"adata_spatial.obsm['{encoder_key}'] not found.")

    rna_encoder = adata_spatial.obsm[encoder_key]
    target_genes = filter_valid_genes(adata_expr, target_genes)

    adata_label = adata_expr[:, target_genes].copy()
    if do_lognorm:
        sc.pp.normalize_total(adata_label, target_sum=target_sum)
        sc.pp.log1p(adata_label)

    label = adata_label.X.toarray() if issparse(adata_label.X) else np.asarray(adata_label.X)
    print(
        f"RNA target label range: min={label.min():.6f}, "
        f"max={label.max():.6f}, mean={label.mean():.6f}"
    )

    proj_feats = project_coords(coords, n_angles)
    train_size = int(0.8 * len(coords))
    encoder_std, _ = standardize_embedding(rna_encoder, train_size)

    input_feats = torch.cat([proj_feats, encoder_std], dim=1).to(torch_device)
    label_tensor = torch.tensor(label, dtype=torch.float32).to(torch_device)

    dataset = TensorDataset(input_feats, label_tensor)
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(
        dataset,
        [train_size, len(dataset) - train_size],
        generator=generator,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = RNAFCModel(input_feats.shape[1], len(target_genes), hidden_dim, n_layers).to(torch_device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    min_val_loss = float("inf")
    counter = 0
    best_w = None

    print(
        f"Start RNA reconstruction: input_dim={input_feats.shape[1]}, "
        f"targets={len(target_genes)}, device={torch_device}"
    )

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for x, y in train_loader:
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)

        avg_train_loss = train_loss / train_size

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                pred = model(x)
                val_loss += criterion(pred, y).item() * x.size(0)

        avg_val_loss = val_loss / len(val_ds)

        if (epoch + 1) % print_interval == 0:
            print(
                f"Epoch [{epoch + 1:5d}/{epochs}] | "
                f"Train MSE: {avg_train_loss:.6f} | Val MSE: {avg_val_loss:.6f}"
            )

        if avg_val_loss < min_val_loss:
            min_val_loss = avg_val_loss
            best_w = model.state_dict()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"RNA early stopping at epoch {epoch + 1}; best Val MSE: {min_val_loss:.6f}")
                break

    if best_w is None:
        raise RuntimeError("RNA reconstruction failed to obtain model weights.")

    model.load_state_dict(best_w)
    model.eval()

    with torch.no_grad():
        recon = model(input_feats).cpu().numpy()

    recon[recon < 0] = 0
    print(
        f"RNA reconstruction range: min={recon.min():.6f}, "
        f"max={recon.max():.6f}, mean={recon.mean():.6f}"
    )

    recon_adata = ad.AnnData(
        X=recon,
        obs=adata_spatial.obs.copy(),
        var=pd.DataFrame(index=target_genes),
        obsm={"spatial": coords, encoder_key: rna_encoder},
        layers={"original_label": label},
    )

    recon_adata.uns["run_params"] = {
        "omics": "RNA",
        "encoder_key": encoder_key,
        "hidden_dim": hidden_dim,
        "n_layers": n_layers,
        "epochs": epochs,
        "lr": lr,
        "n_angles": n_angles,
        "do_lognorm": do_lognorm,
        "target_sum": target_sum,
        "seed": seed,
    }

    print(f"RNA reconstruction finished. Result shape: {recon_adata.shape}")
    return recon_adata


def second_modality_preprocess_single(
    expr: np.ndarray,
    coords: np.ndarray,
    k: int = 8,
    quantile: float = 0.8,
    support_thr: float = 0.3,
) -> np.ndarray:
    """Filter noisy signal using quantile thresholding and neighborhood support."""
    expr = expr.copy()
    q_thr = np.quantile(expr, quantile)
    expr_thr = np.where(expr >= q_thr, expr, 0.0)

    if np.sum(expr_thr > 0) == 0:
        return expr_thr

    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    _, indices = nbrs.kneighbors(coords)

    for i in range(len(expr_thr)):
        if expr_thr[i] == 0:
            continue
        neigh_support = np.mean(expr_thr[indices[i, 1:]] > 0)
        if neigh_support < support_thr:
            expr_thr[i] = 0.0

    return expr_thr


def second_modality_preprocess_batch(
    adata_expr: ad.AnnData,
    adata_spatial: ad.AnnData,
    target_genes: Sequence[str],
    k: int = 8,
    quantile: float = 0.8,
    support_thr: float = 0.3,
) -> ad.AnnData:
    """Apply neighborhood-aware preprocessing to selected second-modality features."""
    coords = adata_spatial.obsm["spatial"]
    target_genes = filter_valid_genes(adata_expr, target_genes)

    adata_proc = adata_expr[:, target_genes].copy()
    adata_proc.layers["raw_X"] = adata_proc.X.copy()

    expr_mat = adata_proc.X.toarray() if issparse(adata_proc.X) else np.asarray(adata_proc.X)

    for idx, gene in enumerate(target_genes):
        print(f"Preprocess feature {idx + 1}/{len(target_genes)}: {gene}")
        expr_mat[:, idx] = second_modality_preprocess_single(
            expr_mat[:, idx],
            coords,
            k=k,
            quantile=quantile,
            support_thr=support_thr,
        )

    adata_proc.X = expr_mat

    raw = adata_proc.layers["raw_X"].toarray() if issparse(adata_proc.layers["raw_X"]) else adata_proc.layers["raw_X"]
    raw_nonzero = np.sum(raw > 0)
    proc_nonzero = np.sum(expr_mat > 0)
    filter_ratio = 1 - proc_nonzero / raw_nonzero if raw_nonzero > 0 else 0

    adata_proc.uns["preprocess_params"] = {
        "k_neighbors": k,
        "quantile": quantile,
        "support_threshold": support_thr,
        "target_genes": list(target_genes),
        "raw_nonzero": int(raw_nonzero),
        "proc_nonzero": int(proc_nonzero),
        "filter_ratio": float(filter_ratio),
    }

    print(
        f"Second-modality preprocessing finished: "
        f"raw_nonzero={raw_nonzero}, processed_nonzero={proc_nonzero}, "
        f"filter_ratio={filter_ratio:.2%}"
    )

    return adata_proc


class DenseGCNConv(nn.Module):
    """Dense GCN layer used by second-modality reconstruction."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        degree = torch.sum(adj, dim=1) + 1e-6
        degree_inv_sqrt = torch.pow(degree, -0.5)
        adj_norm = torch.diag(degree_inv_sqrt) @ adj @ torch.diag(degree_inv_sqrt)
        x = adj_norm @ x
        return self.linear(x)


class SecondModalityGCNModel(nn.Module):
    """GCN model for reconstructing second-modality gene/feature signals."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256, n_layers: int = 3):
        super().__init__()

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.gcn_layers = nn.ModuleList([DenseGCNConv(hidden_dim) for _ in range(n_layers)])

        self.layers.append(nn.Linear(input_dim, hidden_dim))
        self.norms.append(nn.BatchNorm1d(hidden_dim))

        for _ in range(n_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.norms.append(nn.BatchNorm1d(hidden_dim))

        self.out_layer = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x = self.layers[0](x)
        x = self.norms[0](x)
        x = F.relu(x)
        x = self.dropout(x)

        for i in range(1, len(self.layers)):
            x = self.gcn_layers[i - 1](x, adj)
            x = self.layers[i](x)
            x = self.norms[i](x)
            x = F.relu(x)
            x = self.dropout(x)

        return self.out_layer(x)


def build_spatial_adj(coords: np.ndarray, k: int = 12, device: Optional[str] = None) -> torch.Tensor:
    """Build dense KNN adjacency matrix from spatial coordinates."""
    torch_device = _get_device(device)

    coords_norm = normalize_coords(coords)
    tree = KDTree(coords_norm)
    n_spots = len(coords)

    adj = np.zeros((n_spots, n_spots), dtype=np.float32)
    for i in range(n_spots):
        _, idx = tree.query(coords_norm[i], k=k + 1)
        adj[i, idx[1:]] = 1.0

    return torch.tensor(adj, dtype=torch.float32).to(torch_device)


def second_modality_run(
    adata_expr: ad.AnnData,
    adata_spatial: ad.AnnData,
    target_genes: Optional[Sequence[str]],
    hidden_dim: int,
    n_layers: int,
    epochs: int,
    lr: float,
    patience: int,
    encoder_key: str,
    omics_label: str = "ATAC",
    spatial_k: int = 12,
    do_preprocess: bool = True,
    pre_k: int = 8,
    pre_quantile: float = 0.8,
    pre_support: float = 0.3,
    do_lognorm: bool = True,
    target_sum: float = 1e4,
    device: Optional[str] = None,
    seed: int = 42,
) -> ad.AnnData:
    """Run second-modality reconstruction using a GCN model."""
    set_random_seed(seed)
    torch_device = _get_device(device)

    coords = adata_spatial.obsm["spatial"]

    if encoder_key not in adata_spatial.obsm:
        raise KeyError(f"adata_spatial.obsm['{encoder_key}'] not found.")

    encoder = adata_spatial.obsm[encoder_key]
    target_genes = filter_valid_genes(adata_expr, target_genes)

    if do_preprocess:
        adata_expr = second_modality_preprocess_batch(
            adata_expr,
            adata_spatial,
            target_genes,
            k=pre_k,
            quantile=pre_quantile,
            support_thr=pre_support,
        )

    adata_label = adata_expr[:, target_genes].copy()
    label_raw = adata_label.X.toarray() if issparse(adata_label.X) else np.asarray(adata_label.X)

    label = lognorm_transform(label_raw, target_sum=target_sum) if do_lognorm else label_raw

    print(
        f"{omics_label} target label range: min={label.min():.6f}, "
        f"max={label.max():.6f}, mean={label.mean():.6f}"
    )

    scaler = StandardScaler()
    encoder_std = scaler.fit_transform(encoder)
    encoder_tensor = torch.tensor(encoder_std, dtype=torch.float32).to(torch_device)

    adj_matrix = build_spatial_adj(coords, k=spatial_k, device=device)

    model = SecondModalityGCNModel(encoder.shape[1], len(target_genes), hidden_dim, n_layers).to(torch_device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    min_loss = float("inf")
    counter = 0
    best_w = None

    print(
        f"Start {omics_label} reconstruction: encoder_dim={encoder.shape[1]}, "
        f"targets={len(target_genes)}, spatial_k={spatial_k}, device={torch_device}"
    )

    label_tensor = torch.tensor(label, dtype=torch.float32).to(torch_device)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        pred = model(encoder_tensor, adj_matrix)
        loss = criterion(pred, label_tensor)
        loss.backward()
        optimizer.step()

        if loss.item() < min_loss:
            min_loss = loss.item()
            best_w = model.state_dict()
            counter = 0
        else:
            counter += 1

        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch + 1:4d}/{epochs}] | Loss: {loss.item():.6f} | Best Loss: {min_loss:.6f}")

        if counter >= patience:
            print(f"{omics_label} early stopping at epoch {epoch + 1}; best loss: {min_loss:.6f}")
            break

    if best_w is None:
        raise RuntimeError(f"{omics_label} reconstruction failed to obtain model weights.")

    model.load_state_dict(best_w)
    model.eval()

    with torch.no_grad():
        recon = model(encoder_tensor, adj_matrix).cpu().numpy()

    recon[recon < 0] = 0
    print(
        f"{omics_label} reconstruction range: min={recon.min():.6f}, "
        f"max={recon.max():.6f}, mean={recon.mean():.6f}"
    )

    recon_adata = ad.AnnData(
        X=recon,
        obs=adata_spatial.obs.copy(),
        var=pd.DataFrame(index=target_genes),
        obsm={"spatial": coords, encoder_key: encoder},
        layers={"original_label": label},
    )

    recon_adata.uns["run_params"] = {
        "omics": omics_label,
        "encoder_key": encoder_key,
        "hidden_dim": hidden_dim,
        "n_layers": n_layers,
        "epochs": epochs,
        "lr": lr,
        "spatial_k": spatial_k,
        "do_preprocess": do_preprocess,
        "do_lognorm": do_lognorm,
        "target_sum": target_sum,
        "seed": seed,
    }

    if do_preprocess and "preprocess_params" in adata_expr.uns:
        recon_adata.uns["preprocess_params"] = adata_expr.uns["preprocess_params"]

    print(f"{omics_label} reconstruction finished. Result shape: {recon_adata.shape}")
    return recon_adata


def _default_encoder_key(omics_type: str) -> str:
    """Return default encoder key for a given omics type."""
    omics_type = omics_type.upper()

    if omics_type == "RNA":
        return "rna_encoder"
    if omics_type in {"ATAC", "PEAK", "EPIGENOMICS"}:
        return "peak_encoder"
    if omics_type in {"ADT", "PROTEIN"}:
        return "adt_encoder"

    return "embedding"


def omics_reconstruct(
    omics_type: str,
    expr_path: str,
    spatial_path: str,
    target_genes: Optional[Sequence[str]],
    hidden_dim: int = 64,
    n_layers: int = 3,
    epochs: int = 500,
    lr: float = 1e-4,
    patience: int = 30,
    save_path: Optional[str] = None,
    encoder_key: Optional[str] = None,
    device: Optional[str] = None,
    seed: int = 42,
    # RNA-specific parameters
    n_angles: int = 100,
    batch_size: int = 64,
    print_interval: int = 10,
    do_lognorm: bool = True,
    target_sum: float = 1e4,
    # second-modality parameters
    spatial_k: int = 12,
    do_preprocess: Optional[bool] = None,
    pre_k: int = 8,
    pre_quantile: float = 0.8,
    pre_support: float = 0.3,
) -> ad.AnnData:
    """
    Reconstruct selected RNA or second-modality gene/feature signals.

    Parameters
    ----------
    omics_type
        One of ``"RNA"``, ``"ATAC"``, ``"PEAK"``, ``"EPIGENOMICS"``,
        ``"ADT"``, or ``"PROTEIN"``.
    expr_path
        Path to an AnnData file containing the target expression/signal matrix.
    spatial_path
        Path to a SMODER result AnnData file containing spatial coordinates and
        encoder representations.
    target_genes
        Genes or features to reconstruct. If ``None``, all features are used.
    encoder_key
        Key in ``spatial_path.obsm`` used as model input. If ``None``, a default
        is selected based on ``omics_type``.
    save_path
        Optional output path for saving the reconstructed AnnData object.

    Returns
    -------
    AnnData
        Reconstructed AnnData object. ``.X`` stores denoised/reconstructed
        signals, ``.obsm["spatial"]`` stores spatial coordinates, and
        ``.layers["original_label"]`` stores the target labels used for fitting.
    """
    omics_type = omics_type.upper()

    valid_types = {"RNA", "ATAC", "PEAK", "EPIGENOMICS", "ADT", "PROTEIN"}
    if omics_type not in valid_types:
        raise ValueError(f"omics_type must be one of {sorted(valid_types)}; got {omics_type}.")

    if not os.path.exists(expr_path):
        raise FileNotFoundError(f"expr_path does not exist: {expr_path}")

    if not os.path.exists(spatial_path):
        raise FileNotFoundError(f"spatial_path does not exist: {spatial_path}")

    encoder_key = encoder_key or _default_encoder_key(omics_type)

    print("=" * 80)
    print(f"Start {omics_type} reconstruction")
    print(f"Expression path: {expr_path}")
    print(f"SMODER result path: {spatial_path}")
    print(f"Encoder key: {encoder_key}")
    print("=" * 80)

    adata_expr, adata_spatial = load_align_data(expr_path, spatial_path)

    if omics_type == "RNA":
        recon_adata = rna_run(
            adata_expr=adata_expr,
            adata_spatial=adata_spatial,
            target_genes=target_genes,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            epochs=epochs,
            lr=lr,
            patience=patience,
            encoder_key=encoder_key,
            n_angles=n_angles,
            batch_size=batch_size,
            print_interval=print_interval,
            do_lognorm=do_lognorm,
            target_sum=target_sum,
            device=device,
            seed=seed,
        )
    else:
        if do_preprocess is None:
            do_preprocess = omics_type in {"ATAC", "PEAK", "EPIGENOMICS"}

        recon_adata = second_modality_run(
            adata_expr=adata_expr,
            adata_spatial=adata_spatial,
            target_genes=target_genes,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            epochs=epochs,
            lr=lr,
            patience=patience,
            encoder_key=encoder_key,
            omics_label=omics_type,
            spatial_k=spatial_k,
            do_preprocess=do_preprocess,
            pre_k=pre_k,
            pre_quantile=pre_quantile,
            pre_support=pre_support,
            do_lognorm=do_lognorm,
            target_sum=target_sum,
            device=device,
            seed=seed,
        )

    if save_path is not None:
        if not save_path.endswith(".h5ad"):
            save_path += ".h5ad"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        recon_adata.write_h5ad(save_path)
        print(f"Saved reconstruction result to: {save_path}")

    print("=" * 80)
    print(f"{omics_type} reconstruction finished.")
    print("=" * 80)
    return recon_adata
