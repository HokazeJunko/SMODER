import os

from smoder.postprocessing import omics_reconstruct

PROJECT_ROOT = "/data/xiongsc/projects/SMODER"

SPATIAL_PATH = "/data/baiyh/keyan/SpaMultiDecon/Mousebrain_H3K27ac_result/SMODER_f3/spatial_decon_result.h5ad"

# RNA expression matrix for the current Mousebrain_H3K27ac dataset.
EXPR_RNA_PATH = "/data/baiyh/keyan/SpaMultiDecon/Mousebrain_H3K27ac/RNA.h5ad"

# Gene-level epigenomics matrix from the original reconstruction example.
# If an H3K27ac-specific gene-level matrix is provided later, replace this path.
EXPR_EPIGENOMICS_PATH = "/data/baiyh/keyan/SpaMultiDecon/MouseBrain/MouseBrain_H3K27me3_modified.h5ad"

OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "mousebrain_H3K27ac_reconstruction_for_docs")
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_RNA_GENES = ["Penk", "Ppp1r1b", "Sez6l", "Gng2"]
TARGET_EPIGENOMICS_GENES = ["Penk", "Ppp1r1b", "Sez6l", "Gng2"]

HIDDEN_DIM = 256
N_LAYERS = 3
EPOCHS = 500
LR = 1e-4
PATIENCE = 100
SEED = 42

print("=" * 80)
print("Running gene-level epigenomic signal reconstruction")
print("=" * 80)

epigenomics_recon = omics_reconstruct(
    omics_type="EPIGENOMICS",
    expr_path=EXPR_EPIGENOMICS_PATH,
    spatial_path=SPATIAL_PATH,
    target_genes=TARGET_EPIGENOMICS_GENES,
    encoder_key="peak_encoder",
    hidden_dim=HIDDEN_DIM,
    n_layers=N_LAYERS,
    epochs=EPOCHS,
    lr=LR,
    patience=PATIENCE,
    spatial_k=12,
    do_preprocess=True,
    pre_k=8,
    pre_quantile=0.8,
    pre_support=0.3,
    do_lognorm=True,
    target_sum=1e4,
    seed=SEED,
    save_path=os.path.join(OUT_DIR, "ATAC_recon.h5ad"),
)

print("=" * 80)
print("Running RNA reconstruction")
print("=" * 80)

rna_recon = omics_reconstruct(
    omics_type="RNA",
    expr_path=EXPR_RNA_PATH,
    spatial_path=SPATIAL_PATH,
    target_genes=TARGET_RNA_GENES,
    encoder_key="rna_encoder",
    hidden_dim=HIDDEN_DIM,
    n_layers=N_LAYERS,
    epochs=EPOCHS,
    lr=LR,
    patience=PATIENCE,
    n_angles=100,
    batch_size=64,
    print_interval=10,
    do_lognorm=True,
    target_sum=1e4,
    seed=SEED,
    save_path=os.path.join(OUT_DIR, "RNA_recon.h5ad"),
)

print("=" * 80)
print("Done.")
print("Output directory:", OUT_DIR)
print("=" * 80)
