import os

from smoder.postprocessing import omics_reconstruct

PROJECT_ROOT = "/data/xiongsc/projects/SMODER"

SPATIAL_PATH = "/data/baiyh/keyan/SpaMultiDecon/HBC_SMODER_f3/spatial_decon_result.h5ad"

EXPR_RNA_PATH = "/data/baiyh/keyan/SpaMultiDecon/HBC/HBC_RNA.h5ad"
EXPR_ADT_PATH = "/data/baiyh/keyan/SpaMultiDecon/HBC/HBC_ADT.h5ad"

OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "hbc_reconstruction_for_docs")
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_RNA_GENES = ["EPCAM", "KRT8", "COL1A1", "PECAM1"]
TARGET_ADT_MARKERS = ["KRT5.1", "CD68.1", "CD8A.1", "HLA-DRA"]

HIDDEN_DIM = 256
N_LAYERS = 3
EPOCHS = 500
LR = 1e-4
PATIENCE = 100
SEED = 42

print("=" * 80)
print("Running HBC ADT / protein marker reconstruction")
print("=" * 80)

adt_recon = omics_reconstruct(
    omics_type="ADT",
    expr_path=EXPR_ADT_PATH,
    spatial_path=SPATIAL_PATH,
    target_genes=TARGET_ADT_MARKERS,
    encoder_key="adt_encoder",
    hidden_dim=HIDDEN_DIM,
    n_layers=N_LAYERS,
    epochs=EPOCHS,
    lr=LR,
    patience=PATIENCE,
    spatial_k=12,
    do_preprocess=False,
    do_lognorm=True,
    target_sum=1e4,
    seed=SEED,
    save_path=os.path.join(OUT_DIR, "ADT_recon.h5ad"),
)

print("=" * 80)
print("Running HBC RNA reconstruction")
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
