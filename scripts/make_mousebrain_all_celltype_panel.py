import os
import math
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

PROJECT_ROOT = "/data/xiongsc/projects/SMODER"
SPATIAL_RESULT = "/data/baiyh/keyan/SpaMultiDecon/Mousebrain_H3K27ac_result/SMODER_f3/spatial_decon_result.h5ad"
OUT_DIR = os.path.join(PROJECT_ROOT, "docs", "_static", "results", "mousebrain_h3k27ac")
os.makedirs(OUT_DIR, exist_ok=True)

adata = sc.read_h5ad(SPATIAL_RESULT)
coords = np.asarray(adata.obsm["spatial"])
x = coords[:, 0]
y = -coords[:, 1]

if "cell_type_proportions" in adata.obsm:
    prop = np.asarray(adata.obsm["cell_type_proportions"])
    names = list(adata.obs.columns[9:])
    prop_df = pd.DataFrame(prop, index=adata.obs_names, columns=names)
else:
    prop_df = adata.obs.iloc[:, 9:].apply(pd.to_numeric)

celltypes = list(prop_df.columns)

ncols = 6
nrows = math.ceil(len(celltypes) / ncols)

fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.2 * nrows))
axes = np.asarray(axes).reshape(-1)

last_sca = None
for ax, ct in zip(axes, celltypes):
    values = prop_df[ct].values
    last_sca = ax.scatter(
        x, y,
        c=values,
        s=2,
        cmap="viridis",
        vmin=0,
        vmax=1,
        linewidths=0,
    )
    ax.set_title(ct, fontsize=8)
    ax.set_aspect("equal")
    ax.axis("off")

for ax in axes[len(celltypes):]:
    ax.axis("off")

fig.suptitle("Spatial heatmaps of cell-type proportions", fontsize=18, y=0.995)
fig.colorbar(last_sca, ax=axes.tolist(), fraction=0.015, pad=0.01, label="Proportion")
plt.savefig(
    os.path.join(OUT_DIR, "mousebrain_cell_type_proportion_all59.png"),
    dpi=200,
    bbox_inches="tight",
)
plt.close()

print("Saved:", os.path.join(OUT_DIR, "mousebrain_cell_type_proportion_all59.png"))
