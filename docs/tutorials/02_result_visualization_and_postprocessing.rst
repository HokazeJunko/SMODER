Result visualization and postprocessing
=======================================

This tutorial introduces the postprocessing and visualization utilities provided by SMODER.

After running a SMODER pipeline, the main output file is usually:

.. code-block:: text

   spatial_decon_result.h5ad

This AnnData file may contain:

- inferred cell-type proportions,
- learned SMODER embeddings,
- spatial coordinates,
- RNA and second-modality encoder representations,
- reconstructed low-dimensional features.

SMODER also provides utilities for generating spatial result figures and reconstructing denoised gene-level or feature-level signals.

Loading a SMODER result
-----------------------

.. code-block:: python

   import scanpy as sc

   adata = sc.read_h5ad("spatial_decon_result.h5ad")

   print(adata)
   print(adata.obsm.keys())

Commonly used fields include:

.. code-block:: python

   adata.obsm["spatial"]
   adata.obsm["cell_type_proportions"]
   adata.obsm["embedding"]

For trained models that saved encoder representations, the result file may also contain keys such as:

.. code-block:: python

   adata.obsm["rna_encoder"]
   adata.obsm["peak_encoder"]
   adata.obsm["adt_encoder"]

Plotting cell-type proportion heatmaps
--------------------------------------

SMODER provides functions for plotting inferred cell-type proportions.

.. code-block:: python

   import scanpy as sc
   from smoder.visualization import plot_cell_type_proportion_panel

   adata = sc.read_h5ad("spatial_decon_result.h5ad")

   plot_cell_type_proportion_panel(
       adata,
       out_path="cell_type_proportion_panel.png",
       obsm_key="cell_type_proportions",
       obs_start_col=9,
       top_n=12,
       ncols=4,
       title="Spatial heatmaps of selected cell-type proportions",
   )

The argument ``obs_start_col`` is used when cell-type names are stored in ``adata.obs`` after metadata columns. For example:

- Mousebrain H3K27ac: cell-type names start from ``obs[9:]``.
- HBC: cell-type names start from ``obs[8:]``.
- Simulated human melanoma: cell-type names start from ``obs[4:]``.

Plotting all cell types
-----------------------

.. code-block:: python

   from smoder.visualization import plot_all_cell_type_proportions

   plot_all_cell_type_proportions(
       adata,
       out_path="all_cell_type_proportions.png",
       obsm_key="cell_type_proportions",
       obs_start_col=9,
       ncols=6,
       title="Spatial heatmaps of cell-type proportions",
   )

Plotting embedding-based spatial clusters
-----------------------------------------

The learned SMODER embedding can be clustered and visualized spatially.

.. code-block:: python

   from smoder.visualization import plot_embedding_spatial_clustering

   clustered = plot_embedding_spatial_clustering(
       adata,
       out_path="embedding_spatial_clustering.png",
       embedding_key="embedding",
       method="leiden",
       resolution=0.6,
       n_neighbors=15,
   )

The cluster labels are stored in:

.. code-block:: python

   clustered.obs["smoder_cluster"]

Reconstructing denoised RNA expression
--------------------------------------

The function ``omics_reconstruct`` can reconstruct selected RNA genes from trained SMODER embeddings.

.. code-block:: python

   from smoder.postprocessing import omics_reconstruct

   rna_recon = omics_reconstruct(
       omics_type="RNA",
       expr_path="RNA.h5ad",
       spatial_path="spatial_decon_result.h5ad",
       target_genes=["Penk", "Ppp1r1b", "Sez6l", "Gng2"],
       encoder_key="rna_encoder",
       hidden_dim=256,
       n_layers=3,
       epochs=500,
       lr=1e-4,
       patience=100,
       save_path="RNA_recon.h5ad",
   )

The returned AnnData object stores reconstructed values in ``.X``:

.. code-block:: python

   print(rna_recon.shape)
   print(rna_recon.var_names)

Reconstructing second-modality signals
--------------------------------------

For peak or gene-level epigenomic signals, use ``omics_type="EPIGENOMICS"`` or ``omics_type="ATAC"`` with ``peak_encoder``:

.. code-block:: python

   epi_recon = omics_reconstruct(
       omics_type="EPIGENOMICS",
       expr_path="gene_level_epigenomic_signal.h5ad",
       spatial_path="spatial_decon_result.h5ad",
       target_genes=["Penk", "Ppp1r1b", "Sez6l", "Gng2"],
       encoder_key="peak_encoder",
       hidden_dim=256,
       n_layers=3,
       epochs=500,
       lr=1e-4,
       patience=100,
       save_path="epigenomics_recon.h5ad",
   )

For ADT or protein marker signals, use ``omics_type="ADT"`` with ``adt_encoder``:

.. code-block:: python

   adt_recon = omics_reconstruct(
       omics_type="ADT",
       expr_path="HBC_ADT.h5ad",
       spatial_path="spatial_decon_result.h5ad",
       target_genes=["KRT5.1", "CD68.1", "CD8A.1", "HLA-DRA"],
       encoder_key="adt_encoder",
       hidden_dim=256,
       n_layers=3,
       epochs=500,
       lr=1e-4,
       patience=100,
       do_preprocess=False,
       save_path="ADT_recon.h5ad",
   )

Plotting reconstructed heatmaps
-------------------------------

After reconstruction, spatial heatmaps can be generated from the reconstructed AnnData object.

.. code-block:: python

   import scanpy as sc
   from smoder.visualization import plot_reconstruction_heatmaps

   rna_recon = sc.read_h5ad("RNA_recon.h5ad")

   plot_reconstruction_heatmaps(
       rna_recon,
       out_dir="figures",
       prefix="rna",
       title_prefix="Denoised RNA expression",
   )

Notes
-----

The reconstruction utilities are intended for downstream analysis and visualization. They require the SMODER result file to contain the appropriate encoder representations, such as ``rna_encoder``, ``peak_encoder``, or ``adt_encoder``.

Representative result figures are shown in the :doc:`../results` page.
