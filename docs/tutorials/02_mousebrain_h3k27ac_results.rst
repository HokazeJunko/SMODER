SMODER Tutorial 02: Mouse Brain H3K27ac Result Visualization
============================================================

This tutorial demonstrates how to visualize representative SMODER results for the mouse brain RNA + H3K27ac peak example.

We show:

1. spatial heatmaps of inferred cell-type proportions,
2. spatial heatmaps of denoised RNA markers,
3. spatial heatmaps of denoised gene-level epigenomic signals,
4. spatial clustering based on the learned embeddings.

Input files
-----------

This tutorial assumes that the following output files are available:

.. code-block:: text

   spatial_decon_result.h5ad
   RNA_recon.h5ad
   ATAC_recon.h5ad

The file ``spatial_decon_result.h5ad`` stores SMODER outputs such as inferred cell-type proportions, spatial coordinates, learned embeddings, and encoder representations.

Run the reconstruction workflow
-------------------------------

If the reconstruction outputs are not ready yet, run:

.. code-block:: bash

   cd /data/xiongsc/projects/SMODER
   /data/xiongsc/conda_envs/SMODER/bin/python scripts/run_mousebrain_h3k27ac_reconstruction_for_docs.py

This generates reconstructed RNA and gene-level epigenomic result files under:

.. code-block:: text

   outputs/mousebrain_H3K27ac_reconstruction_for_docs/

Run the plotting workflow
-------------------------

Generate the result figures by running:

.. code-block:: bash

   cd /data/xiongsc/projects/SMODER
   /data/xiongsc/conda_envs/SMODER/bin/python scripts/plot_mousebrain_h3k27ac_results_for_docs.py

The figures will be written to:

.. code-block:: text

   docs/_static/results/mousebrain_h3k27ac/

Representative results
----------------------

Cell-type proportion heatmaps
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. image:: ../_static/results/mousebrain_h3k27ac/mousebrain_cell_type_proportion_top12.png
   :width: 100%
   :align: center

A compact view of selected spatial heatmaps of inferred cell-type proportions.

Spatial clustering from learned embeddings
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. image:: ../_static/results/mousebrain_h3k27ac/mousebrain_embedding_spatial_clustering.png
   :width: 70%
   :align: center

Spatial clustering based on SMODER learned embeddings.

Denoised RNA marker heatmaps
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 50 50

   * - .. image:: ../_static/results/mousebrain_h3k27ac/mousebrain_rna_Gng2_denoised_heatmap.png
          :width: 100%

       Gng2

     - .. image:: ../_static/results/mousebrain_h3k27ac/mousebrain_rna_Penk_denoised_heatmap.png
          :width: 100%

       Penk

   * - .. image:: ../_static/results/mousebrain_h3k27ac/mousebrain_rna_Ppp1r1b_denoised_heatmap.png
          :width: 100%

       Ppp1r1b

     - .. image:: ../_static/results/mousebrain_h3k27ac/mousebrain_rna_Sez6l_denoised_heatmap.png
          :width: 100%

       Sez6l

Denoised gene-level epigenomic signal heatmaps
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 50 50

   * - .. image:: ../_static/results/mousebrain_h3k27ac/mousebrain_epigenomics_Gng2_denoised_heatmap.png
          :width: 100%

       Gng2

     - .. image:: ../_static/results/mousebrain_h3k27ac/mousebrain_epigenomics_Penk_denoised_heatmap.png
          :width: 100%

       Penk

   * - .. image:: ../_static/results/mousebrain_h3k27ac/mousebrain_epigenomics_Ppp1r1b_denoised_heatmap.png
          :width: 100%

       Ppp1r1b

     - .. image:: ../_static/results/mousebrain_h3k27ac/mousebrain_epigenomics_Sez6l_denoised_heatmap.png
          :width: 100%

       Sez6l
