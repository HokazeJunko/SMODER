SMODER Tutorial 04: HBC Result Visualization
============================================

This tutorial demonstrates how to visualize representative SMODER results for the human breast cancer (HBC) RNA + ADT example.

We show:

1. spatial heatmaps of inferred cell-type proportions,
2. spatial heatmaps of denoised RNA markers,
3. spatial heatmaps of denoised ADT markers,
4. spatial clustering based on the learned embeddings.

Input files
-----------

This tutorial assumes that the following output files are available:

.. code-block:: text

   spatial_decon_result.h5ad
   RNA_recon.h5ad
   ADT_recon.h5ad

The file ``spatial_decon_result.h5ad`` stores SMODER outputs such as inferred cell-type proportions, spatial coordinates, learned embeddings, and encoder representations.

Run the reconstruction workflow
-------------------------------

If the reconstruction outputs are not ready yet, run:

.. code-block:: bash

   cd /data/xiongsc/projects/SMODER
   /data/xiongsc/conda_envs/SMODER/bin/python scripts/run_hbc_reconstruction_for_docs.py

This generates reconstructed RNA and ADT result files under:

.. code-block:: text

   outputs/hbc_reconstruction_for_docs/

Run the plotting workflow
-------------------------

Generate the result figures by running:

.. code-block:: bash

   cd /data/xiongsc/projects/SMODER
   /data/xiongsc/conda_envs/SMODER/bin/python scripts/plot_hbc_results_for_docs.py

The figures will be written to:

.. code-block:: text

   docs/_static/results/hbc/

Representative results
----------------------

Cell-type proportion heatmaps
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. image:: ../_static/results/hbc/hbc_cell_type_proportion_top9.png
   :width: 100%
   :align: center

A compact panel of inferred cell-type proportions in the HBC dataset.

Spatial clustering from learned embeddings
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. image:: ../_static/results/hbc/hbc_embedding_spatial_clustering.png
   :width: 70%
   :align: center

Spatial clustering based on SMODER learned embeddings.

Denoised RNA marker heatmaps
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 50 50

   * - .. image:: ../_static/results/hbc/hbc_rna_COL1A1_denoised_heatmap.png
          :width: 100%

       COL1A1

     - .. image:: ../_static/results/hbc/hbc_rna_EPCAM_denoised_heatmap.png
          :width: 100%

       EPCAM

   * - .. image:: ../_static/results/hbc/hbc_rna_KRT8_denoised_heatmap.png
          :width: 100%

       KRT8

     - .. image:: ../_static/results/hbc/hbc_rna_PECAM1_denoised_heatmap.png
          :width: 100%

       PECAM1

Denoised ADT marker heatmaps
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 50 50

   * - .. image:: ../_static/results/hbc/hbc_adt_CD68.1_denoised_heatmap.png
          :width: 100%

       CD68.1

     - .. image:: ../_static/results/hbc/hbc_adt_CD8A.1_denoised_heatmap.png
          :width: 100%

       CD8A.1

   * - .. image:: ../_static/results/hbc/hbc_adt_HLA-DRA_denoised_heatmap.png
          :width: 100%

       HLA-DRA

     - .. image:: ../_static/results/hbc/hbc_adt_KRT5.1_denoised_heatmap.png
          :width: 100%

       KRT5.1
