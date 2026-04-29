SMODER Tutorial 03: Simulated Human Melanoma Result Visualization
=================================================================

This tutorial demonstrates how to visualize representative SMODER results for the simulated human melanoma dataset.

For this dataset, we mainly show spatial heatmaps of inferred cell-type proportions.

Input file
----------

This tutorial assumes that the following SMODER result file is available:

.. code-block:: text

   spatial_decon_result.h5ad

The result file contains spatial coordinates and inferred cell-type proportions.

Run the plotting workflow
-------------------------

Generate the visualization figures by running:

.. code-block:: bash

   cd /data/xiongsc/projects/SMODER
   /data/xiongsc/conda_envs/SMODER/bin/python scripts/plot_simulated_human_melanoma_results_for_docs.py

The figures will be written to:

.. code-block:: text

   docs/_static/results/simulated_human_melanoma/

Representative results
----------------------

All cell-type proportion heatmaps
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. image:: ../_static/results/simulated_human_melanoma/simulated_cell_type_proportion_all10.png
   :width: 100%
   :align: center

Spatial heatmaps of inferred cell-type proportions across all simulated cell types.

Compact proportion panel
^^^^^^^^^^^^^^^^^^^^^^^^

.. image:: ../_static/results/simulated_human_melanoma/simulated_cell_type_proportion_top10.png
   :width: 100%
   :align: center

A compact panel view of representative cell-type proportion heatmaps.

Notes
-----

The plotting script uses a warm sequential colormap to make proportion hotspots more visually distinguishable in the simulated dataset.
