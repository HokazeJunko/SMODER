Data
====

SMODER currently uses externally prepared datasets rather than shipping large raw data files inside the GitHub repository.

Dataset availability
--------------------

The validated example workflow currently uses a mouse brain RNA + peak dataset, including:

- a single-cell reference RNA dataset
- a spatial RNA dataset
- a second spatial modality dataset

Why data is not stored directly in the repository
-------------------------------------------------

Large research datasets are typically not committed directly to GitHub because:

- they can be too large for efficient version control
- they make the repository harder to clone and maintain
- they are usually better distributed through dedicated data hosting services

Instead, this repository provides documentation for how to organize and prepare the required input files.

Expected directory layout
-------------------------

A typical dataset layout is:

.. code-block:: text

   <data_root>/mousebrain_H3K27ac/
   ├── sc_mousebrain_processed.h5ad
   ├── RNA.h5ad
   └── peak.h5ad

Required files
--------------

The current validated mouse brain example expects:

- ``sc_mousebrain_processed.h5ad``: single-cell reference RNA
- ``RNA.h5ad``: spatial RNA
- ``peak.h5ad``: spatial peak data

How to prepare data
-------------------

At the current stage, users should download or prepare the required dataset files separately and place them into a local data directory matching the expected structure.

The exact file paths can then be configured in the pipeline configuration.

Example
-------

After preparing the dataset, the input directory may look like:

.. code-block:: text

   data/
   └── mousebrain_H3K27ac/
      ├── sc_mousebrain_processed.h5ad
      ├── RNA.h5ad
      └── peak.h5ad

Notes
-----

- The validated workflow currently assumes a Linux + conda environment.
- Future releases may provide additional download links, dataset mirrors, or automated download scripts.