Data
====

The current validated example workflow uses a mouse brain RNA + peak dataset.

Required input files
--------------------

The current example expects three input files:

- ``sc_mousebrain_processed.h5ad``: single-cell reference RNA
- ``RNA.h5ad``: spatial RNA
- ``peak.h5ad``: spatial peak data

Example directory layout
------------------------

A typical dataset layout is:

.. code-block:: text

   <data_root>/mousebrain_H3K27ac/
   ├── sc_mousebrain_processed.h5ad
   ├── RNA.h5ad
   └── peak.h5ad

How to prepare data
-------------------

Please prepare the required dataset files separately and place them into a local data directory matching the expected structure.

The file paths can then be configured in the pipeline configuration.

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

- The current validated workflow uses single-cell reference RNA, spatial RNA, and a second spatial modality.
- The main tutorial currently follows the validated mouse brain example.
- Future releases may provide additional dataset links, mirrors, or download scripts.