Installation
============

Requirements
------------

The current validated environment is:

- Linux
- conda-based Python environment
- Python 3.12

The main tutorial and recommended workflow are currently oriented toward this environment.

Clone the repository
--------------------

.. code-block:: bash

   git clone <repository_url>
   cd SMODER

Create a Python environment
---------------------------

For example:

.. code-block:: bash

   conda create -n smoder python=3.12
   conda activate smoder

Install the package
-------------------

Install SMODER in editable mode:

.. code-block:: bash

   pip install -e .

Current recommended execution method
------------------------------------

.. code-block:: bash

   python -m smoder.pipelines.mousebrain_h3k27ac

Data preparation
----------------

Before running the example workflow, please prepare the required input files as described in the :doc:`data` page.