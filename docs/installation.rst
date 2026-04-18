Installation
============

Requirements
------------

The current validated environment is:

- Linux
- conda-based Python environment
- Python 3.12

The main tutorial and recommended workflow are currently oriented toward this environment.

Install from PyPI
-----------------

The recommended way to install SMODER is:

.. code-block:: bash

   pip install smoder

Install from GitHub
-------------------

To install the latest development version from source:

.. code-block:: bash

   git clone https://github.com/HokazeJunko/SMODER.git
   cd SMODER
   pip install -e .

Create a Python environment
---------------------------

For example:

.. code-block:: bash

   conda create -n smoder python=3.12
   conda activate smoder

Current recommended execution method
------------------------------------

.. code-block:: bash

   python -m smoder.pipelines.mousebrain_h3k27ac

Data preparation
----------------

Before running the example workflow, please prepare the required input files as described in the :doc:`data` page.