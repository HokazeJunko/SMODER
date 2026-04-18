Installation
============

Clone the repository
--------------------

.. code-block:: bash

   git clone <repository_url>
   cd SMODER

Create a Python environment
---------------------------

A dedicated Python environment is recommended.

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