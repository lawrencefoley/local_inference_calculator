Installation
============

Requirements
------------

* Python 3.10 or higher

No external dependencies beyond Python standard library.

Installation
------------

Clone the repository or navigate to the project directory:

.. code-block:: bash

   cd local_inference_calculator

Project Structure
-----------------

.. code-block:: text

   local_inference_calculator/
   ├── __init__.py       # Package init
   ├── models.py         # LLM model database
   ├── gpus.py           # GPU database
   ├── calculator.py     # VRAM calculation logic
   ├── main.py           # Command-line interface
   └── docs/             # Documentation (this directory)
