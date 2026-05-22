Installation
============

Requirements
------------

* Python 3.11 or higher
* uv

This project uses uv and depends on click for the command-line interface.

Installation
------------

Clone the repository and sync dependencies:

.. code-block:: bash

   git clone https://github.com/lawrencefoley/local_inference_calculator.git
   cd local_inference_calculator
   uv sync

Run from a checkout:

.. code-block:: bash

   uv run local-inference-calculator --help

Run as a uv tool from a checkout:

.. code-block:: bash

   uvx --from . local-inference-calculator --help

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
