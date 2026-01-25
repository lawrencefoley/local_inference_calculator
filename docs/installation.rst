Installation
============

Requirements
------------

* Python 3.10 or higher / Python 3.10 ou superior

No external dependencies beyond Python standard library.

Não há dependências externas além da biblioteca padrão do Python.

Installation
------------

Clone the repository or navigate to the project directory:

Clone o repositório ou navegue até o diretório do projeto:

.. code-block:: bash

   cd local_inference_calculator

Project Structure
-----------------

.. code-block:: text

   local_inference_calculator/
   ├── __init__.py       # Package init
   ├── models.py         # LLM model database / Base de dados de LLMs
   ├── gpus.py           # GPU database / Base de dados de GPUs
   ├── calculator.py     # VRAM calculation logic / Lógica de cálculo de VRAM
   ├── main.py           # Command-line interface / Interface de linha de comando
   └── docs/             # Documentation (this directory) / Documentação (este diretório)
