Instalação
===========

Requisitos
----------

* Python 3.11 ou superior
* uv

Este projeto usa uv e depende do click para a interface de linha de comando.

Instalação via Clonagem
------------------------

.. code-block:: bash

   git clone https://github.com/lawrencefoley/local_inference_calculator.git
   cd local_inference_calculator
   uv sync

Verificação da Instalação
--------------------------

.. code-block:: bash

   uv run local-inference-calculator --help

Se a mensagem de ajuda for exibida, a instalação está correta.

Execução como ferramenta uv
---------------------------

.. code-block:: bash

   uvx --from . local-inference-calculator --help

Requirements para Desenvolvimento
----------------------------------

Para construir a documentação:

.. code-block:: bash

   uv pip install -r docs/requirements.txt
