Instalação
===========

Requisitos
----------

Python 3.8 ou superior.

Não há dependências externas além da biblioteca padrão do Python.

Instalação via Clonagem
------------------------

.. code-block:: bash

   git clone <repository-url>
   cd local_inference_calculator

Instalação via pip (quando disponível)
--------------------------------------

.. code-block:: bash

   pip install local-inference-calculator

Verificação da Instalação
--------------------------

.. code-block:: bash

   python main.py --help

Se a mensagem de ajuda for exibida, a instalação está correta.

If help message is displayed, installation is correct.

Ambiente Virtual (Opcional)
----------------------------

Recomendado para isolar dependências:

.. code-block:: bash

   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # ou
   venv\Scripts\activate  # Windows

Requirements para Desenvolvimento
----------------------------------

Para construir a documentação:

.. code-block:: bash

   cd docs
   pip install -r requirements.txt
