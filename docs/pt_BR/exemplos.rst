Exemplos / Examples
====================

Esta seção fornece exemplos práticos de uso do Local Inference Calculator.

This section provides practical examples of using the Local Inference Calculator.

Cenário 1: GPU Gamer com 8GB
-----------------------------

Você tem uma RTX 3060 (8GB) e quer saber quais modelos rodam.

You have an RTX 3060 (8GB) and want to know which models run.

.. code-block:: bash

   python main.py -c 4096 --gpu-type consumer --only-runs

Saída esperada:

Expected output:

.. code-block:: text

   Modelos que rodam em 8GB (contexto 4096):
   - 0.6B a 3B (FP16, INT8, INT4)
   - 7B (INT4 apenas)


Cenário 2: Estação de Trabalho com 24GB
----------------------------------------

Você tem uma RTX 4090 (24GB) e quer rodar um modelo 7B com contexto maior.

You have a RTX 4090 (24GB) and want to run a 7B model with larger context.

.. code-block:: bash

   python main.py -m 7 -c 16384

Isso mostrará se 7B com 16K contexto cabe em 24GB.

This will show if 7B with 16K context fits in 24GB.


Cenário 3: Comparando GPUs Datacenter
--------------------------------------

Comparando A100 (40GB) vs A100 (80GB) para modelo 70B.

Comparing A100 (40GB) vs A100 (80GB) for 70B model.

.. code-block:: bash

   python main.py -m 70 -c 8192 -q int4 --gpu-type datacenter


Cenário 4: Planejamento de Contexto Longo
------------------------------------------

Projetando requisitos de VRAM para contexto de 32K.

Projecting VRAM requirements for 32K context.

.. code-block:: bash

   python main.py -m 13 -c 32768

A ferramenta mostrará o KV cache projetado para contextos maiores.

The tool will show projected KV cache for larger contexts.


Cenário 5: Exportando Resultados
---------------------------------

Exportando todas as combinações para análise posterior.

Exporting all combinations for later analysis.

.. code-block:: bash

   python main.py -c 8192 --export-json analysis.json

Ou em CSV para planilhas:

Or in CSV for spreadsheets:

.. code-block:: bash

   python main.py -c 8192 --export-csv analysis.csv


Cenário 6: Múltiplos Modelos
-----------------------------

Verificando vários tamanhos de modelo de uma vez.

Checking multiple model sizes at once.

.. code-block:: bash

   # Shell script para verificar todos os tamanhos
   for size in 7 13 34 70; do
       echo "=== Modelo ${size}B ==="
       python main.py -m $size -c 8192
   done


Casos de Uso Avançados
-----------------------

Batch Processing
~~~~~~~~~~~~~~~~

Processando múltiplos contextos:

.. code-block:: bash

   for ctx in 4096 8192 16384 32768; do
       python main.py -m 13 -c $ctx --export-json results_${ctx}.json
   done

Comparação de Quantização
~~~~~~~~~~~~~~~~~~~~~~~~~~

Comparando diferentes níveis de quantização:

.. code-block:: bash

   python main.py -m 70 -c 8192 -q fp16   # ~175 GB
   python main.py -m 70 -c 8192 -q int8   # ~105 GB
   python main.py -m 70 -c 8192 -q int4   # ~72 GB


Script de Exemplo Python
-------------------------

Usando a API diretamente em Python:

Using the API directly in Python:

.. code-block:: python

   from calculator import VRAMCalculator
   from models import get_model
   from gpus import get_gpu

   # Criar calculadora
   calc = VRAMCalculator()

   # Obter modelo
   model = get_model(7)  # 7B model

   # Calcular VRAM para 8K contexto
   result = calc.calculate_vram(model, context_length=8192)

   print(f"VRAM Total: {result['total_vram_gb']} GB")
   print(f"Parâmetros: {result['params_memory_gb']} GB")
   print(f"KV Cache: {result['kv_cache_gb']} GB")

   # Ver compatibilidade com GPU específica
   gpu = get_gpu("RTX 4090")
   if calc.fits_in_gpu(result['total_vram_gb'], gpu['vram_gb']):
       print("RODA na RTX 4090!")
   else:
       print("NÃO roda na RTX 4090")
