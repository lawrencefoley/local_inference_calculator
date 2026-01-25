Guia do Usuário
================

Uso Básico
----------

Listar Modelos Disponíveis
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Para ver todos os modelos disponíveis no banco de dados:

.. code-block:: bash

   python main.py --list-models

Isso mostra nome do modelo, tamanho, arquitetura e requisitos de KV cache
para cada modelo.

Verificar Modelo Específico
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Para verificar os requisitos de VRAM para um modelo específico:

.. code-block:: bash

   python main.py --model 7 --context 8192
   python main.py -m 70 -c 16384 -q int4

A saída inclui:

* **Breakdown de VRAM**: Parâmetros, overhead e memória de KV cache
* **VRAM Total Requerido**: VRAM mínimo e recomendado da GPU
* **Compatibilidade de GPU**: Lista de GPUs que podem rodar o modelo,
  com porcentagem de VRAM livre

Todas as Combinações
~~~~~~~~~~~~~~~~~~~~

Para ver todas as combinações modelo × GPU:

.. code-block:: bash

   python main.py --context 4096

Ou com contexto maior:

.. code-block:: bash

   python main.py -c 8192


Opções de Linha de Comando
---------------------------

.. code-block:: bash

   python main.py [OPTIONS]

Opções disponíveis:

.. code-block:: bash

   -h, --help            Mostra mensagem de ajuda e sai
   -c, --context CONTEXT Tamanho do contexto em tokens (padrão: 4096)
   --list-models         Lista todos os modelos disponíveis
   -m SIZE, --model SIZE Tamanho do modelo em bilhões (ex: 0.6, 7, 13, 70)
   --gpu-type TYPE       Tipo de GPU: consumer, datacenter, all (padrão: all)
   --only-runs           Mostra apenas combinações que rodem
   --group-gpu           Agrupa resultados por GPU ao invés de modelo
   --summary TYPE        Resumo: model, gpu, both, none (padrão: both)
   --export-csv FILE     Exporta resultados para CSV
   --export-json FILE    Exporta resultados para JSON
   -q, --quantization Q  Precisão do modelo (fp32, fp16, int8, int4)


Exemplos de Uso
---------------

Apenas GPUs consumer:

.. code-block:: bash

   python main.py -c 8192 --gpu-type consumer

Mostrar apenas combinações viáveis:

.. code-block:: bash

   python main.py -c 4096 --only-runs

Exportar resultados para JSON:

.. code-block:: bash

   python main.py -c 8192 --export-json resultados.json

Usar quantização INT4 para modelos maiores:

.. code-block:: bash

   python main.py -c 8192 -q int4 --only-runs

Verificar se um modelo 70B roda em RTX 4090:

.. code-block:: bash

   python main.py --model 70 --context 8192


Precisões Suportadas
--------------------

FP32 (Float32)
    4 bytes por parâmetro. Maior precisão, maior uso de VRAM.
    Raramente usado para inferência LLM devido ao alto uso de VRAM.

FP16 (Float16)
    2 bytes por parâmetro. Padrão da indústria para inferência.
    Excelente precisão com metade do uso de VRAM do FP32.

INT8 (Inteiro 8-bit)
    1 byte por parâmetro. Quantização agressiva.
    Pequena perda de qualidade com economia significativa de VRAM.

INT4 (Inteiro 4-bit)
    0.5 byte por parâmetro. Quantização muito agressiva.
    Maiores economias mas perda de qualidade mais notável.


Interpretando os Resultados
---------------------------

``RODA`` (verde)
    A GPU tem VRAM suficiente para rodar o modelo com o contexto especificado.

``NÃO RODA`` (vermelho)
    A GPU não tem VRAM suficiente.

Aviso de baixa margem de segurança
    ``⚠️ Baixa margem de segurança`` indica que a combinação roda mas com
    menos de 10% de VRAM livre. Isso pode causar OOM em cenários reais
    devido a variações de implementação.


Fórmula de Cálculo de VRAM
---------------------------

A ferramenta usa um modelo conservador com quatro componentes:

**1. Parâmetros do Modelo**

Memória base para pesos do modelo. Como params_billion está em bilhões:

.. code-block:: python

   params_memory_gb = params_billion × bytes_per_param

Exemplo: modelo 70B em FP16 (2 bytes/param):
70 × 2 = 140 GB

**2. Overhead**

.. code-block:: python

   overhead_gb = params_memory_gb × 0.30  # 30%

**3. KV Cache**

.. code-block:: python

   kv_cache_gb = (kv_cache_mb_per_token × context_tokens × multiplier) / 1024

**4. Total**

.. code-block:: python

   total_vram_gb = model_with_overhead_gb + kv_cache_gb


Notas Técnicas
----------------

Os cálculos assumem alocação de memória estilo PyTorch (HF Transformers, vLLM).
Backends diferentes podem ter comportamento de memória variável:

* **TensorRT-LLM**: Alocadores customizados, ~10-20% menos memória
* **llama.cpp (GGUF)**: Arquivos memory-mapped, ~20-30% menos memória
* **EXL2**: Alocação otimizada, overhead mínimo

Para explicações detalhadas de termos técnicos usados nos cálculos de VRAM,
veja :doc:`glossario`.
