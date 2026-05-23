Guia do Usuário
================

Uso Básico
----------

Listar Modelos Disponíveis
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Para ver todos os modelos disponíveis no banco de dados:

.. code-block:: bash

   uv run llmfit --list-models

Isso mostra nome do modelo, tamanho, arquitetura e requisitos de KV cache
para cada modelo.

Verificar Modelo Específico
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Para verificar os requisitos de VRAM para um modelo específico:

.. code-block:: bash

   uv run llmfit --model 7 --context 8192
   uv run llmfit -m 70 -c 16384 -q int4

A saída inclui:

* **Breakdown de VRAM**: Parâmetros, overhead e memória de KV cache
* **VRAM Total Requerido**: VRAM mínimo e recomendado da GPU
* **Compatibilidade de GPU**: Lista de GPUs que podem rodar o modelo,
  com porcentagem de VRAM livre

Todas as Combinações
~~~~~~~~~~~~~~~~~~~~

Para ver todas as combinações modelo × GPU:

.. code-block:: bash

   uv run llmfit --context 4096

Ou com contexto maior:

.. code-block:: bash

   uv run llmfit -c 8192


Contexto máximo para um limite de VRAM
--------------------------------------

Use ``--vram`` com uma quantização para ver quais modelos cabem e o contexto máximo estimado:

.. code-block:: bash

   uv run llmfit --vram 24 --quantization int4
   uv run llmfit --vram 16 --quantization fp16 --mode conservative

Combine com ``--model``, ``--params-b`` ou ``--config`` para verificar um único modelo.

Hugging Face config.json
------------------------

Derive metadados de KV cache a partir de um ``config.json`` do Hugging Face:

.. code-block:: bash

   uv run llmfit --config path/to/config.json --params-b 7 --context 8192

Se o config não incluir a contagem de parâmetros, informe com ``--params-b``. Use ``--model-name`` para sobrescrever o nome exibido.

Opções de Linha de Comando
---------------------------

.. code-block:: bash

   uv run llmfit [OPTIONS]

Opções Básicas
~~~~~~~~~~~~~~

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
   --mode MODE           Modo de cálculo (theoretical, conservative, production)

Opções Avançadas (v0.2.0)
~~~~~~~~~~~~~~~~~~~~~~~~~

Otimização de Offload de Camadas
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   --optimize      Mostrar configuração ótima de offload de camadas

Calcula quantas camadas transformer cabem na VRAM da GPU, exibindo:

* Camadas na GPU vs CPU
* Parâmetro ``--gpu-layers`` recomendado para llama.cpp
* Estimativa de impacto na performance
* Opções de offload para todas as GPUs disponíveis

Análise de Offload de CPU
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   --cpu          Habilitar cálculos de offload de CPU
   --ram GB        RAM do sistema disponível em GB (padrão: 32.0)
   --pcie-gen GEN         Geração PCIe: 3.0, 4.0 ou 5.0 (padrão: 4.0)

Calcula configuração de inferência híbrida GPU+CPU:

* Requisitos de RAM do sistema
* Impacto da largura de banda PCIe na performance
* Estimativa de tokens/segundo
* Distribuição de camadas entre GPU e CPU

Suporte Multi-GPU
^^^^^^^^^^^^^^^^^

.. code-block:: bash

   --multi            Habilitar modo multi-GPU
   --gpus CONFIG    Configuração multi-GPU (ex: "2x4090,1x3090")
   --multi-mode MODE  Modo de paralelismo: tensor ou pipeline (padrão: tensor)

Formato de configuração:

* Homogênea: ``3x4090`` (3 GPUs idênticas)
* Heterogênea: ``2x4090,1x3090`` (GPUs mistas)
* Nomes parciais: ``2x3090,1x4090``

Modos:

* **tensor** - Pesos do modelo divididos entre GPUs (mesmas camadas, shards diferentes)
* **pipeline** - Camadas diferentes em GPUs diferentes

Suporte a Formatos de Modelo
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   --gguf-file ARQUIVO    Nome do arquivo GGUF para auto-detectar quantização
   --format FORMATO        Formato do modelo: fp16, gguf, exl2, gptq, awq (padrão: fp16)

Auto-detecção GGUF suporta: Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_0, F16, F32

Multiplicadores de overhead por formato:

* FP16: 1.0x (linha de base)
* GGUF: 1.15x (+15% para estrutura de metadados)
* EXL2: 1.05x (+5% layout otimizado)
* GPTQ: 1.10x (+10% metadados de quantização)
* AWQ: 1.08x (+8% quantização activation-aware)


Exemplos de Uso
---------------

Apenas GPUs consumer:

.. code-block:: bash

   uv run llmfit -c 8192 --gpu-type consumer

Mostrar apenas combinações viáveis:

.. code-block:: bash

   uv run llmfit -c 4096 --only-runs

Exportar resultados para JSON:

.. code-block:: bash

   uv run llmfit -c 8192 --export-json resultados.json

Usar quantização INT4 para modelos maiores:

.. code-block:: bash

   uv run llmfit -c 8192 -q int4 --only-runs

Verificar se um modelo 70B roda em RTX 4090:

.. code-block:: bash

   uv run llmfit --model 70 --context 8192


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
