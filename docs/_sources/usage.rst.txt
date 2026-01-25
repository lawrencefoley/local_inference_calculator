User Guide
==========

Basic Usage
------------

To calculate inference viability with defaults (context 4096, FP16):

Para calcular a viabilidade de inferência com os padrões (contexto 4096, FP16):

.. code-block:: bash

   python main.py --context 4096

Or specifying the context:

Ou especificando o contexto:

.. code-block:: bash

   python main.py -c 8192


Command-Line Options
--------------------

.. code-block:: bash

   python main.py [OPTIONS]

Available options:

Opções disponíveis:

.. code-block:: bash

   -h, --help            Show help message and exit
   -c, --context CONTEXT Tamanho do contexto em tokens (default: 4096)
   --gpu-type TYPE       Tipo de GPU: consumer, datacenter, all (default: all)
   --only-runs           Mostrar apenas combinações que rodam
   --group-gpu           Agrupar resultados por GPU em vez de por modelo
   --summary TYPE        Resumo: model, gpu, both, none (default: both)
   --export-csv FILE     Exportar resultados para CSV
   --export-json FILE    Exportar resultados para JSON
   -q, --quantization Q  Precisão: fp32, fp16, int8, int4 (default: fp16)


Usage Examples
--------------

Consumer GPUs only:

Apenas GPUs consumer:

.. code-block:: bash

   python main.py -c 8192 --gpu-type consumer

Show only viable combinations:

Mostrar apenas combinações viáveis:

.. code-block:: bash

   python main.py -c 4096 --only-runs

Export results to JSON:

Exportar resultados para JSON:

.. code-block:: bash

   python main.py -c 8192 --export-json results.json

Use INT4 quantization for larger models:

Usar quantização INT4 para modelos maiores:

.. code-block:: bash

   python main.py -c 8192 -q int4 --only-runs


Supported Precisions
--------------------

FP32 (Float32)
    4 bytes per parameter. Highest precision, highest VRAM usage.
    Raramente usado para inferência de LLMs devido ao alto consumo.
    Rarely used for LLM inference due to high VRAM usage.

FP16 (Float16)
    2 bytes per parameter. Industry standard for inference.
    Excelente precisão com metade do consumo do FP32.
    Excellent precision with half the VRAM usage of FP32.

INT8 (8-bit Integer)
    1 byte per parameter. Aggressive quantization.
    Pequena perda de qualidade com economia significativa.
    Small quality loss with significant VRAM savings.

INT4 (4-bit Integer)
    0.5 byte per parameter. Very aggressive quantization.
    Maior economia, mas perda de qualidade mais perceptível.
    Highest savings but more noticeable quality loss.


Interpreting Results
--------------------

``RUNS`` (green)
    The GPU has sufficient VRAM to run the model with the specified context.
    A GPU tem VRAM suficiente para rodar o modelo com o contexto especificado.

``DOESN'T RUN`` (red)
    The GPU doesn't have enough VRAM.
    A GPU não tem VRAM suficiente.

Low safety margin warning
    ``⚠️ Low safety margin`` indicates the combination runs but with less
    than 10% VRAM free. This may cause OOM in real scenarios due to
    implementation variations.

    ``⚠️ Margem de segurança baixa`` indica que a combinação roda mas com
    menos de 10% de VRAM livre. Isso pode causar OOM em cenários reais
    devido a variações de implementação.


VRAM Calculation Model
----------------------

The VRAM calculation considers four components:

O cálculo de VRAM considera quatro componentes:

1. **Model Parameters**

   .. code-block:: python

      params_memory_gb = (params_billion * bytes_per_param) / 1024

2. **Overhead** (runtime, activations, etc.)

   .. code-block:: python

      overhead_gb = params_memory_gb * 0.30

3. **KV Cache**

   .. code-block:: python

      kv_cache_gb = (kv_cache_mb_per_token * context_tokens * multiplier) / 1024

4. **Total**

   .. code-block:: python

      total_vram_gb = model_with_overhead_gb + kv_cache_gb

Where ``multiplier`` depends on precision:
Onde ``multiplier`` depende da precisão:

* FP32: 2.0
* FP16: 1.0
* INT8: 0.6
* INT4: 0.6
