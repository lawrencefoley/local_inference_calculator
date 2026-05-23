Exemplos / Examples
====================

Esta seção fornece exemplos práticos de uso do Local Inference Calculator.

This section provides practical examples of using the Local Inference Calculator.

Cenário 1: GPU Gamer com 8GB
-----------------------------

Você tem uma RTX 3060 (8GB) e quer saber quais modelos rodam.

You have an RTX 3060 (8GB) and want to know which models run.

.. code-block:: bash

   uv run llmfit -c 4096 --gpu-type consumer --only-runs

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

   uv run llmfit -m 7 -c 16384

Isso mostrará se 7B com 16K contexto cabe em 24GB.

This will show if 7B with 16K context fits in 24GB.


Cenário 3: Comparando GPUs Datacenter
--------------------------------------

Comparando A100 (40GB) vs A100 (80GB) para modelo 70B.

Comparing A100 (40GB) vs A100 (80GB) for 70B model.

.. code-block:: bash

   uv run llmfit -m 70 -c 8192 -q int4 --gpu-type datacenter


Cenário 4: Planejamento de Contexto Longo
------------------------------------------

Projetando requisitos de VRAM para contexto de 32K.

Projecting VRAM requirements for 32K context.

.. code-block:: bash

   uv run llmfit -m 13 -c 32768

A ferramenta mostrará o KV cache projetado para contextos maiores.

The tool will show projected KV cache for larger contexts.


Cenário 5: Exportando Resultados
---------------------------------

Exportando todas as combinações para análise posterior.

Exporting all combinations for later analysis.

.. code-block:: bash

   uv run llmfit -c 8192 --export-json analysis.json

Ou em CSV para planilhas:

Or in CSV for spreadsheets:

.. code-block:: bash

   uv run llmfit -c 8192 --export-csv analysis.csv


Cenário 6: Múltiplos Modelos
-----------------------------

Verificando vários tamanhos de modelo de uma vez.

Checking multiple model sizes at once.

.. code-block:: bash

   # Shell script para verificar todos os tamanhos
   for size in 7 13 34 70; do
       echo "=== Modelo ${size}B ==="
       uv run llmfit -m $size -c 8192
   done


Casos de Uso Avançados
-----------------------

Batch Processing
~~~~~~~~~~~~~~~~

Processando múltiplos contextos:

.. code-block:: bash

   for ctx in 4096 8192 16384 32768; do
       uv run llmfit -m 13 -c $ctx --export-json results_${ctx}.json
   done

Comparação de Quantização
~~~~~~~~~~~~~~~~~~~~~~~~~~

Comparando diferentes níveis de quantização:

.. code-block:: bash

   uv run llmfit -m 70 -c 8192 -q fp16   # ~175 GB
   uv run llmfit -m 70 -c 8192 -q int8   # ~105 GB
   uv run llmfit -m 70 -c 8192 -q int4   # ~72 GB


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


Exemplos Avançados (v0.2.0)
----------------------------

Exemplo 9: Otimização de Offload de Camadas
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Calcule a distribuição ótima de camadas na GPU para um modelo que não cabe completamente na VRAM:

Calculate optimal GPU layer distribution for a model that doesn't fully fit in VRAM:

.. code-block:: bash

   uv run llmfit --model 70 --context 8192 --optimize-config --quantization int4

Saída:

Output:

.. code-block:: text

   CONFIGURAÇÃO ÓTIMA DE OFFLOAD DE CAMADAS
   ======================================================================
   Modelo: LLaMA 2 70B / LLaMA 3.1 70B (70B parâmetros)
   GPU:   RTX 3090 (24 GB VRAM)
   Total de Camadas: 80

   Distribuição de Camadas:
     Camadas na GPU:  0
     Camadas na CPU:  80
     Razão de Offload:  0.0%

   Uso de Memória:
     VRAM da GPU usada:  32.50 GB / 24 GB
     RAM da CPU usada:   45.00 GB

   Impacto na Performance:
     Desaceleração estimada: 1000%

   Configuração Recomendada:
     llama.cpp:  --gpu-layers 0
     AutoGPTQ:   --gpu-memory 32.5G

Para GPUs com mais VRAM, você verá opções de offload parcial.

For GPUs with more VRAM, you'll see partial offload options.


Exemplo 10: Análise de Offload de CPU
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Analise inferência híbrida GPU+CPU com considerações de largura de banda PCIe:

Analyze hybrid GPU+CPU inference with PCIe bandwidth considerations:

.. code-block:: bash

   uv run llmfit --model 13 --context 8192 --cpu-offload --system-ram 64 --pcie-gen 4.0

Saída:

Output:

.. code-block:: text

   ANÁLISE DE OFFLOAD DE CPU
   ======================================================================
   Requisitos do Sistema:
     RAM do sistema necessária: 8.50 GB
     RAM do sistema disponível: 64.00 GB
     Status: ✓ Cabe na RAM do sistema

   Configuração PCIe:
     Geração: PCIe 4.0
     Largura de banda: ~24 GB/s efetiva

   Estimativa de Performance:
     Velocidade de tokens: ~35.0 tokens/segundo
     Razão de velocidade: 100.0% da GPU completa

   Distribuição de Camadas:
     Camadas na GPU:  40
     Camadas na CPU:  0
     Razão de Offload:  100.0%

Impacto da geração PCIe:

Impact of PCIe generation:

* **PCIe 3.0**: ~12 GB/s efetivo (transferência de dados mais lenta)
* **PCIe 4.0**: ~24 GB/s efetivo (padrão para GPUs modernas)
* **PCIe 5.0**: ~48 GB/s efetivo (melhor para offload de CPU)


Exemplo 11: Configuração Multi-GPU
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Calcule paralelismo tensor entre múltiplas GPUs:

Calculate tensor parallelism across multiple GPUs:

.. code-block:: bash

   uv run llmfit --params-b 405 --context 8192 --quantization int4 --multi-gpu --gpu-config "2x4090,1x3090"

Saída:

Output:

.. code-block:: text

   CONFIGURAÇÃO MULTI-GPU
   ======================================================================
   Modelo: Custom Model 405B (405B parâmetros)
   Status: NÃO RODA
     Gargalo: RTX 3090
     Overhead de comunicação: 5.28 GB

   Alocação por GPU:
     ✗ RTX 4090              25.32 GB / 24 GB
       Shard: 50.0%

   Configuração de Framework:
     tensor_parallel_size: 3
     mode: tensor_parallel
     vllm: --tensor-parallel-size 3

Configurações heterogêneas (modelos de GPU mistos) são suportadas.
A alocação de cada GPU é proporcional à sua capacidade de VRAM.

Heterogeneous configurations (mixed GPU models) are supported.
Each GPU's allocation is proportional to its VRAM capacity.


Exemplo 12: Auto-detecção GGUF
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Auto-detecte quantização a partir do nome do arquivo GGUF:

Auto-detect quantization from GGUF filename:

.. code-block:: bash

   uv run llmfit --gguf-file "llama-2-7b.Q4_K_M.gguf" --context 4096

Saída:

Output:

.. code-block:: text

   Arquivo GGUF detectado: llama-2-7b.Q4_K_M.gguf
     Quantização: Q4_K_M
     Bits efetivos: 5.50 bits/param
     Usando quantização: int4

Quantizações GGUF suportadas:

Supported GGUF quantizations:

* Q2_K, Q2_K_S, Q2_K_M, Q2_K_L (~3-4 bits efetivos)
* Q3_K, Q3_K_S, Q3_K_M, Q3_K_L, Q3_K_XS (~4-5 bits)
* Q4_K, Q4_K_S, Q4_K_M, Q4_0, Q4_1 (~5 bits)
* Q5_K, Q5_K_S, Q5_K_M, Q5_0, Q5_1 (~6 bits)
* Q6_K (~7 bits)
* Q8_0 (8 bits)
* F16, F32 (ponto flutuante)


Uso Programático Avançado
~~~~~~~~~~~~~~~~~~~~~~~~~~

Usando os calculadores avançados programaticamente:

Using the advanced calculators programmatically:

.. code-block:: python

   from calculator import LayerOffloadCalculator, CPUOffloadCalculator
   from multi_gpu import MultiGPUCalculator, MultiGPUConfig, MultiGPUMode
   from formats import detect_gguf_quantization
   from models import get_model_by_size
   from gpus import get_all_gpus

   # Cálculo de offload de camadas
   layer_calc = LayerOffloadCalculator(quantization=Quantization.INT4)
   model = get_model_by_size(70)
   gpu = get_all_gpus()[0]  # RTX 3060

   result = layer_calc.calculate_optimal_offload(model, gpu, context_tokens=8192)
   print(f"Camadas na GPU: {result.layers_on_gpu}/{result.total_layers}")
   print(f"Recomendado: --gpu-layers {result.layers_on_gpu}")

   # Cálculo multi-GPU
   multi_calc = MultiGPUCalculator(quantization=Quantization.INT4)
   config = MultiGPUConfig(
       gpus=[gpu1, gpu2, gpu3],
       mode=MultiGPUMode.TENSOR_PARALLEL
   )
   result = multi_calc.calculate(model, config, context_tokens=8192)

   # Detecção GGUF
   gguf_info = detect_gguf_quantization("llama-2-7b.Q4_K_M.gguf")
   print(f"Quantização: {gguf_info.quant_name}")
   print(f"Bits efetivos: {gguf_info.bits_per_param}")
