Examples
========

Example 1: List Available Models
---------------------------------

.. code-block:: bash

   uv run llmfit --list-models

Output:

.. code-block:: text

   AVAILABLE MODELS
   ======================================================================
     [7B] LLaMA 2 / Mistral / Qwen 7B
          Architecture: decoder-only
          Default precision: fp16
          KV cache: 0.6 MB/token (FP16 baseline)

     [13B] LLaMA 2 13B
          Architecture: decoder-only
          Default precision: fp16
          KV cache: 0.9 MB/token (FP16 baseline)

     ...

---

Example 2: How Much VRAM Do I Need for a 70B Model?
----------------------------------------------------

.. code-block:: bash

   uv run llmfit --model 70 --context 8192

Output:

.. code-block:: text

   VRAM BREAKDOWN: LLaMA 2 70B / LLaMA 3.1 70B
   ======================================================================
   Configuration:
     Context: 8,192 tokens
     Quantization: FP16

   Memory Breakdown:
     Model parameters: 13.67 GB
     Overhead (30%):      4.10 GB
     Model + overhead:    17.77 GB
     KV cache:            28.67 GB
     ----------------------------------------
     TOTAL VRAM:          46.44 GB

   Minimum GPU VRAM required: 46.4 GB
   Recommended (with margin): 51.1 GB

---

Example 3: Small Model with Fractional Size (0.6B)
--------------------------------------------------

Small models like Phi-3 Mini (3.8B) or Qwen2-0.5B use fractional sizes:

.. code-block:: bash

   uv run llmfit --model 0.6 --context 8192
   uv run llmfit -m 3.8 -c 4096 -q int4

These models are designed for edge devices and can run on GPUs with as little as 4-6 GB VRAM.

---

Example 4: Can I Run a 70B Model with INT4 on RTX 4090?
---------------------------------------------------------

.. code-block:: bash

   uv run llmfit -m 70 -c 16384 -q int4

This shows that with INT4 quantization, a 70B model with 16k context
requires ~34 GB, so it fits on an RTX 4090 (24 GB) but would need
quantization or a larger GPU for the full context.

---

Example 5: Which GPU to Buy for a 34B Model?
---------------------------------------------

.. code-block:: bash

   uv run llmfit --model 34 --context 8192

The output will show all GPUs that can run the model, ranked by
free VRAM percentage.

---

Example 6: Compare All Quantizations
---------------------------------------

.. code-block:: bash

   for q in fp32 fp16 int8 int4; do
       echo "=== $q ==="
       uv run llmfit -m 7 -c 8192 -q $q | grep "TOTAL VRAM"
   done

This clearly shows how INT4 allows much larger models on the same GPU.

---

Example 7: Check Google Colab Compatibility
---------------------------------------------

.. code-block:: bash

   uv run llmfit --model 13 --context 16384 -q int4 --gpu-type datacenter | grep -i colab

This quickly shows which Colab GPU tier can run your desired model.

---

Example 8: Export Results
-------------------------

.. code-block:: bash

   uv run llmfit -c 8192 --export-json results.json

.. code-block:: bash

   uv run llmfit -c 8192 --export-csv results.csv

The exported files contain all combinations for further analysis.

---

Programmatic Usage
------------------

You can also use the library directly in Python:

.. code-block:: python

   from calculator import VRAMCalculator, Quantization
   from models import get_all_models
   from gpus import get_all_gpus

   # Create calculator with INT4 quantization
   calc = VRAMCalculator(quantization=Quantization.INT4)

   # Calculate for context 8192
   model = get_all_models()[0]  # 7B
   gpu = get_all_gpus()[0]  # RTX 3060

   result = calc.evaluate_pair(model, gpu, context_tokens=8192)

   print(f"Status: {result.status.value}")
   print(f"VRAM required: {result.required_vram_gb:.1f} GB")
   print(f"VRAM available: {result.gpu_vram_gb} GB")

---
Advanced Examples (v0.2.0)
--------------------------

Example 9: Layer Offload Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Calculate optimal GPU layer distribution for a model that doesn't fully fit in VRAM:

.. code-block:: bash

   uv run llmfit --model 70 --context 8192 --optimize-config --quantization int4

Output:

.. code-block:: text

   OPTIMAL LAYER OFFLOAD CONFIGURATION
   ======================================================================
   Model: LLaMA 2 70B / LLaMA 3.1 70B (70B parameters)
   GPU:   RTX 3090 (24 GB VRAM)
   Total Layers: 80

   Layer Distribution:
     Layers on GPU:  0
     Layers on CPU:  80
     Offload Ratio:  0.0%

   Memory Usage:
     GPU VRAM used:  32.50 GB / 24 GB
     CPU RAM used:   45.00 GB

   Performance Impact:
     Estimated slowdown: 1000%

   Recommended Configuration:
     llama.cpp:  --gpu-layers 0
     AutoGPTQ:   --gpu-memory 32.5G

For GPUs with more VRAM, you'll see partial offload options.

---
Example 10: CPU Offload Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Analyze hybrid GPU+CPU inference with PCIe bandwidth considerations:

.. code-block:: bash

   uv run llmfit --model 13 --context 8192 --cpu-offload --system-ram 64 --pcie-gen 4.0

Output:

.. code-block:: text

   CPU OFFLOAD ANALYSIS
   ======================================================================
   System Requirements:
     System RAM required: 8.50 GB
     System RAM available: 64.00 GB
     Status: ✓ Fits in system RAM

   PCIe Configuration:
     Generation: PCIe 4.0
     Bandwidth: ~24 GB/s effective

   Performance Estimate:
     Token speed: ~35.0 tokens/second
     Speed ratio: 100.0% of full GPU

   Layer Distribution:
     Layers on GPU:  40
     Layers on CPU:  0
     Offload Ratio:  100.0%

PCIe generation impact:

* **PCIe 3.0**: ~12 GB/s effective (slower data transfer)
* **PCIe 4.0**: ~24 GB/s effective (standard for modern GPUs)
* **PCIe 5.0**: ~48 GB/s effective (best for CPU offload)

---
Example 11: Multi-GPU Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Calculate tensor parallelism across multiple GPUs:

.. code-block:: bash

   uv run llmfit --params-b 405 --context 8192 --quantization int4 --multi-gpu --gpu-config "2x4090,1x3090"

Output:

.. code-block:: text

   MULTI-GPU CONFIGURATION
   ======================================================================
   Model: Custom Model 405B (405B parameters)
   Status: DOESN'T RUN
     Bottleneck: RTX 3090
     Communication overhead: 5.28 GB

   Per-GPU Allocation:
     ✗ RTX 4090              25.32 GB / 24 GB
       Shard: 50.0%

   Framework Configuration:
     tensor_parallel_size: 3
     mode: tensor_parallel
     vllm: --tensor-parallel-size 3

Heterogeneous configurations (mixed GPU models) are supported.
Each GPU's allocation is proportional to its VRAM capacity.

---
Example 12: GGUF Auto-Detection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Auto-detect quantization from GGUF filename:

.. code-block:: bash

   uv run llmfit --gguf-file "llama-2-7b.Q4_K_M.gguf" --context 4096

Output:

.. code-block:: text

   GGUF file detected: llama-2-7b.Q4_K_M.gguf
     Quantization: Q4_K_M
     Effective bits: 5.50 bits/param
     Using quantization: int4

Supported GGUF quantizations:

* Q2_K, Q2_K_S, Q2_K_M, Q2_K_L (~3-4 bits effective)
* Q3_K, Q3_K_S, Q3_K_M, Q3_K_L, Q3_K_XS (~4-5 bits)
* Q4_K, Q4_K_S, Q4_K_M, Q4_0, Q4_1 (~5 bits)
* Q5_K, Q5_K_S, Q5_K_M, Q5_0, Q5_1 (~6 bits)
* Q6_K (~7 bits)
* Q8_0 (8 bits)
* F16, F32 (floating point)

---
Example 13: Model Format Comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compare VRAM requirements across different model formats:

.. code-block:: bash

   uv run llmfit --model 7 --context 8192 --format fp16
   uv run llmfit --model 7 --context 8192 --format gguf
   uv run llmfit --model 7 --context 8192 --format exl2

Format overhead is automatically applied to calculations.

---
Advanced Programmatic Usage
~~~~~~~~~~~~~~~~~~~~~~~~~~

Using the advanced calculators programmatically:

.. code-block:: python

   from calculator import LayerOffloadCalculator, CPUOffloadCalculator
   from multi_gpu import MultiGPUCalculator, MultiGPUConfig, MultiGPUMode
   from formats import detect_gguf_quantization
   from models import get_model_by_size
   from gpus import get_all_gpus

   # Layer offload calculation
   layer_calc = LayerOffloadCalculator(quantization=Quantization.INT4)
   model = get_model_by_size(70)
   gpu = get_all_gpus()[0]  # RTX 3060

   result = layer_calc.calculate_optimal_offload(model, gpu, context_tokens=8192)
   print(f"Layers on GPU: {result.layers_on_gpu}/{result.total_layers}")
   print(f"Recommended: --gpu-layers {result.layers_on_gpu}")

   # Multi-GPU calculation
   multi_calc = MultiGPUCalculator(quantization=Quantization.INT4)
   config = MultiGPUConfig(
       gpus=[gpu1, gpu2, gpu3],
       mode=MultiGPUMode.TENSOR_PARALLEL
   )
   result = multi_calc.calculate(model, config, context_tokens=8192)

   # GGUF detection
   gguf_info = detect_gguf_quantization("llama-2-7b.Q4_K_M.gguf")
   print(f"Quantization: {gguf_info.quant_name}")
   print(f"Effective bits: {gguf_info.bits_per_param}")
