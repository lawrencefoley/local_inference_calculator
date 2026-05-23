User Guide
==========

Basic Usage
------------

List Available Models
~~~~~~~~~~~~~~~~~~~~

To see all available models in the database:

.. code-block:: bash

   uv run llmfit --list-models

This shows model name, size, architecture, and KV cache requirements for each model.

Check Specific Model
~~~~~~~~~~~~~~~~~~~~~

To check VRAM requirements for a specific model:

.. code-block:: bash

   uv run llmfit --model 7 --context 8192
   uv run llmfit -m 70 -c 16384 -q int4

The output includes:

* **VRAM Breakdown**: Parameters, overhead, and KV cache memory
* **Total VRAM Required**: Minimum and recommended GPU VRAM
* **GPU Compatibility**: List of GPUs that can run the model, with free VRAM percentage

All Combinations
~~~~~~~~~~~~~~~~

To see all model × GPU combinations:

.. code-block:: bash

   uv run llmfit --context 4096

Or with a larger context:

.. code-block:: bash

   uv run llmfit -c 8192


Max Context for a VRAM Budget
-----------------------------

Use ``--vram-gb`` with a quantization to see which models fit and their estimated maximum context:

.. code-block:: bash

   uv run llmfit --vram-gb 24 --quantization int4
   uv run llmfit --vram-gb 16 --quantization fp16 --mode conservative

Combine it with ``--model``, ``--params-b``, or ``--config-json`` to check a single model.

Hugging Face config.json
------------------------

Derive KV cache metadata from a Hugging Face ``config.json``:

.. code-block:: bash

   uv run llmfit --config-json path/to/config.json --params-b 7 --context 8192

If the config does not include a parameter count, pass it with ``--params-b``. Use ``--model-name`` to override the display name.

Command-Line Options
--------------------

.. code-block:: bash

   uv run llmfit [OPTIONS]

Basic Options
~~~~~~~~~~~~~

.. code-block:: bash

   -h, --help            Show help message and exit
   -c, --context CONTEXT Context size in tokens (default: 4096)
   --list-models         List all available models
   -m SIZE, --model SIZE  Model size in billions (e.g., 0.6, 7, 13, 70)
   --gpu-type TYPE       GPU type: consumer, datacenter, all (default: all)
   --only-runs           Show only running combinations
   --group-gpu           Group results by GPU instead of model
   --summary TYPE        Summary: model, gpu, both, none (default: both)
   --export-csv FILE     Export results to CSV
   --export-json FILE    Export results to JSON
   -q, --quantization Q  Model precision (fp32, fp16, int8, int4)
   --mode MODE           Calculation mode (theoretical, conservative, production)

Advanced Options (v0.2.0)
~~~~~~~~~~~~~~~~~~~~~~~~

Layer Offload Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   --optimize-config      Show optimal layer offload configuration

Calculates how many transformer layers can fit in GPU VRAM, displaying:

* Layers on GPU vs CPU
* Recommended ``--gpu-layers`` parameter for llama.cpp
* Performance impact estimation
* Offload options for all available GPUs

CPU Offload Analysis
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   --cpu-offload          Enable CPU offload calculations
   --system-ram GB        System RAM available in GB (default: 32.0)
   --pcie-gen GEN         PCIe generation: 3.0, 4.0, or 5.0 (default: 4.0)

Calculates hybrid GPU+CPU inference configuration:

* System RAM requirements
* PCIe bandwidth impact on performance
* Estimated tokens/second
* Layer distribution between GPU and CPU

Multi-GPU Support
^^^^^^^^^^^^^^^^^

.. code-block:: bash

   --multi-gpu            Enable multi-GPU mode
   --gpu-config CONFIG    Multi-GPU configuration (e.g., "2x4090,1x3090")
   --multi-gpu-mode MODE  Parallelism mode: tensor or pipeline (default: tensor)

Configuration format:

* Homogeneous: ``3x4090`` (3 identical GPUs)
* Heterogeneous: ``2x4090,1x3090`` (mixed GPUs)
* Partial names: ``2x3090,1x4090``

Modes:

* **tensor** - Model weights split across GPUs (same layers, different shards)
* **pipeline** - Different layers on different GPUs

Model Format Support
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   --gguf-file FILENAME  GGUF filename to auto-detect quantization
   --format FORMAT        Model format: fp16, gguf, exl2, gptq, awq (default: fp16)

GGUF auto-detection supports: Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_0, F16, F32

Format overhead multipliers:

* FP16: 1.0x (baseline)
* GGUF: 1.15x (+15% for metadata structure)
* EXL2: 1.05x (+5% optimized layout)
* GPTQ: 1.10x (+10% quantization metadata)
* AWQ: 1.08x (+8% activation-aware quantization)


Usage Examples
--------------

Consumer GPUs only:

.. code-block:: bash

   uv run llmfit -c 8192 --gpu-type consumer

Show only viable combinations:

.. code-block:: bash

   uv run llmfit -c 4096 --only-runs

Export results to JSON:

.. code-block:: bash

   uv run llmfit -c 8192 --export-json results.json

Use INT4 quantization for larger models:

.. code-block:: bash

   uv run llmfit -c 8192 -q int4 --only-runs

Check if a 70B model fits on RTX 4090:

.. code-block:: bash

   uv run llmfit --model 70 --context 8192


Supported Precisions
--------------------

FP32 (Float32)
    4 bytes per parameter. Highest precision, highest VRAM usage.
    Rarely used for LLM inference due to high VRAM usage.

FP16 (Float16)
    2 bytes per parameter. Industry standard for inference.
    Excellent precision with half the VRAM usage of FP32.

INT8 (8-bit Integer)
    1 byte per parameter. Aggressive quantization.
    Small quality loss with significant VRAM savings.

INT4 (4-bit Integer)
    0.5 byte per parameter. Very aggressive quantization.
    Highest savings but more noticeable quality loss.


Interpreting Results
--------------------

``RUNS`` (green)
    The GPU has sufficient VRAM to run the model with the specified context.

``DOESN'T RUN`` (red)
    The GPU doesn't have enough VRAM.

Low safety margin warning
    ``⚠️ Low safety margin`` indicates the combination runs but with less
    than 10% VRAM free. This may cause OOM in real scenarios due to
    implementation variations.


VRAM Calculation Formula
-----------------------

The tool uses a conservative model with four components:

**1. Model Parameters**

Base memory for model weights. Since params_billion is in billions:

.. code-block:: python

   params_memory_gb = params_billion × bytes_per_param

Example: 70B model in FP16 (2 bytes/param):
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


Technical Notes
---------------

The calculations assume PyTorch-style memory allocation (HF Transformers, vLLM).
Different backends may have varying memory behavior:

* **TensorRT-LLM**: Custom allocators, ~10-20% less memory
* **llama.cpp (GGUF)**: Memory-mapped files, ~20-30% less memory
* **EXL2**: Optimized allocation, minimal overhead

For detailed explanations of technical terms used in VRAM calculations,
see :doc:`glossary`.
