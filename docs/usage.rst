User Guide
==========

Basic Usage
------------

List Available Models
~~~~~~~~~~~~~~~~~~~~

To see all available models in the database:

.. code-block:: bash

   python main.py --list-models

This shows model name, size, architecture, and KV cache requirements for each model.

Check Specific Model
~~~~~~~~~~~~~~~~~~~~~

To check VRAM requirements for a specific model:

.. code-block:: bash

   python main.py --model 7 --context 8192
   python main.py -m 70 -c 16384 -q int4

The output includes:

* **VRAM Breakdown**: Parameters, overhead, and KV cache memory
* **Total VRAM Required**: Minimum and recommended GPU VRAM
* **GPU Compatibility**: List of GPUs that can run the model, with free VRAM percentage

All Combinations
~~~~~~~~~~~~~~~~

To see all model × GPU combinations:

.. code-block:: bash

   python main.py --context 4096

Or with a larger context:

.. code-block:: bash

   python main.py -c 8192


Command-Line Options
--------------------

.. code-block:: bash

   python main.py [OPTIONS]

Available options:

.. code-block:: bash

   -h, --help            Show help message and exit
   -c, --context CONTEXT Context size in tokens (default: 4096)
   --list-models         List all available models
   -m SIZE, --model SIZE  Model size in billions (e.g., 7, 13, 70)
   --gpu-type TYPE       GPU type: consumer, datacenter, all (default: all)
   --only-runs           Show only running combinations
   --group-gpu           Group results by GPU instead of model
   --summary TYPE        Summary: model, gpu, both, none (default: both)
   --export-csv FILE     Export results to CSV
   --export-json FILE    Export results to JSON
   -q, --quantization Q  Model precision (fp32, fp16, int8, int4)


Usage Examples
--------------

Consumer GPUs only:

.. code-block:: bash

   python main.py -c 8192 --gpu-type consumer

Show only viable combinations:

.. code-block:: bash

   python main.py -c 4096 --only-runs

Export results to JSON:

.. code-block:: bash

   python main.py -c 8192 --export-json results.json

Use INT4 quantization for larger models:

.. code-block:: bash

   python main.py -c 8192 -q int4 --only-runs

Check if a 70B model fits on RTX 4090:

.. code-block:: bash

   python main.py --model 70 --context 8192


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
