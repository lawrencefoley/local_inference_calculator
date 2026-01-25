Examples
========

Example 1: List Available Models
---------------------------------

.. code-block:: bash

   python main.py --list-models

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

   python main.py --model 70 --context 8192

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

Example 3: Can I Run a 70B Model with INT4 on RTX 4090?
---------------------------------------------------------

.. code-block:: bash

   python main.py -m 70 -c 16384 -q int4

This shows that with INT4 quantization, a 70B model with 16k context
requires ~34 GB, so it fits on an RTX 4090 (24 GB) but would need
quantization or a larger GPU for the full context.

---

Example 4: Which GPU to Buy for a 34B Model?
---------------------------------------------

.. code-block:: bash

   python main.py --model 34 --context 8192

The output will show all GPUs that can run the model, ranked by
free VRAM percentage.

---

Example 5: Compare All Quantizations
---------------------------------------

.. code-block:: bash

   for q in fp32 fp16 int8 int4; do
       echo "=== $q ==="
       python main.py -m 7 -c 8192 -q $q | grep "TOTAL VRAM"
   done

This clearly shows how INT4 allows much larger models on the same GPU.

---

Example 6: Check Google Colab Compatibility
---------------------------------------------

.. code-block:: bash

   python main.py --model 13 --context 16384 -q int4 --gpu-type datacenter | grep -i colab

This quickly shows which Colab GPU tier can run your desired model.

---

Example 7: Export Results
-------------------------

.. code-block:: bash

   python main.py -c 8192 --export-json results.json

.. code-block:: bash

   python main.py -c 8192 --export-csv results.csv

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
