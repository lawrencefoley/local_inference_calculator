Examples
========

Example 1: Which GPU to buy for a 7B model?
---------------------------------------------

We want to run a 7B model with context 8192. What's the minimum GPU required?

Queremos rodar um modelo 7B com contexto 8192. Qual a mínima GPU necessária?

.. code-block:: bash

   python main.py -c 8192 --gpu-type consumer --only-runs

Result shows that even 8GB GPUs are sufficient for a 7B in FP16.

O resultado mostraria que mesmo GPUs de 8GB são suficientes para um 7B em FP16.

---

Example 2: Can I run a 70B model on my RTX 4090?
--------------------------------------------------

.. code-block:: bash

   python main.py -c 4096 --gpu-type consumer --only-runs | grep "70B"

Output:

Saída:

.. code-block:: text

   70B    14.2    RTX 3090        24    RUNS
   70B    14.2    RTX 4090        24    RUNS

Yes! A 70B model in FP16 fits on an RTX 4090 with context 4096.

Sim! Um modelo 70B em FP16 cabe em uma RTX 4090 com contexto 4096.

---

Example 3: What about context 16384?
-------------------------------------

.. code-block:: bash

   python main.py -c 16384 -q fp16 --gpu-type consumer | grep "70B"

The 70B no longer runs on RTX 4090. Let's try INT8:

O 70B não roda mais na RTX 4090. Vamos tentar INT8:

.. code-block:: bash

   python main.py -c 16384 -q int8 --gpu-type consumer | grep "70B"

Now it works! With INT8, the 70B runs with context 16384.

Agora sim! Com INT8, o 70B roda com contexto 16384.

---

Example 4: Comparing all quantizations
---------------------------------------

.. code-block:: bash

   for q in fp32 fp16 int8 int4; do
       echo "=== $q ==="
       python main.py -c 8192 -q $q --only-runs --summary none | head -5
   done

This clearly shows how INT4 allows much larger models on the same GPU.

Isso mostra claramente como INT4 permite modelos muito maiores na mesma GPU.

---

Example 5: Exporting for analysis
----------------------------------

.. code-block:: bash

   python main.py -c 8192 --export-json results.json

.. code-block:: bash

   python main.py -c 8192 --export-csv results.csv

The exported files contain all combinations for further analysis.

Os arquivos exportados contêm todas as combinações para análise posterior.

---

Programmatic Usage
------------------

You can also use the library directly in Python:

Você também pode usar a biblioteca diretamente em Python:

.. code-block:: python

   from calculator import VRAMCalculator, Quantization
   from models import get_all_models
   from gpus import get_consumer_gpus

   # Create calculator with INT4 quantization
   # Criar calculadora com quantização INT4
   calc = VRAMCalculator(quantization=Quantization.INT4)

   # Calculate for context 8192
   # Calcular para contexto 8192
   model = get_all_models()[0]  # 7B
   gpu = get_consumer_gpus()[0]  # RTX 3060

   result = calc.evaluate_pair(model, gpu, context_tokens=8192)

   print(f"Status: {result.status.value}")
   print(f"VRAM required: {result.required_vram_gb:.1f} GB")
   print(f"VRAM available: {result.gpu_vram_gb} GB")
