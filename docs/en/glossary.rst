Glossary / Glossário
====================

This section explains technical terms used throughout the documentation.

Esta seção explica termos técnicos utilizados ao longo da documentação.

Memory Allocator
----------------

**PyTorch-style (HF Transformers, vLLM)**

The default memory allocation pattern used by PyTorch-based frameworks. Memory is allocated
contiguously for model weights and activations, with additional overhead for CUDA operations.
This allocator tends to be conservative in memory management, reserving more VRAM than strictly
necessary to avoid fragmentation and ensure smooth operation.

Framework characteristics:

* **Hugging Face Transformers**: Uses standard PyTorch CUDA allocator with additional overhead
  for model loading and optimization passes
* **vLLM**: Implements PagedAttention with more efficient memory management, but still follows
  PyTorch-style allocation patterns for model weights

**Note: TensorRT-LLM, llama.cpp, EXL2**

These frameworks use different memory allocation strategies that may result in lower VRAM usage:

* **TensorRT-LLM**: Uses custom memory allocators optimized for TensorRT engines, with more
  aggressive memory pooling and less overhead
* **llama.cpp (GGUF)**: Uses quantization-aware allocation with memory-mapped files,
  often requiring 20-30% less VRAM than PyTorch equivalents
* **EXL2 (ExLlamaV2)**: Implements extremely optimized memory allocation with minimal overhead,
  supporting variable-rate quantization per layer

When calculating VRAM requirements, this tool assumes PyTorch-style allocation for consistency.
If using these alternative frameworks, actual VRAM usage may be 10-30% lower than estimated.

LoRA Adapters
-------------

**Low-Rank Adaptation** is a parameter-efficient fine-tuning technique that adds trainable
rank decomposition matrices to existing model weights instead of fine-tuning all parameters.

Memory impact:

* **Per adapter**: +0.5 to 2 GB VRAM depending on rank and model size
* **Rank 8-16**: Typical configuration, ~0.5-1 GB overhead
* **Rank 32-64**: Higher capacity adapters, ~1-2 GB overhead
* **Multiple adapters**: Memory scales linearly with active adapters

VRAM calculation impact:

.. code-block:: python

   # With LoRA, add adapter memory to total
   total_vram = base_model + kv_cache + lora_adapter_overhead

The tool assumes no active LoRA adapters by default. If using LoRA, add 0.5-2 GB per
adapter to the estimated VRAM requirements.

Speculative Decoding
--------------------

Also known as **draft decoding** or **assistant decoding**, this technique uses a smaller
draft model to predict tokens that are then verified by the target model.

Memory impact:

* **Draft model overhead**: +30-50% additional VRAM for the draft model
* **Example**: Using a 7B draft model with a 70B target requires memory for both models
* **Typical draft size**: 10-20% of target model parameters

Trade-offs:

* **Pros**: 2-3x faster token generation
* **Cons**: Significantly higher memory usage, requires two models in memory

The tool assumes no speculative decoding. If enabled, calculate memory for both models
and add 10-15% for verification overhead.

Aggressive Paged KV
-------------------

**Paged KV Cache** is a memory management technique inspired by operating system paging,
where the KV cache is divided into fixed-size blocks (pages) that can be allocated and
deallocated dynamically.

Implementation variants:

* **Standard paged KV**: vLLM's default implementation, efficient memory usage
* **Aggressive paged KV**: More aggressive pre-allocation and caching strategies

Memory impact:

* **Standard**: 10-20% reduction in KV cache memory vs contiguous allocation
* **Aggressive**: Additional 5-10% savings but higher CPU overhead for page management

The tool uses conservative KV cache estimates. Aggressive paged KV implementations may
use 10-30% less memory than calculated for the KV cache component.

KV Cache (Key-Value Cache)
---------------------------

The **KV cache** stores attention keys and values for all previously generated tokens
in a sequence, avoiding recomputation during autoregressive generation.

Memory calculation:

.. code-block:: python

   kv_cache_memory = 2 × num_layers × hidden_dim × context_length × bytes_per_element

Components:

* **Keys**: Cache for attention keys (projections from query tokens)
* **Values**: Cache for attention values (content to attend to)
* **Context length**: Linear scaling - 2x context = 2x KV cache memory

Precision impact:

* **FP32 KV**: 4 bytes per element (rarely used)
* **FP16/BF16 KV**: 2 bytes per element (standard)
* **INT8 KV**: 1 byte per element (experimental, limited support)
* **FP8 KV**: 1 byte per element (newer GPUs only, H100+)

Context Scaling
---------------

KV cache memory scales **linearly** with context length, making long-context scenarios
extremely memory-intensive.

Scaling examples (7B model, FP16):

+------------+--------------+------------------+
| Context    | KV Cache     | Total VRAM       |
+============+==============+==================+
| 4K tokens  | ~2.1 GB      | ~18 GB           |
+------------+--------------+------------------+
| 8K tokens  | ~4.2 GB      | ~20 GB           |
+------------+--------------+------------------+
| 16K tokens | ~8.4 GB      | ~24 GB           |
+------------+--------------+------------------+
| 32K tokens | ~16.8 GB     | ~33 GB           |
+------------+--------------+------------------+
| 128K tokens| ~67.2 GB     | ~83 GB           |
+------------+--------------+------------------+

Practical implications:

* **4K → 8K**: +2.1 GB (manageable)
* **8K → 32K**: +12.6 GB (may exceed consumer GPU limits)
* **Long-context models**: Typically require multi-GPU or quantization

The tool displays projected KV cache for extended contexts (16K, 32K, 128K) to help
plan for scenarios requiring larger context windows.

Additional Terms
----------------

**Batch Size**

Number of sequences processed simultaneously. Memory scales linearly with batch size.
This tool assumes batch_size=1. Production serving with batch_size>8 requires 2-10x
more VRAM.

**Tensor Parallelism**

Splitting model weights across multiple GPUs. Each GPU stores a portion of each layer.
Enables running models too large for a single GPU, but requires high-speed NVLink
interconnects.

**Pipeline Parallelism**

Splitting model layers across multiple GPUs. Each GPU stores complete consecutive layers.
Lower communication overhead than tensor parallelism but introduces pipeline bubbles.

**Flash Attention**

An attention algorithm that reduces memory usage by materializing the attention matrix
in blocks rather than all at once. Provides 2-4x reduction in activation memory
during training, but limited impact on inference VRAM (which is dominated by weights
and KV cache).

**Quantization (AWQ, GPTQ, NF4)**

Alternative quantization methods with different trade-offs:

* **AWQ (Activation-aware Weight Quantization)**: Optimizes for activation distribution,
  better INT4 performance
* **GPTQ**: Post-training quantization, good compression but may require calibration
* **NF4 (4-bit NormalFloat)**: Quantization designed for normally-distributed weights,
  used in QLoRA

**ACT (Activation Checkpointing)**

Trading compute for memory by recomputing activations during backward pass. Reduces
memory during training but has minimal impact on inference VRAM.

**Mixture of Experts (MoE)**

Models where only a subset of parameters (experts) are activated per token. MoE models
like Mixtral 8x7B have 47B total parameters but only use ~13B per forward pass,
reducing active memory requirements.

**RoPE Scaling (Rotary Position Embedding)**

Techniques to extend context length beyond training limits:
* **Linear interpolation**: Smoothly extrapolates positions
* **YaRN (Yet another RoPE extensioN)**: NTK-aware interpolation
* **Dynamic NTK**: Adjusts interpolation based on position

These affect model quality but not VRAM calculation directly.
