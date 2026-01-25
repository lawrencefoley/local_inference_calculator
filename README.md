# Local Inference Calculator

Capacity planning tool for local LLM inference.

Quickly discover which language models run on your GPU for a given context size.

## Installation

```bash
cd local_inference_calculator
```

No external dependencies beyond Python standard library.

## Usage

### List Available Models

```bash
python main.py --list-models
```

### Check Specific Model VRAM Requirements

```bash
python main.py --model 7 --context 8192
python main.py -m 70 -c 16384 -q int4
python main.py -m 0.6 -c 8192      # Small models (0.6B, 1B, etc.)
python main.py -m 70 -c 8192 -q int4 --mode production
```

This shows:
- Detailed VRAM breakdown (parameters, overhead, KV cache)
- **Real-world usage estimates** (idle vs peak VRAM)
- Calculation assumptions (batch_size, no LoRA, etc.)
- Warnings for GPUs running at the limit
- Minimum and recommended GPU VRAM
- List of compatible GPUs with free VRAM percentage

### Basic (All Combinations)

```bash
python main.py --context 4096
```

### Consumer GPUs Only

```bash
python main.py -c 8192 --gpu-type consumer
```

### Show Only Viable Combinations

```bash
python main.py -c 4096 --only-runs
```

### Export Results

```bash
python main.py -c 4096 --export-json results.json
python main.py -c 4096 --export-csv results.csv
```

### Calculation Modes

The tool supports three calculation modes for different scenarios:

```bash
python main.py -c 8192 --mode theoretical   # Ideal minimum
python main.py -c 8192 --mode conservative  # Default (10% buffer)
python main.py -c 8192 --mode production    # Real-world serving (25% buffer)
```

## Professional Features

### Real-World Usage Estimates

The tool shows both idle and peak VRAM usage:

- **Idle**: Model loaded in memory, no active generation
- **Peak**: During token generation (KV cache fully allocated)

Example output for 7B FP16 with 8k context:
```
Real-World Usage Estimates:
  Idle (model loaded):    18.20 GB
  Peak (generation):      23.48 GB
```

### Calculation Assumptions

All calculations assume:
- **Memory allocator**: PyTorch-style (HF Transformers, vLLM)
  - Note: TensorRT-LLM, llama.cpp, EXL2 may have different behavior
- `batch_size = 1` (no batching)
- No LoRA adapters active
- No speculative decoding
- No tool calling overhead
- No aggressive paged KV

If any of these features are used, additional VRAM will be required:
- **Batch size > 1**: Linear scaling with batch size
- **LoRA adapters**: +0.5-2GB per adapter
- **Speculative decoding**: +30-50% memory
- **Tool calling**: Variable overhead

### Context Scaling

The tool displays projected KV cache for larger contexts:

```
Note: KV cache scales linearly with context length
  → 16k context ≈ 10.6 GB KV cache
  → 32k context ≈ 21.1 GB KV cache
```

This helps you plan for extended context scenarios (16k, 32k, 128k).

### 24GB GPU Warnings

When a model requires 22-24GB on a 24GB GPU, the tool displays:
```
⚠️  WARNING: 24GB GPUs run at the limit.
   Any batching, adapters (LoRA), or additional features may cause OOM.
```

This indicates the combination works in theory but has no safety margin for real-world usage.

## VRAM Calculation

### Calculation Modes

The tool supports three calculation modes:

| Mode | Description | KV Cache Buffer | Use Case |
|------|-------------|-----------------|----------|
| **theoretical** | Ideal minimum, no extra buffer | 0% | Best-case scenario analysis |
| **conservative** | Default mode, minimal overhead | 10% | General capacity planning |
| **production** | Real-world serving with buffers | 25% | Production deployment planning |

### How It Works

The tool estimates VRAM requirements using a model that accounts for four components:

#### 1. Model Parameters

Base memory required to store the model weights:

```
params_memory_gb = params_billion × bytes_per_param
```

Example: 70B model in FP16 (2 bytes/param): 70 × 2 = 140 GB

#### 2. Overhead

Additional memory for runtime operations, activations, and framework overhead:

```
overhead_gb = params_memory_gb × 0.30  # 30% overhead
```

#### 3. KV Cache

Memory for the attention cache during inference. This scales with context size and includes a mode-dependent buffer:

```
kv_cache_gb = (kv_cache_mb_per_token × context_tokens × multiplier × mode_buffer) / 1024
```

Where `mode_buffer` is 1.0 (theoretical), 1.1 (conservative), or 1.25 (production).

#### 4. Total VRAM

```
total_vram_gb = model_with_overhead_gb + kv_cache_gb
```

### Precision Impact

| Precision | Bytes/Param | VRAM (7B model, 8k) | VRAM (70B model, 8k) |
|-----------|-------------|---------------------|----------------------|
| FP32      | 4.0         | ~30 GB              | ~210 GB              |
| FP16      | 2.0         | ~23 GB              | ~175 GB              |
| INT8      | 1.0         | ~15 GB              | ~105 GB              |
| INT4      | 0.5         | ~10 GB              | ~72 GB               |

**Note**: INT4 values assume FP16 KV cache (realistic for most frameworks). Theoretical minimum with quantized KV cache would be lower, but this is not yet widely supported in production.

The tool displays quantization as "Quantization backend" (e.g., `FP16 (2.0 bytes/param)`) to align with how inference frameworks report precision. Future versions may support additional quantization methods (AWQ, GPTQ, NF4).

### KV Cache Multiplier

KV cache memory varies by precision since some frameworks support quantized KV cache:

| Precision | Multiplier | Notes |
|-----------|------------|-------|
| FP32      | 2.0×       | FP32 uses 2x the space of FP16 |
| FP16      | 1.0×       | Baseline - standard for most frameworks |
| INT8      | 0.85×      | INT8 weights, but KV cache often FP16 |
| INT4      | 0.85×      | INT4 weights, but KV cache usually FP16 |

**Important caveat**: In most production stacks today, **KV cache stays in FP16/BF16** even with quantized weights (INT4/INT8). KV cache quantization is experimental and only supported by specific backends (vLLM paged KV, custom kernels, EXL2).

The tool accounts for this realistic behavior:
- **Weights**: INT4 (0.5 bytes/param)
- **KV cache**: FP16 (2.0 bytes/param) → 4× larger than if quantized

This is why a 7B INT4 model still needs ~10GB for 8k context, not the theoretical minimum.

### Decision Logic

A model × GPU combination is considered viable if:

```
total_vram_required ≤ gpu_vram_gb
```

A warning is issued if free VRAM is less than 10% (low safety margin).

## Project Structure

```
local_inference_calculator/
├── models.py      # LLM model database
├── gpus.py        # GPU database
├── calculator.py  # VRAM calculation logic
├── main.py        # CLI
├── docs/          # Documentation (generated by Sphinx)
└── README.md
```

## Implemented Features

- [x] Support for FP32, FP16, INT8, INT4 quantization
- [x] Consumer, datacenter, and Google Colab GPU database
- [x] Model-specific VRAM breakdown with real-world estimates
- [x] Three calculation modes (theoretical, conservative, production)
- [x] Explicit batch_size and calculation assumptions
- [x] 24GB GPU limit warnings
- [x] CSV/JSON export
- [x] Production-ready KV cache estimation with mode-dependent buffers
- [x] Sphinx-generated documentation

## Roadmap

- [ ] Multi-GPU support
- [ ] CPU offload
- [ ] Support for specific model formats (GGUF, etc.)
