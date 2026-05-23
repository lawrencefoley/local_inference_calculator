# Local Inference Calculator

Capacity planning tool for local LLM inference.

Quickly discover which language models run on your GPU for a given context size.

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management and packaging.

```bash
git clone https://github.com/lawrencefoley/local_inference_calculator.git
cd local_inference_calculator
uv sync
```

Run the CLI from a checkout:

```bash
uv run llmfit --help
```

Install/run it as a uv tool from the local checkout:

```bash
uvx --from . llmfit --help
```

After publishing or installing from a Git URL, you can also run it with `uvx` using the package source.

## Usage

### List Available Models

```bash
uv run llmfit --list-models
```

### Check Specific Model VRAM Requirements

```bash
uv run llmfit --model 7 --context 8192
uv run llmfit -m 70 -c 16384 -q int4
uv run llmfit -m 0.6 -c 8192      # Small models (0.6B, 1B, etc.)
uv run llmfit -m 70 -c 8192 -q int4 --mode production
```

This shows:
- Detailed VRAM breakdown (parameters, overhead, KV cache)
- **Real-world usage estimates** (idle vs peak VRAM)
- Calculation assumptions (batch_size, no LoRA, etc.)
- Warnings for GPUs running at the limit
- Minimum and recommended GPU VRAM
- List of compatible GPUs with free VRAM percentage

### Find Max Context for a VRAM Budget

Use `--vram` with a quantization to see which models fit and their estimated maximum context:

```bash
uv run llmfit --vram 24 --quantization int4
uv run llmfit --vram 16 --quantization fp16 --mode conservative
```

You can combine it with `--model`, `--params-b`, or `--config` to check a single model.

### Add a Model from Hugging Face `config.json`

You can derive model metadata, including KV cache MB/token, from a Hugging Face `config.json`:

```bash
uv run llmfit --config path/to/config.json --params-b 7 --context 8192
```

The parser reads common fields such as `num_hidden_layers`, `hidden_size`, `num_attention_heads`, and `num_key_value_heads`. If the config does not include a parameter count, pass it with `--params-b`. Use `--model-name` to override the display name.

### Advanced Configuration Options

#### Layer Offload Optimization

Calculate optimal GPU layer offload for models that don't fully fit in VRAM:

```bash
uv run llmfit --model 70 --context 8192 --optimize --quantization int4
```

This shows:
- How many layers can fit on GPU vs CPU
- Recommended `--gpu-layers` parameter for llama.cpp
- Performance impact estimation
- Offload options for all available GPUs

#### CPU Offload Analysis

Calculate hybrid GPU+CPU inference configuration:

```bash
uv run llmfit --model 70 --context 8192 --cpu --ram 64 --pcie-gen 4.0
```

This shows:
- System RAM requirements
- PCIe bandwidth impact
- Estimated tokens/second
- Layer distribution between GPU and CPU

#### Multi-GPU Configuration

Calculate tensor parallelism or pipeline parallelism across multiple GPUs:

```bash
uv run llmfit --params-b 405 --context 8192 --quantization int4 --multi --gpus "2x4090,1x3090"
uv run llmfit --params-b 405 --multi --gpus "3x3090" --multi-mode pipeline
```

Supported configurations:
- Homogeneous: `3x4090` (3 identical GPUs)
- Heterogeneous: `2x4090,1x3090` (mixed GPUs)
- Modes: `tensor` (default), `pipeline`

#### GGUF Format Support

Auto-detect GGUF quantization from filename:

```bash
uv run llmfit --gguf-file "llama-2-7b.Q4_K_M.gguf" --context 4096
```

Detected quantizations: Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_0, F16, F32

#### Model Format Selection

Specify model format for accurate memory overhead:

```bash
uv run llmfit --model 7 --context 8192 --format gguf --quantization int4
uv run llmfit --model 13 --context 8192 --format exl2
```

Supported formats: `fp16`, `gguf`, `exl2`, `gptq`, `awq`

### Basic (All Combinations)

```bash
uv run llmfit --context 4096
```

### Consumer GPUs Only

```bash
uv run llmfit -c 8192 --gpu-type consumer
```

### Show Only Viable Combinations

```bash
uv run llmfit -c 4096 --only-runs
```

### Export Results

```bash
uv run llmfit -c 4096 --export-json results.json
uv run llmfit -c 4096 --export-csv results.csv
```

### Calculation Modes

The tool supports three calculation modes for different scenarios:

```bash
uv run llmfit -c 8192 --mode theoretical   # Ideal minimum
uv run llmfit -c 8192 --mode conservative  # Default (10% buffer)
uv run llmfit -c 8192 --mode production    # Real-world serving (25% buffer)
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

> **See the Glossary** for detailed explanations of technical terms like Memory Allocator, LoRA Adapters, Speculative Decoding, KV Cache, Context Scaling, and more.

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
├── calculator.py  # VRAM calculation logic + Layer/CPU offload
├── formats.py     # Model format definitions (GGUF, EXL2, etc.)
├── multi_gpu.py   # Multi-GPU support (tensor/pipeline parallelism)
├── main.py        # CLI
├── docs/          # Documentation (Sphinx, multilingual)
│   ├── en/        # English documentation
│   ├── pt_BR/     # Portuguese (Brazil) documentation
│   └── Makefile   # Build with: make html LANG=en or LANG=pt_BR
└── README.md
```

## Implemented Features

### Core Features
- [x] Support for FP32, FP16, INT8, INT4 quantization
- [x] Consumer, datacenter, and Google Colab GPU database (40+ GPUs)
- [x] Model-specific VRAM breakdown with real-world estimates
- [x] Three calculation modes (theoretical, conservative, production)
- [x] Explicit batch_size and calculation assumptions
- [x] 24GB GPU limit warnings
- [x] CSV/JSON export
- [x] Production-ready KV cache estimation with mode-dependent buffers
- [x] Sphinx-generated bilingual documentation (English/Portuguese)

### Advanced Features (New in v0.2.0)
- [x] **Layer Offload Calculator**: Optimal GPU layer distribution
- [x] **CPU Offload Calculator**: Hybrid GPU+CPU inference analysis
- [x] **Multi-GPU Support**: Tensor and pipeline parallelism
- [x] **GGUF Format**: Auto-detection and memory calculations
- [x] **Model Format Support**: FP16, GGUF, EXL2, GPTQ, AWQ
- [x] **PCIe Bandwidth Analysis**: Performance impact estimation

## Roadmap

- [ ] Additional quantization formats (NF4, Marlin)
- [ ] vLLM-specific memory calculations
- [ ] Interactive web interface

## Documentation

The project includes comprehensive Sphinx documentation in **English** and **Portuguese (Brazil)**.

### Building Documentation

Install documentation dependencies:

```bash
uv pip install -r docs/requirements.txt
```

Build English documentation:

```bash
make html LANG=en
# or
cd en && make html
```

Build Portuguese documentation:

```bash
make html LANG=pt_BR
# or
cd pt_BR && make html
```

Build all languages:

```bash
make all
```

The built HTML will be in `docs/_build/{lang}/html/index.html`.

### Documentation Contents

- **Installation** - Setup instructions
- **User Guide** - Command-line usage and examples
- **Glossary** - Technical terms explained (Memory Allocator, LoRA, Speculative Decoding, KV Cache, Context Scaling, etc.)
- **API Reference** - Python API documentation
- **Examples** - Practical use cases and scripts

## Contributing

Contributions are welcome! Here are some ways you can help:

### Adding New Models or GPUs

To add a new model, edit `models.py` and add an entry to the `MODELS` list:

```python
{
    "name": "ModelName",
    "params_billion": 13,
    "architecture": "arch_name",
    "kv_cache_mb_per_token": 0.128,  # Adjust based on architecture
},
```

To add a new GPU, edit `gpus.py` and add to the `GPUS` list:

```python
{
    "name": "GPU Name",
    "vram_gb": 24,
    "type": "consumer",  # or "datacenter"
},
```

### Documentation Updates

The project maintains bilingual documentation (English and Portuguese):

- English docs: `docs/en/`
- Portuguese docs: `docs/pt_BR/`

When updating documentation, please update both language versions.

### Development Setup

```bash
# Clone the repository
git clone <your-fork-url>
cd local_inference_calculator

# Create a virtual environment (recommended)
uv sync

# Run the CLI while developing
uv run llmfit --list-models

# Build documentation to verify changes
cd docs && make html LANG=en
```

### Submitting Changes

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Style

- Follow PEP 8 for Python code
- Add docstrings to new functions and classes
- Update relevant documentation when adding features
- Keep the CLI output consistent with existing patterns

### Pull Requests

To contribute, simply open a pull request with your changes. All contributions are welcome!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
