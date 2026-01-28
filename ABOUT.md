 ---
  Local Inference Calculator

  A capacity planning tool for local LLM inference. Quickly discover which language models run on your GPU for a given
  context size.

  Features
  - 📊 VRAM Calculation: Conservative estimation model considering parameters, overhead, and KV cache
  - 🎯 Multiple Precisions: Support for FP32, FP16, INT8, and INT4 quantization
  - 💻 Comprehensive GPU Database: 40+ GPUs including consumer (NVIDIA/AMD), datacenter, and Google Colab
  - 📦 Export Options: Save results to JSON or CSV
  - 📚 CLI & Library: Use as a command-line tool or import as a Python library

  Advanced Features (v0.2.0)
  - 🚀 Layer Offload: Optimal GPU layer distribution for llama.cpp, AutoGPTQ
  - 🖥️ CPU Offload: Hybrid GPU+CPU inference with PCIe bandwidth analysis
  - 🔗 Multi-GPU: Tensor and pipeline parallelism across heterogeneous GPUs
  - 📁 GGUF Support: Auto-detect quantization from GGUF filenames (Q2_K through Q8_0)
  - 🎨 Format Support: Memory calculations for GGUF, EXL2, GPTQ, AWQ formats

  Quick Start
  python main.py --context 4096 --only-runs

  Advanced Examples
  # Layer offload optimization
  python main.py --model 70 --context 8192 --optimize-config --quantization int4

  # CPU offload analysis
  python main.py --model 70 --context 8192 --cpu-offload --system-ram 64 --pcie-gen 4.0

  # Multi-GPU configuration
  python main.py --params-b 405 --multi-gpu --gpu-config "2x4090,1x3090"

  # GGUF auto-detection
  python main.py --gguf-file "llama-2-7b.Q4_K_M.gguf" --context 4096

  Use Cases
  - Plan which GPU to buy for your target model
  - Determine if a model fits on your current hardware
  - Compare VRAM requirements across quantization levels
  - Estimate capacity for Google Colab environments
  - Optimize layer offload for hybrid GPU+CPU inference
  - Plan multi-GPU setups for large models

  Supported Models: 0.5B to 670B parameter models including LLaMA, Mistral, Qwen, Phi, DeepSeek, and more.

  Supported Formats: FP16, GGUF, EXL2, GPTQ, AWQ

  Requirements: Python 3.10+ (no external dependencies)

  ---
