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

  Quick Start
  python main.py --context 4096 --only-runs

  Use Cases
  - Plan which GPU to buy for your target model
  - Determine if a model fits on your current hardware
  - Compare VRAM requirements across quantization levels
  - Estimate capacity for Google Colab environments

  Supported Models: 0.5B to 180B parameter models including LLaMA, Mistral, Qwen, Phi, and more.

  Requirements: Python 3.10+ (no external dependencies)

  ---
