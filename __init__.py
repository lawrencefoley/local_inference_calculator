"""
Local Inference Calculator - Capacity planning tool for LLMs.

This tool estimates the viability of running LLM inference
locally on different GPUs, considering context size,
model architecture, and VRAM capacity.

Local Inference Calculator - Ferramenta de capacity planning para LLMs.

Esta ferramenta estima a viabilidade de executar inferência de LLMs
localmente em diferentes GPUs, considerando tamanho do contexto,
arquitetura do modelo e capacidade de VRAM.
"""

__version__ = "0.2.0"

# Core imports
# Importações principais
from calculator import (
    CalculationBreakdown,
    CalculationMode,
    CPUOffloadCalculator,
    CPUOffloadResult,
    InferenceResult,
    LayerOffloadCalculator,
    LayerOffloadResult,
    Quantization,
    Status,
    VRAMCalculator,
)

# Advanced features
# Recursos avançados
from formats import GGUFInfo, ModelFormat, detect_gguf_quantization
from gpus import GPU, get_all_gpus
from models import LLMModel, get_all_models
from multi_gpu import (
    MultiGPUCalculator,
    MultiGPUConfig,
    MultiGPUMode,
    MultiGPUResult,
    create_multi_gpu_config,
    parse_gpu_config_string,
)

__all__ = [
    # Core
    "LLMModel",
    "GPU",
    "VRAMCalculator",
    "Quantization",
    "CalculationMode",
    "InferenceResult",
    "Status",
    "CalculationBreakdown",
    "get_all_models",
    "get_all_gpus",
    # Model formats
    "ModelFormat",
    "detect_gguf_quantization",
    "GGUFInfo",
    # Advanced calculators
    "LayerOffloadCalculator",
    "LayerOffloadResult",
    "CPUOffloadCalculator",
    "CPUOffloadResult",
    # Multi-GPU
    "MultiGPUConfig",
    "MultiGPUCalculator",
    "MultiGPUMode",
    "MultiGPUResult",
    "parse_gpu_config_string",
    "create_multi_gpu_config",
]
