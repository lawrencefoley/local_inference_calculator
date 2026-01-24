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

__version__ = "0.1.0"

from models import LLMModel, get_all_models
from gpus import GPU, get_all_gpus
from calculator import VRAMCalculator, Quantization

__all__ = [
    "LLMModel",
    "GPU",
    "VRAMCalculator",
    "Quantization",
    "get_all_models",
    "get_all_gpus",
]
