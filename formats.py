"""
Model format definitions and memory characteristics for LLM inference.

Defines different model file formats (GGUF, EXL2, GPTQ, AWQ) and their
specific memory overhead characteristics.

Definições de formatos de modelo e características de memória para inferência de LLMs.
Define diferentes formatos de arquivo de modelo (GGUF, EXL2, GPTQ, AWQ)
e suas características específicas de overhead de memória.
"""

import re
from dataclasses import dataclass
from enum import Enum


class ModelFormat(Enum):
    """Supported model file formats for inference.

    Formatos de arquivo de modelo suportados para inferência.

    Each format has different memory characteristics:
    - FP16: Baseline format, no special overhead
    - GGUF: llama.cpp format with metadata structure
    - EXL2: ExLlama2 format with optimized memory layout
    - GPTQ: GPTQ quantization format
    - AWQ: Activation-aware quantization format
    """

    FP16 = "fp16"
    GGUF = "gguf"
    EXL2 = "exl2"
    GPTQ = "gptq"
    AWQ = "awq"


class Quantization(Enum):
    """Quantization levels for model weights.

    Níveis de quantização para pesos do modelo.
    """

    FP32 = "fp32"  # 4 bytes per parameter
    FP16 = "fp16"  # 2 bytes per parameter
    INT8 = "int8"  # 1 byte per parameter
    INT4 = "int4"  # 0.5 byte per parameter (packed)

    @property
    def bytes_per_param(self) -> float:
        """Returns bytes per parameter for this precision.

        Retorna bytes por parâmetro para esta precisão.
        """
        return BYTES_PER_PARAM[self]


# Bytes per parameter for each quantization level
# Bytes por parâmetro para cada nível de quantização
BYTES_PER_PARAM: dict[Quantization, float] = {
    Quantization.FP32: 4.0,  # 32 bits = 4 bytes
    Quantization.FP16: 2.0,  # 16 bits = 2 bytes
    Quantization.INT8: 1.0,  # 8 bits = 1 byte
    Quantization.INT4: 0.5,  # 4 bits = 0.5 byte (packed)
}


# Format-specific memory overhead multipliers
# Multiplicadores de overhead de memória específicos por formato
# These represent additional memory beyond the base parameter memory
# Estes representam memória adicional além da memória base de parâmetros
FORMAT_OVERHEAD: dict[ModelFormat, float] = {
    ModelFormat.FP16: 1.0,  # Baseline - no additional overhead
    ModelFormat.GGUF: 1.15,  # +15% for metadata structure, tensor indexing
    ModelFormat.EXL2: 1.05,  # +5% optimized layout, minimal overhead
    ModelFormat.GPTQ: 1.10,  # +10% quantization metadata, calibration data
    ModelFormat.AWQ: 1.08,  # +8% activation-aware quantization overhead
}


# GGUF quantization pattern mapping
# Mapeamento de padrões de quantização GGUF
# Maps GGUF quantization names to effective quantization levels
# Mapeia nomes de quantização GGUF para níveis efetivos de quantização
GGUF_QUANT_PATTERNS: dict[str, Quantization] = {
    # 2-bit quantizations (effective ~3-4 bits)
    "Q2_K": Quantization.INT4,
    "Q2_K_S": Quantization.INT4,
    "Q2_K_M": Quantization.INT4,
    "Q2_K_L": Quantization.INT4,
    # 3-bit quantizations (effective ~3-4 bits)
    "Q3_K": Quantization.INT4,
    "Q3_K_S": Quantization.INT4,
    "Q3_K_M": Quantization.INT4,
    "Q3_K_L": Quantization.INT4,
    "Q3_K_XS": Quantization.INT4,
    # 4-bit quantizations (effective ~4-5 bits)
    "Q4_K": Quantization.INT4,
    "Q4_K_S": Quantization.INT4,
    "Q4_K_M": Quantization.INT4,
    "Q4_0": Quantization.INT4,
    "Q4_1": Quantization.INT4,
    # 5-bit quantizations (effective ~5 bits, closer to INT8)
    "Q5_K": Quantization.INT8,
    "Q5_K_S": Quantization.INT8,
    "Q5_K_M": Quantization.INT8,
    "Q5_0": Quantization.INT8,
    "Q5_1": Quantization.INT8,
    # 6-bit quantizations (effective ~6 bits)
    "Q6_K": Quantization.INT8,
    # 8-bit quantization
    "Q8_0": Quantization.INT8,
    # Floating point
    "F16": Quantization.FP16,
    "F32": Quantization.FP32,
}


# Effective bits per parameter for GGUF quantization patterns
# Used for more accurate memory calculations
# Bits efetivos por parâmetro para padrões de quantização GGUF
GGUF_BITS_PER_PARAM: dict[str, float] = {
    "Q2_K": 3.5,  # ~3.5 bits effective
    "Q2_K_S": 3.0,
    "Q2_K_M": 3.5,
    "Q2_K_L": 3.75,
    "Q3_K": 4.5,  # ~4.5 bits effective
    "Q3_K_S": 3.5,
    "Q3_K_M": 4.5,
    "Q3_K_L": 5.0,
    "Q3_K_XS": 3.0,
    "Q4_K": 5.5,  # ~5.5 bits effective
    "Q4_K_S": 4.5,
    "Q4_K_M": 5.5,
    "Q4_0": 4.5,
    "Q4_1": 5.0,
    "Q5_K": 6.5,  # ~6.5 bits effective
    "Q5_K_S": 5.5,
    "Q5_K_M": 6.5,
    "Q5_0": 5.5,
    "Q5_1": 6.0,
    "Q6_K": 7.5,  # ~7.5 bits effective
    "Q8_0": 8.0,  # 8 bits
    "F16": 16.0,
    "F32": 32.0,
}


@dataclass
class GGUFInfo:
    """Information extracted from a GGUF filename.

    Informações extraídas de um nome de arquivo GGUF.

    Attributes:
        quant_name: Name of the quantization (e.g., "Q4_K_M")
        quant_type: Effective quantization type as string
        bits_per_param: Effective bits per parameter
        is_gguf: True if the file appears to be a GGUF format
    """

    quant_name: str | None
    quant_type: str  # "fp32", "fp16", "int8", or "int4"
    bits_per_param: float
    is_gguf: bool


def detect_gguf_quantization(filename: str) -> GGUFInfo:
    """Detect GGUF quantization from filename.

    Detecta quantização GGUF a partir do nome do arquivo.

    Args:
        filename: Model filename to analyze

    Returns:
        GGUFInfo with detected quantization details

    Examples:
        >>> detect_gguf_quantization("llama-2-7b.Q4_K_M.gguf")
        GGUFInfo(quant_name="Q4_K_M", quant_type="int4", bits_per_param=5.5, is_gguf=True)

        >>> detect_gguf_quantization("model.bin")
        GGUFInfo(quant_name=None, quant_type="fp16", bits_per_param=16.0, is_gguf=False)
    """
    filename_upper = filename.upper()

    # Check if .gguf extension
    is_gguf = filename_upper.endswith(".GGUF")

    # Try to match GGUF quantization patterns
    # Pattern matches like Q4_K_M, Q2_K, Q8_0, etc.
    pattern = r"(Q[234568]_K(?:_[SMLX])?|Q[234568]_[01]|Q[234568]_K|Q8_0|F16|F32)"

    match = re.search(pattern, filename_upper)

    if match:
        quant_name = match.group(1)
        quantization = GGUF_QUANT_PATTERNS.get(quant_name, Quantization.INT4)
        bits_per_param = GGUF_BITS_PER_PARAM.get(quant_name, 4.5)

        return GGUFInfo(
            quant_name=quant_name, quant_type=quantization.value, bits_per_param=bits_per_param, is_gguf=True
        )

    # No quantization detected - assume FP16
    return GGUFInfo(quant_name=None, quant_type=Quantization.FP16.value, bits_per_param=16.0, is_gguf=is_gguf)


def get_format_from_filename(filename: str) -> ModelFormat:
    """Detect model format from filename extension.

    Detecta formato do modelo a partir da extensão do arquivo.

    Args:
        filename: Model filename

    Returns:
        ModelFormat enum value

    Examples:
        >>> get_format_from_filename("model.Q4_K_M.gguf")
        ModelFormat.GGUF

        >>> get_format_from_filename("model.exl2")
        ModelFormat.EXL2
    """
    filename_lower = filename.lower()

    if filename_lower.endswith(".gguf"):
        return ModelFormat.GGUF
    elif filename_lower.endswith(".exl2"):
        return ModelFormat.EXL2
    elif ".gptq" in filename_lower or filename_lower.endswith(".pt"):
        return ModelFormat.GPTQ
    elif ".awq" in filename_lower or filename_lower.endswith(".awq"):
        return ModelFormat.AWQ
    else:
        # Default to FP16
        return ModelFormat.FP16


def calculate_format_multiplier(format: ModelFormat) -> float:
    """Get the memory overhead multiplier for a format.

    Obtém o multiplicador de overhead de memória para um formato.

    Args:
        format: ModelFormat enum value

    Returns:
        Multiplier to apply to base parameter memory
    """
    return FORMAT_OVERHEAD.get(format, 1.0)


def get_bytes_per_param_from_bits(bits: float) -> float:
    """Convert bits per parameter to bytes per parameter.

    Converte bits por parâmetro para bytes por parâmetro.

    Args:
        bits: Bits per parameter

    Returns:
        Bytes per parameter
    """
    return bits / 8.0
