"""
VRAM usage calculator for LLM inference.

Implements the logic for estimating the memory required to run
language models on specific GPUs.

Calculadora de uso de VRAM para inferência de LLMs.
Implementa a lógica de estimativa de memória necessária para rodar
modelos de linguagem em GPUs específicas.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List

from models import LLMModel, get_all_models
from gpus import GPU, get_all_gpus


class Quantization(Enum):
    """Supported precision/quantization types for inference.

    Tipos de precisão/quantização suportados para inferência.
    """

    FP32 = "fp32"      # 4 bytes per parameter (float32) / 4 bytes por parâmetro
    FP16 = "fp16"      # 2 bytes per parameter (float16 / half precision)
    INT8 = "int8"      # 1 byte per parameter (8-bit quantization)
    INT4 = "int4"      # 0.5 byte per parameter (4-bit quantization, packed)

    @property
    def bytes_per_param(self) -> float:
        """Returns bytes per parameter for this precision.

        Retorna bytes por parâmetro para esta precisão.
        """
        return BYTES_PER_PARAM[self]

    @property
    def kv_cache_multiplier(self) -> float:
        """KV cache multiplier based on precision.

        Multiplicador para KV cache baseado na precisão.

        Note: KV cache usually remains in FP16 even with quantized weights,
        but some frameworks support quantized KV cache with INT8/INT4.

        Nota: KV cache geralmente permanece em FP16 mesmo com pesos quantizados,
        mas com INT8/INT4 alguns frameworks suportam KV cache quantizado.
        """
        return KV_CACHE_MULTIPLIER.get(self, 1.0)


class Status(Enum):
    """Inference feasibility status.

    Status de viabilidade da inferência.
    """
    RUNS = "RUNS"      # RUNS / RODA
    NOT_RUNS = "DOESN'T RUN"  # DOESN'T RUN / NÃO RODA


class CalculationMode(Enum):
    """VRAM calculation mode.

    Modo de cálculo de VRAM.
    """
    THEORETICAL = "theoretical"   # Ideal minimum, batch=1, no padding/alignment
    CONSERVATIVE = "conservative"   # Current default, some overhead buffer
    PRODUCTION = "production"     # Real-world serving (batch>1, buffers, fragmentation)


@dataclass
class InferenceResult:
    """Feasibility analysis result for a model × GPU pair.

    Resultado de análise de viabilidade para um par modelo × GPU.

    Attributes:
        model_name: Name of the LLM model
        model_params_billion: Model size in billions of parameters
        gpu_name: Name of the GPU
        gpu_vram_gb: GPU VRAM capacity in GB
        required_vram_gb: VRAM required for inference
        status: Feasibility status (RUNS or DOESN'T RUN)
        vram_free_percent: Percentage of VRAM remaining
        quantization: Quantization type used
        warning: Optional warning message
    """

    model_name: str
    model_params_billion: int
    gpu_name: str
    gpu_vram_gb: int
    required_vram_gb: float
    status: Status
    vram_free_percent: float
    quantization: Quantization
    warning: str | None = None

    def to_dict(self) -> dict:
        """Converts to serializable dictionary.

        Converte para dicionário serializável.
        """
        return {
            "model": self.model_name,
            "model_size": f"{self.model_params_billion}B",
            "gpu": self.gpu_name,
            "gpu_vram_gb": self.gpu_vram_gb,
            "required_vram_gb": round(self.required_vram_gb, 2),
            "status": self.status.value,
            "vram_free_percent": round(self.vram_free_percent, 1),
            "quantization": self.quantization.value,
            "warning": self.warning,
        }


@dataclass
class CalculationBreakdown:
    """Detailed VRAM calculation breakdown.

    Breakdown detalhado do cálculo de VRAM.

    Attributes:
        params_memory_gb: Memory for model parameters in GB
        overhead_gb: Memory overhead in GB
        model_with_overhead_gb: Parameters + overhead in GB
        kv_cache_gb: KV cache memory in GB
        total_vram_gb: Total VRAM required in GB
    """

    params_memory_gb: float
    overhead_gb: float
    model_with_overhead_gb: float
    kv_cache_gb: float
    total_vram_gb: float

    def to_dict(self) -> dict:
        """Converts to serializable dictionary.

        Converte para dicionário serializável.
        """
        return {
            "params_memory_gb": round(self.params_memory_gb, 2),
            "overhead_gb": round(self.overhead_gb, 2),
            "model_with_overhead_gb": round(self.model_with_overhead_gb, 2),
            "kv_cache_gb": round(self.kv_cache_gb, 2),
            "total_vram_gb": round(self.total_vram_gb, 2),
        }


# ============================================================================
# CALCULATION CONSTANTS (Conservative values)
# CONSTANTES DE CÁLCULO (Valores conservadores)
# ============================================================================

# Bytes per parameter for each precision
# Bytes por parâmetro para cada precisão
BYTES_PER_PARAM = {
    Quantization.FP32: 4.0,  # 32 bits = 4 bytes
    Quantization.FP16: 2.0,  # 16 bits = 2 bytes
    Quantization.INT8: 1.0,  # 8 bits = 1 byte
    Quantization.INT4: 0.5,  # 4 bits = 0.5 byte (packed)
}

# KV cache multiplier per precision
# Multiplicador do KV cache por precisão
# IMPORTANT: In most production stacks today, KV cache stays in FP16/BF16
# even with quantized weights (INT4/INT8). KV cache quantization is experimental.
# IMPORTANTE: Na maioria dos stacks de produção hoje, KV cache permanece em FP16/BF16
# mesmo com pesos quantizados (INT4/INT8). Quantização de KV cache é experimental.
#
# Conservative values: assume FP16 for KV cache unless explicitly optimized
# Valores conservadores: assumimos FP16 para KV cache exceto quando otimizado explicitamente
KV_CACHE_MULTIPLIER = {
    Quantization.FP32: 2.0,  # FP32 uses 2x the space of FP16
    Quantization.FP16: 1.0,  # Baseline - standard for most frameworks
    Quantization.INT8: 0.85,  # INT8 weights, but KV cache often FP16 (conservative)
    Quantization.INT4: 0.85,  # INT4 weights, but KV cache usually FP16 (realistic)
}
# Note: 0.85 assumes some KV cache optimization (paged KV, compression).
# For strict real-world accuracy with INT4 weights, use 1.0 (FP16 KV cache).
# Only vLLM paged KV, custom kernels, or EXL2-like backends support quantized KV cache.

# Overhead factor (runtime, activations, etc.)
# Fator de overhead (runtime, activations, etc.)
# Conservative: framework + memory overhead during inference
# Conservador: framework + overhead de memória durante inferência
OVERHEAD_FACTOR = 0.30  # 30% overhead

# Minimum safety margin to consider viable
# Margem de segurança mínima para considerar viável
SAFETY_MARGIN_THRESHOLD = 0.10  # 10%


class VRAMCalculator:
    """VRAM calculator for LLM inference.

    Calculadora de VRAM para inferência de LLMs.
    """

    def __init__(
        self,
        quantization: Quantization = Quantization.FP16,
        overhead_factor: float = OVERHEAD_FACTOR,
        calculation_mode: CalculationMode = CalculationMode.CONSERVATIVE,
    ):
        """Initialize the calculator.

        Inicializa a calculadora.

        Args:
            quantization: Quantization type (default: FP16)
            overhead_factor: Overhead factor (0.30 = 30%)
            calculation_mode: Calculation mode for VRAM estimation
        """
        self.quantization = quantization
        self.overhead_factor = overhead_factor
        self.calculation_mode = calculation_mode

    def calculate_params_memory(self, params_billion: int) -> float:
        """Calculate base memory for model parameters.

        Calcula memória base dos parâmetros do modelo.

        Formula: params_memory_gb = params_billion * bytes_per_param
        Fórmula: params_memory_gb = params_billion × BYTES_PER_PARAM

        Note: params_billion is in billions, and 1 billion bytes = 1 GB.
        So for FP16 (2 bytes/param): 70B model = 70 × 2 = 140 GB

        Args:
            params_billion: Model size in billions of parameters

        Returns:
            Memory in GB
        """
        bytes_per_param = BYTES_PER_PARAM[self.quantization]
        # params_billion is in billions, 1 billion bytes = 1 GB
        # For FP16: 7B × 2 bytes = 14 GB, 70B × 2 bytes = 140 GB
        params_memory_gb = params_billion * bytes_per_param
        return params_memory_gb

    def calculate_overhead(self, params_memory_gb: float) -> float:
        """Calculate memory overhead (runtime, activations, etc.).

        Calcula overhead de memória (runtime, activations, etc.).

        Formula: overhead = params_memory * overhead_factor
        Fórmula: overhead = params_memory × overhead_factor

        Args:
            params_memory_gb: Base parameter memory in GB

        Returns:
            Overhead in GB
        """
        return params_memory_gb * self.overhead_factor

    def calculate_kv_cache(
        self,
        kv_cache_mb_per_token: float,
        context_tokens: int,
    ) -> float:
        """Calculate memory required for KV cache.

        Calcula memória necessária para KV cache.

        Formula: kv_cache_gb = (kv_cache_mb_per_token * context_tokens * multiplier * mode_buffer) / 1024
        Fórmula: kv_cache_gb = (kv_cache_mb_per_token × context_tokens × multiplier × mode_buffer) / 1024

        The base KV cache is defined for FP16. For other precisions, we apply
        a multiplier: FP32 uses 2x, INT8/INT4 may use less depending on the framework.

        O KV cache base é definido para FP16. Para outras precisões, aplicamos
        um multiplicador: FP32 usa 2x, INT8/INT4 podem usar menos dependendo do framework.

        The calculation mode adds a buffer for production scenarios:
        - THEORETICAL: No buffer (ideal minimum, batch=1, no padding)
        - CONSERVATIVE: 10% buffer (minimal overhead)
        - PRODUCTION: 25% buffer (batch>1, fragmentation, real-world serving)

        Args:
            kv_cache_mb_per_token: MB per token for the model (FP16 baseline)
            context_tokens: Context size in tokens

        Returns:
            KV cache in GB
        """
        multiplier = self.quantization.kv_cache_multiplier

        # Production buffer based on calculation mode
        # Buffer de produção baseado no modo de cálculo
        mode_buffer = {
            CalculationMode.THEORETICAL: 1.0,   # No extra buffer / Sem buffer extra
            CalculationMode.CONSERVATIVE: 1.1,  # 10% buffer for overhead
            CalculationMode.PRODUCTION: 1.25,   # 25% buffer for real-world serving
        }.get(self.calculation_mode, 1.0)

        kv_cache_mb = kv_cache_mb_per_token * context_tokens * multiplier * mode_buffer
        kv_cache_gb = kv_cache_mb / 1024
        return kv_cache_gb

    def calculate_total_vram(
        self,
        model: LLMModel,
        context_tokens: int,
    ) -> CalculationBreakdown:
        """Calculate total VRAM required for a model with given context.

        Calcula VRAM total necessária para um modelo com contexto dado.

        Args:
            model: LLM model to evaluate
            context_tokens: Context size in tokens

        Returns:
            CalculationBreakdown with calculation details
        """
        # 5.1 Model parameters
        # Parâmetros do modelo
        params_memory_gb = self.calculate_params_memory(model.params_billion)

        # 5.2 Overhead
        overhead_gb = self.calculate_overhead(params_memory_gb)
        model_with_overhead_gb = params_memory_gb + overhead_gb

        # 5.3 KV Cache
        kv_cache_gb = self.calculate_kv_cache(
            model.kv_cache_mb_per_token,
            context_tokens,
        )

        # 5.4 Total VRAM
        # VRAM total
        total_vram_gb = model_with_overhead_gb + kv_cache_gb

        return CalculationBreakdown(
            params_memory_gb=params_memory_gb,
            overhead_gb=overhead_gb,
            model_with_overhead_gb=model_with_overhead_gb,
            kv_cache_gb=kv_cache_gb,
            total_vram_gb=total_vram_gb,
        )

    def evaluate_pair(
        self,
        model: LLMModel,
        gpu: GPU,
        context_tokens: int,
        quantization: Quantization | None = None,
    ) -> InferenceResult:
        """Evaluate if a model × GPU pair is viable for the given context.

        Avalia se um par modelo × GPU é viável para o contexto dado.

        Args:
            model: LLM model to evaluate
            gpu: GPU to evaluate
            context_tokens: Context size in tokens
            quantization: Override quantization (uses instance default if None)

        Returns:
            InferenceResult with status and details
        """
        breakdown = self.calculate_total_vram(model, context_tokens)
        required_vram = breakdown.total_vram_gb

        # 6. Decision logic
        # Lógica de decisão
        if required_vram <= gpu.vram_gb:
            status = Status.RUNS
            vram_free_percent = ((gpu.vram_gb - required_vram) / gpu.vram_gb) * 100

            # Warning if margin < 10%
            # Aviso se margem < 10%
            warning = None
            if vram_free_percent < (SAFETY_MARGIN_THRESHOLD * 100):
                warning = f"Low safety margin ({vram_free_percent:.1f}% free) / Margem de segurança baixa"

        else:
            status = Status.NOT_RUNS
            vram_free_percent = 0.0
            warning = None

        return InferenceResult(
            model_name=model.name,
            model_params_billion=model.params_billion,
            gpu_name=gpu.name,
            gpu_vram_gb=gpu.vram_gb,
            required_vram_gb=required_vram,
            status=status,
            vram_free_percent=vram_free_percent,
            quantization=quantization or self.quantization,
            warning=warning,
        )

    def calculate_all_combinations(
        self,
        context_tokens: int,
        models: List[LLMModel] | None = None,
        gpus: List[GPU] | None = None,
    ) -> List[InferenceResult]:
        """Calculate feasibility for all model × GPU combinations.

        Calcula viabilidade para todas as combinações modelo × GPU.

        Args:
            context_tokens: Context size in tokens
            models: List of models (uses all if None)
            gpus: List of GPUs (uses all if None)

        Returns:
            List of InferenceResult for all combinations
        """
        if models is None:
            models = get_all_models()
        if gpus is None:
            gpus = get_all_gpus()

        results = []
        for model in models:
            for gpu in gpus:
                result = self.evaluate_pair(model, gpu, context_tokens)
                results.append(result)

        return results


def calculate_inference(context_tokens: int) -> dict:
    """Main calculation function (simplified interface).

    Função principal de cálculo (interface simplificada).

    Args:
        context_tokens: Context size in tokens

    Returns:
        Dictionary with structured results
    """
    calculator = VRAMCalculator()
    results = calculator.calculate_all_combinations(context_tokens)

    return {
        "context_tokens": context_tokens,
        "quantization": calculator.quantization.value,
        "results": [r.to_dict() for r in results],
    }
