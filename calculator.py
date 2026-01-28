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
from typing import List, Optional, Dict

from models import LLMModel, get_all_models
from gpus import GPU, get_all_gpus
from formats import ModelFormat, FORMAT_OVERHEAD


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


# ============================================================================
# LAYER OFFLOAD CALCULATOR
# CALCULADORA DE OFFLOAD DE CAMADAS
# ============================================================================

@dataclass
class LayerOffloadResult:
    """Result for optimal layer offload calculation.

    Resultado para cálculo de offload ótimo de camadas.

    Attributes:
        total_layers: Total number of layers in the model
        layers_on_gpu: Number of layers that fit on GPU
        layers_on_cpu: Number of layers that must remain on CPU
        gpu_vram_used: VRAM used for GPU layers
        cpu_ram_used: System RAM used for CPU layers
        offload_ratio: Ratio of layers on GPU (0.0 to 1.0)
        performance_impact: Estimated performance impact (0-100% slower)
        recommended_gpu_split: Recommended --gpu-layers parameter for llama.cpp
    """
    total_layers: int
    layers_on_gpu: int
    layers_on_cpu: int
    gpu_vram_used: float  # GB
    cpu_ram_used: float   # GB
    offload_ratio: float
    performance_impact: float  # Percentage slower
    recommended_gpu_split: str
    status: str  # "full_gpu", "partial_offload", "cpu_only"


class LayerOffloadCalculator:
    """Calculates optimal layer offload configuration for hybrid GPU+CPU inference.

    Calcula configuração ótima de offload de camadas para inferência híbrida GPU+CPU.

    This calculator helps determine how many transformer layers can fit in GPU VRAM
    for scenarios where the full model doesn't fit, enabling partial offloading
    strategies used by llama.cpp, AutoGPTQ, and other frameworks.

    Esta calculadora ajuda a determinar quantas camadas transformer cabem na VRAM
    para cenários onde o modelo completo não cabe, permitindo estratégias de offload
    parcial usadas por llama.cpp, AutoGPTQ e outros frameworks.
    """

    def __init__(
        self,
        quantization: Quantization = Quantization.FP16,
        safety_margin_gb: float = 1.0,
        model_format: ModelFormat = ModelFormat.FP16,
    ):
        """Initialize the layer offload calculator.

        Inicializa a calculadora de offload de camadas.

        Args:
            quantization: Quantization type for layer size calculation
            safety_margin_gb: Safety margin in GB to reserve
            model_format: Model format for overhead calculation
        """
        self.quantization = quantization
        self.safety_margin_gb = safety_margin_gb
        self.model_format = model_format

    def estimate_layer_size_gb(
        self,
        model: LLMModel,
    ) -> float:
        """Estimate the VRAM size of a single transformer layer.

        Estima o tamanho VRAM de uma única camada transformer.

        Args:
            model: LLM model to analyze

        Returns:
            Estimated size of one layer in GB

        Formula:
            layer_size = (model_params / num_layers) * bytes_per_param * format_overhead
        """
        num_layers = model.estimated_layers
        params_per_layer = model.params_billion / num_layers

        bytes_per_param = BYTES_PER_PARAM[self.quantization]
        format_multiplier = FORMAT_OVERHEAD.get(self.model_format, 1.0)

        # Layer size in GB (params_billion is already in "GB worth of bytes at 1 byte/param")
        layer_size_gb = params_per_layer * bytes_per_param * format_multiplier

        return layer_size_gb

    def calculate_kv_cache_memory(
        self,
        model: LLMModel,
        context_tokens: int,
    ) -> float:
        """Calculate KV cache memory requirement.

        Calcula requisito de memória de KV cache.

        Args:
            model: LLM model
            context_tokens: Context size in tokens

        Returns:
            KV cache memory in GB
        """
        kv_cache_mb = model.kv_cache_mb_per_token * context_tokens
        kv_cache_gb = kv_cache_mb / 1024
        return kv_cache_gb

    def calculate_optimal_offload(
        self,
        model: LLMModel,
        gpu: GPU,
        context_tokens: int,
    ) -> LayerOffloadResult:
        """Calculate optimal layer offload configuration.

        Calcula configuração ótima de offload de camadas.

        Args:
            model: LLM model to analyze
            gpu: GPU to use for offload
            context_tokens: Context size in tokens

        Returns:
            LayerOffloadResult with optimal configuration
        """
        total_layers = model.estimated_layers
        layer_size_gb = self.estimate_layer_size_gb(model)

        # Calculate available VRAM for layers
        # Available = Total VRAM - KV cache - overhead - safety margin
        kv_cache_gb = self.calculate_kv_cache_memory(model, context_tokens)

        # Runtime overhead (activations, buffers)
        params_memory_gb = model.params_billion * BYTES_PER_PARAM[self.quantization]
        overhead_gb = params_memory_gb * OVERHEAD_FACTOR

        # VRAM available for layers
        available_for_layers = gpu.vram_gb - kv_cache_gb - overhead_gb - self.safety_margin_gb

        # Calculate how many layers fit on GPU
        if available_for_layers <= 0:
            # Not enough VRAM even for KV cache
            layers_on_gpu = 0
        else:
            layers_on_gpu = min(total_layers, int(available_for_layers / layer_size_gb))

        layers_on_cpu = total_layers - layers_on_gpu
        offload_ratio = layers_on_gpu / total_layers if total_layers > 0 else 0

        # Calculate memory usage
        gpu_vram_used = layers_on_gpu * layer_size_gb + kv_cache_gb + overhead_gb
        cpu_ram_used = layers_on_cpu * layer_size_gb

        # Estimate performance impact
        # Full GPU: 0% impact
        # Partial offload: 2-5% slower per 10% of layers on CPU (PCIe bottleneck)
        # CPU only: ~10-20x slower (very rough estimate)
        if layers_on_gpu == total_layers:
            performance_impact = 0.0
            status = "full_gpu"
        elif layers_on_gpu == 0:
            performance_impact = 1000.0  # ~10x slower = 1000% slower
            status = "cpu_only"
        else:
            # Estimate: 3% slower per 10% of layers on CPU
            cpu_ratio = layers_on_cpu / total_layers
            performance_impact = cpu_ratio * 30.0  # 30% per full offload equivalent
            status = "partial_offload"

        # Recommended llama.cpp --gpu-layers parameter
        if layers_on_gpu == 0:
            recommended = "0 (CPU only)"
        elif layers_on_gpu == total_layers:
            recommended = f"{layers_on_gpu} (all layers, full GPU)"
        else:
            recommended = f"{layers_on_gpu} (of {total_layers} total)"

        return LayerOffloadResult(
            total_layers=total_layers,
            layers_on_gpu=layers_on_gpu,
            layers_on_cpu=layers_on_cpu,
            gpu_vram_used=gpu_vram_used,
            cpu_ram_used=cpu_ram_used,
            offload_ratio=offload_ratio,
            performance_impact=performance_impact,
            recommended_gpu_split=recommended,
            status=status,
        )


# ============================================================================
# CPU OFFLOAD CALCULATOR
# CALCULADORA DE OFFLOAD DE CPU
# ============================================================================

@dataclass
class PCIeConfig:
    """PCIe bandwidth configuration.

    Configuração de largura de banda PCIe.

    Attributes:
        generation: PCIe generation (3.0, 4.0, 5.0)
        bandwidth_gb_s: Theoretical bandwidth in GB/s (per lane x16)
        lanes: Number of lanes (typically x16 for GPUs)
    """
    generation: str
    bandwidth_gb_s: float
    lanes: int = 16

    @property
    def effective_bandwidth_gb_s(self) -> float:
        """Effective bandwidth accounting for protocol overhead.

        Largura de banda efetiva considerando overhead de protocolo.
        """
        # Real-world effective bandwidth is ~70-80% of theoretical
        return self.bandwidth_gb_s * 0.75


# PCIe bandwidth specifications (x16)
# Especificações de largura de banda PCIe (x16)
PCIE_CONFIGS: Dict[str, PCIeConfig] = {
    "3.0": PCIeConfig(generation="3.0", bandwidth_gb_s=16.0, lanes=16),   # ~16 GB/s theoretical
    "4.0": PCIeConfig(generation="4.0", bandwidth_gb_s=32.0, lanes=16),   # ~32 GB/s theoretical
    "5.0": PCIeConfig(generation="5.0", bandwidth_gb_s=64.0, lanes=16),   # ~64 GB/s theoretical
}


@dataclass
class CPUOffloadResult:
    """Result for CPU offload calculation.

    Resultado para cálculo de offload de CPU.

    Attributes:
        system_ram_required: Total system RAM required in GB
        system_ram_available: System RAM available in GB
        fits_in_ram: Whether the model fits in system RAM
        offload_config: Layer offload configuration
        pcie_generation: PCIe generation used
        estimated_token_speed: Estimated tokens/second with offload
        speed_vs_full_gpu: Speed ratio vs full GPU (0.0 to 1.0)
    """
    system_ram_required: float  # GB
    system_ram_available: float  # GB
    fits_in_ram: bool
    offload_config: LayerOffloadResult
    pcie_generation: str
    estimated_token_speed: float  # tokens/second
    speed_vs_full_gpu: float  # Ratio (0.0 to 1.0)


class CPUOffloadCalculator:
    """Calculates hybrid GPU+CPU inference configurations.

    Calcula configurações de inferência híbrida GPU+CPU.

    This calculator helps determine:
    - How much system RAM is needed for CPU offload
    - Performance impact based on PCIe bandwidth
    - Optimal layer distribution between GPU and CPU

    Esta calculadora ajuda a determinar:
    - Quanta RAM do sistema é necessária para offload de CPU
    - Impacto de performance baseado em largura de banda PCIe
    - Distribuição ótima de camadas entre GPU e CPU
    """

    def __init__(
        self,
        quantization: Quantization = Quantization.FP16,
        system_ram_gb: float = 32.0,
        pcie_generation: str = "4.0",
        model_format: ModelFormat = ModelFormat.FP16,
    ):
        """Initialize the CPU offload calculator.

        Inicializa a calculadora de offload de CPU.

        Args:
            quantization: Quantization type
            system_ram_gb: Available system RAM in GB
            pcie_generation: PCIe generation (3.0, 4.0, 5.0)
            model_format: Model format
        """
        self.quantization = quantization
        self.system_ram_gb = system_ram_gb
        self.pcie_config = PCIE_CONFIGS.get(pcie_generation, PCIE_CONFIGS["4.0"])
        self.model_format = model_format
        self.layer_calculator = LayerOffloadCalculator(
            quantization=quantization,
            model_format=model_format,
        )

    def calculate_total_model_memory(self, model: LLMModel) -> float:
        """Calculate total model memory requirement.

        Calcula requisito total de memória do modelo.

        Args:
            model: LLM model

        Returns:
            Total memory in GB
        """
        bytes_per_param = BYTES_PER_PARAM[self.quantization]
        format_multiplier = FORMAT_OVERHEAD.get(self.model_format, 1.0)

        # Base parameter memory
        params_memory = model.params_billion * bytes_per_param * format_multiplier

        # Add overhead for runtime
        overhead = params_memory * OVERHEAD_FACTOR

        return params_memory + overhead

    def estimate_offload_performance(
        self,
        offload_result: LayerOffloadResult,
        model: LLMModel,
    ) -> tuple[float, float]:
        """Estimate performance with offload configuration.

        Estima performance com configuração de offload.

        Args:
            offload_result: Layer offload configuration result
            model: LLM model

        Returns:
            Tuple of (estimated_tokens_per_second, speed_ratio_vs_full_gpu)
        """
        # Baseline: full GPU inference on RTX 3090-class hardware
        # These are rough estimates for a 7B model at FP16
        # Adjust by model size
        baseline_speed = 50.0 / (model.params_billion / 7.0)  # tokens/sec

        if offload_result.status == "full_gpu":
            return baseline_speed, 1.0
        elif offload_result.status == "cpu_only":
            # CPU inference: ~10-20x slower
            cpu_speed = baseline_speed / 15.0
            return cpu_speed, 1.0 / 15.0
        else:
            # Partial offload: speed depends on GPU layer ratio
            # and PCIe bandwidth for data transfer
            gpu_ratio = offload_result.offload_ratio

            # Base speed from GPU layers
            gpu_speed = baseline_speed * gpu_ratio

            # CPU contribution (slower)
            cpu_speed = (baseline_speed / 15.0) * (1 - gpu_ratio)

            # Combine (simplified model)
            estimated_speed = gpu_speed + cpu_speed

            # Adjust for PCIe bandwidth bottleneck
            # Faster PCIe = better performance with offload
            pcie_multiplier = self.pcie_config.effective_bandwidth_gb_s / 24.0  # Normalized to PCIe 4.0
            estimated_speed *= (0.7 + 0.3 * pcie_multiplier)

            speed_ratio = estimated_speed / baseline_speed

            return estimated_speed, speed_ratio

    def calculate_offload(
        self,
        model: LLMModel,
        gpu: GPU,
        context_tokens: int,
    ) -> CPUOffloadResult:
        """Calculate CPU offload configuration.

        Calcula configuração de offload de CPU.

        Args:
            model: LLM model
            gpu: GPU to use
            context_tokens: Context size in tokens

        Returns:
            CPUOffloadResult with configuration details
        """
        # Get layer offload configuration
        offload_config = self.layer_calculator.calculate_optimal_offload(
            model, gpu, context_tokens
        )

        # Calculate system RAM requirement
        # CPU layers + any additional runtime overhead on CPU
        total_memory = self.calculate_total_model_memory(model)
        system_ram_required = offload_config.cpu_ram_used + 2.0  # +2GB for CPU runtime overhead

        fits_in_ram = system_ram_required <= self.system_ram_gb

        # Estimate performance
        estimated_speed, speed_ratio = self.estimate_offload_performance(
            offload_config, model
        )

        return CPUOffloadResult(
            system_ram_required=system_ram_required,
            system_ram_available=self.system_ram_gb,
            fits_in_ram=fits_in_ram,
            offload_config=offload_config,
            pcie_generation=self.pcie_config.generation,
            estimated_token_speed=estimated_speed,
            speed_vs_full_gpu=speed_ratio,
        )
