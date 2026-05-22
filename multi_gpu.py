"""
Multi-GPU support for LLM inference.

Implements calculations for tensor parallelism and pipeline parallelism
across multiple GPUs, including heterogeneous configurations.

Suporte multi-GPU para inferência de LLMs.

Implementa cálculos para paralelismo tensor e pipeline paralelo
em múltiplas GPUs, incluindo configurações heterogêneas.
"""

from dataclasses import dataclass
from enum import Enum

from calculator import BYTES_PER_PARAM, OVERHEAD_FACTOR, Quantization
from formats import FORMAT_OVERHEAD, ModelFormat
from gpus import GPU
from models import LLMModel


class MultiGPUMode(Enum):
    """Multi-GPU parallelism mode.

    Modo de paralelismo multi-GPU.

    - TENSOR_PARALLEL: Model weights split across GPUs (same layers, different shards)
    - PIPELINE_PARALLEL: Layers distributed across GPUs (different layers on each GPU)
    """

    TENSOR_PARALLEL = "tensor_parallel"
    PIPELINE_PARALLEL = "pipeline_parallel"


@dataclass
class MultiGPUConfig:
    """Configuration for multi-GPU inference.

    Configuração para inferência multi-GPU.

    Attributes:
        gpus: List of GPUs to use (can be heterogeneous)
        mode: Parallelism mode (tensor or pipeline)
        communication_overhead: Additional VRAM % for inter-GPU communication
    """

    gpus: list[GPU]
    mode: MultiGPUMode = MultiGPUMode.TENSOR_PARALLEL
    communication_overhead: float = 0.08  # 8% overhead for communication buffers

    @property
    def total_vram(self) -> float:
        """Total VRAM across all GPUs in GB."""
        return sum(gpu.vram_gb for gpu in self.gpus)

    @property
    def min_vram(self) -> float:
        """Minimum VRAM among all GPUs in GB."""
        return min(gpu.vram_gb for gpu in self.gpus) if self.gpus else 0

    @property
    def max_vram(self) -> float:
        """Maximum VRAM among all GPUs in GB."""
        return max(gpu.vram_gb for gpu in self.gpus) if self.gpus else 0

    @property
    def gpu_count(self) -> int:
        """Number of GPUs in the configuration."""
        return len(self.gpus)

    @property
    def is_homogeneous(self) -> bool:
        """True if all GPUs have the same VRAM capacity."""
        if not self.gpus:
            return True
        vram_set = set(gpu.vram_gb for gpu in self.gpus)
        return len(vram_set) == 1


@dataclass
class MultiGPUAllocation:
    """Memory allocation for a single GPU in multi-GPU setup.

    Alocação de memória para uma única GPU em configuração multi-GPU.

    Attributes:
        gpu: The GPU being allocated
        vram_used_gb: VRAM used on this GPU
        layer_count: Number of layers assigned to this GPU (pipeline mode)
        shard_ratio: Ratio of model weights on this GPU (tensor mode)
        is_bottleneck: Whether this GPU is the performance bottleneck
    """

    gpu: GPU
    vram_used_gb: float
    layer_count: int = 0
    shard_ratio: float = 0.0
    is_bottleneck: bool = False


@dataclass
class MultiGPUResult:
    """Result of multi-GPU inference calculation.

    Resultado de cálculo de inferência multi-GPU.

    Attributes:
        total_vram_required: Total VRAM required across all GPUs
        per_gpu_allocation: Allocation per GPU (gpu_name -> allocation)
        layers_per_gpu: Layers assigned to each GPU (pipeline mode)
        shard_per_gpu: Shard ratio per GPU (tensor mode)
        status: Feasibility status
        bottleneck_gpu: Name of the bottleneck GPU (if any)
        communication_overhead_gb: Additional VRAM for communication
        effective_utilization: Ratio of total VRAM actually utilized
        recommended_framework_config: Framework-specific configuration hints
    """

    total_vram_required: float
    per_gpu_allocation: dict[str, MultiGPUAllocation]
    layers_per_gpu: dict[str, int]
    shard_per_gpu: dict[str, float]
    status: str  # "runs", "doesnt_run", "needs_more_gpus"
    bottleneck_gpu: str | None
    communication_overhead_gb: float
    effective_utilization: float
    recommended_framework_config: dict[str, str]


class MultiGPUCalculator:
    """Calculator for multi-GPU LLM inference configurations.

    Calculadora para configurações de inferência multi-GPU de LLMs.

    Supports both tensor parallelism (splitting weights across GPUs)
    and pipeline parallelism (distributing layers across GPUs),
    including heterogeneous GPU configurations.

    Suporta tanto paralelismo tensor (dividindo pesos entre GPUs)
    quanto paralelismo pipeline (distribuindo camadas entre GPUs),
    incluindo configurações heterogêneas de GPU.
    """

    def __init__(
        self,
        quantization: Quantization = Quantization.FP16,
        model_format: ModelFormat = ModelFormat.FP16,
        communication_overhead: float = 0.08,
    ):
        """Initialize the multi-GPU calculator.

        Inicializa a calculadora multi-GPU.

        Args:
            quantization: Quantization type for memory calculation
            model_format: Model format for overhead calculation
            communication_overhead: Communication overhead ratio (0.08 = 8%)
        """
        self.quantization = quantization
        self.model_format = model_format
        self.communication_overhead = communication_overhead

    def calculate_base_model_memory(self, model: LLMModel) -> float:
        """Calculate base model memory requirement.

        Calcula requisito base de memória do modelo.

        Args:
            model: LLM model

        Returns:
            Base memory in GB
        """
        bytes_per_param = BYTES_PER_PARAM[self.quantization]
        format_multiplier = FORMAT_OVERHEAD.get(self.model_format, 1.0)

        params_memory = model.params_billion * bytes_per_param * format_multiplier
        overhead = params_memory * OVERHEAD_FACTOR

        return params_memory + overhead

    def calculate_tensor_parallel(
        self,
        model: LLMModel,
        config: MultiGPUConfig,
        context_tokens: int,
    ) -> MultiGPUResult:
        """Calculate tensor parallelism configuration.

        Calcula configuração de paralelismo tensor.

        In tensor parallel mode, model weights are split across GPUs.
        Each GPU stores a portion of each layer's weights.
        VRAM per GPU = (model_memory / num_gpus) + communication_overhead

        No modo tensor paralelo, os pesos do modelo são divididos entre GPUs.
        Cada GPU armazena uma porção dos pesos de cada camada.
        VRAM por GPU = (memória_modelo / num_gpus) + overhead_comunicação

        Args:
            model: LLM model to analyze
            config: Multi-GPU configuration
            context_tokens: Context size in tokens

        Returns:
            MultiGPUResult with allocation details
        """
        base_memory = self.calculate_base_model_memory(model)

        # KV cache (needed on each GPU in tensor parallel mode)
        kv_cache_gb = (model.kv_cache_mb_per_token * context_tokens) / 1024

        # Communication overhead for inter-GPU synchronization
        comm_overhead_gb = base_memory * config.communication_overhead

        # Memory per GPU (even distribution for tensor parallel)
        # In heterogeneous setups, we're limited by the smallest GPU
        per_gpu_base = base_memory / config.gpu_count

        # Check if we can do balanced distribution on heterogeneous GPUs
        if not config.is_homogeneous:
            # For heterogeneous: allocate based on VRAM capacity
            total_vram = config.total_vram
            per_gpu_allocation: dict[str, MultiGPUAllocation] = {}
            shard_per_gpu: dict[str, float] = {}

            for gpu in config.gpus:
                # Allocate proportionally to VRAM capacity
                gpu_ratio = gpu.vram_gb / total_vram
                gpu_memory = base_memory * gpu_ratio + kv_cache_gb + comm_overhead_gb * gpu_ratio

                shard_per_gpu[gpu.name] = gpu_ratio

                per_gpu_allocation[gpu.name] = MultiGPUAllocation(
                    gpu=gpu,
                    vram_used_gb=gpu_memory,
                    shard_ratio=gpu_ratio,
                    is_bottleneck=gpu.vram_gb == config.min_vram,
                )
        else:
            # Homogeneous: equal distribution
            per_gpu_memory = per_gpu_base + kv_cache_gb + comm_overhead_gb

            per_gpu_allocation = {}
            shard_per_gpu = {}

            for gpu in config.gpus:
                shard_per_gpu[gpu.name] = 1.0 / config.gpu_count
                per_gpu_allocation[gpu.name] = MultiGPUAllocation(
                    gpu=gpu,
                    vram_used_gb=per_gpu_memory,
                    shard_ratio=1.0 / config.gpu_count,
                    is_bottleneck=False,
                )

        # Determine status
        bottleneck = None
        all_fit = True
        for gpu in config.gpus:
            alloc = per_gpu_allocation[gpu.name]
            if alloc.vram_used_gb > gpu.vram_gb:
                all_fit = False
                if bottleneck is None:
                    bottleneck = gpu.name

        status = "runs" if all_fit else "doesnt_run"

        # Calculate effective utilization
        total_used = sum(alloc.vram_used_gb for alloc in per_gpu_allocation.values())
        effective_utilization = total_used / config.total_vram if config.total_vram > 0 else 0

        # Framework configuration hints
        recommended_framework_config = {
            "tensor_parallel_size": str(config.gpu_count),
            "mode": "tensor_parallel",
            "llama_cpp": "--split-mode layer (or use manual GPU selection)",
            "vllm": f"--tensor-parallel-size {config.gpu_count}",
        }

        return MultiGPUResult(
            total_vram_required=base_memory + kv_cache_gb * config.gpu_count + comm_overhead_gb,
            per_gpu_allocation=per_gpu_allocation,
            layers_per_gpu={},  # Not applicable for tensor parallel
            shard_per_gpu=shard_per_gpu,
            status=status,
            bottleneck_gpu=bottleneck,
            communication_overhead_gb=comm_overhead_gb,
            effective_utilization=effective_utilization,
            recommended_framework_config=recommended_framework_config,
        )

    def calculate_pipeline_parallel(
        self,
        model: LLMModel,
        config: MultiGPUConfig,
        context_tokens: int,
    ) -> MultiGPUResult:
        """Calculate pipeline parallelism configuration.

        Calcula configuração de paralelismo pipeline.

        In pipeline parallel mode, different layers are on different GPUs.
        VRAM per GPU varies based on layer allocation.
        Layers are distributed based on GPU VRAM capacity.

        No modo pipeline paralelo, diferentes camadas estão em diferentes GPUs.
        VRAM por GPU varia baseado na alocação de camadas.
        Camadas são distribuídas baseadas na capacidade de VRAM da GPU.

        Args:
            model: LLM model to analyze
            config: Multi-GPU configuration
            context_tokens: Context size in tokens

        Returns:
            MultiGPUResult with allocation details
        """
        base_memory = self.calculate_base_model_memory(model)
        num_layers = model.estimated_layers

        # Memory per layer
        bytes_per_param = BYTES_PER_PARAM[self.quantization]
        format_multiplier = FORMAT_OVERHEAD.get(self.model_format, 1.0)
        params_per_layer = model.params_billion / num_layers
        layer_memory_gb = params_per_layer * bytes_per_param * format_multiplier

        # KV cache and overhead per GPU (varies by layer allocation)
        runtime_overhead_gb = base_memory * OVERHEAD_FACTOR / config.gpu_count

        # Distribute layers based on VRAM capacity
        total_vram = config.total_vram
        per_gpu_allocation: dict[str, MultiGPUAllocation] = {}
        layers_per_gpu: dict[str, int] = {}

        remaining_layers = num_layers

        for gpu in config.gpus:
            if remaining_layers == 0:
                # No more layers to assign
                layers_per_gpu[gpu.name] = 0
                per_gpu_allocation[gpu.name] = MultiGPUAllocation(
                    gpu=gpu,
                    vram_used_gb=runtime_overhead_gb,
                    layer_count=0,
                )
                continue

            # Allocate layers proportionally to VRAM capacity
            gpu_ratio = gpu.vram_gb / total_vram
            target_layers = max(1, int(num_layers * gpu_ratio))

            # Cap at remaining layers
            allocated_layers = min(target_layers, remaining_layers)
            layers_per_gpu[gpu.name] = allocated_layers
            remaining_layers -= allocated_layers

            # Calculate VRAM for this GPU
            gpu_memory = allocated_layers * layer_memory_gb + runtime_overhead_gb

            per_gpu_allocation[gpu.name] = MultiGPUAllocation(
                gpu=gpu,
                vram_used_gb=gpu_memory,
                layer_count=allocated_layers,
                is_bottleneck=False,
            )

        # Determine status
        bottleneck = None
        all_fit = True
        for gpu in config.gpus:
            alloc = per_gpu_allocation[gpu.name]
            if alloc.vram_used_gb > gpu.vram_gb:
                all_fit = False
                if bottleneck is None:
                    bottleneck = gpu.name

        status = "runs" if all_fit else "doesnt_run"

        # Calculate effective utilization
        total_used = sum(alloc.vram_used_gb for alloc in per_gpu_allocation.values())
        effective_utilization = total_used / config.total_vram if config.total_vram > 0 else 0

        # Framework configuration hints
        recommended_framework_config = {
            "num_pipeline_stages": str(config.gpu_count),
            "mode": "pipeline_parallel",
            "llama_cpp": "--split-mode layer (layers distributed by VRAM)",
            "vllm": f"--pipeline-parallel-size {config.gpu_count}",
        }

        return MultiGPUResult(
            total_vram_required=base_memory,
            per_gpu_allocation=per_gpu_allocation,
            layers_per_gpu=layers_per_gpu,
            shard_per_gpu={},  # Not applicable for pipeline parallel
            status=status,
            bottleneck_gpu=bottleneck,
            communication_overhead_gb=0.0,  # Less communication overhead in pipeline
            effective_utilization=effective_utilization,
            recommended_framework_config=recommended_framework_config,
        )

    def calculate(
        self,
        model: LLMModel,
        config: MultiGPUConfig,
        context_tokens: int = 4096,
    ) -> MultiGPUResult:
        """Calculate multi-GPU configuration based on mode.

        Calcula configuração multi-GPU baseado no modo.

        Args:
            model: LLM model to analyze
            config: Multi-GPU configuration
            context_tokens: Context size in tokens

        Returns:
            MultiGPUResult with configuration details
        """
        if config.mode == MultiGPUMode.TENSOR_PARALLEL:
            return self.calculate_tensor_parallel(model, config, context_tokens)
        else:
            return self.calculate_pipeline_parallel(model, config, context_tokens)


def parse_gpu_config_string(config_str: str, gpu_database: list[GPU]) -> list[GPU]:
    """Parse a GPU configuration string into a list of GPUs.

    Analisa uma string de configuração de GPU em uma lista de GPUs.

    Format: "2x4090,1x3090" or "3xRTX 3090" or "4090,4090,3090"

    Formato: "2x4090,1x3090" ou "3xRTX 3090" ou "4090,4090,3090"

    Args:
        config_str: Configuration string to parse
        gpu_database: Available GPUs to search

    Returns:
        List of GPU instances

    Examples:
        >>> parse_gpu_config_string("2x4090,1x3090", gpu_database)
        [GPU(name="RTX 4090", ...), GPU(name="RTX 4090", ...), GPU(name="RTX 3090", ...)]

        >>> parse_gpu_config_string("3x RTX 3090", gpu_database)
        [GPU(name="RTX 3090", ...), GPU(name="RTX 3090", ...), GPU(name="RTX 3090", ...)]
    """
    import re

    gpus = []
    parts = config_str.split(",")

    # Create a mapping from lowercase names to GPU objects
    gpu_map = {gpu.name.lower(): gpu for gpu in gpu_database}

    for part in parts:
        part = part.strip()

        # Match "Nx GPU_NAME" or just "GPU_NAME"
        match = re.match(r"(\d+)x\s*(.+)", part, re.IGNORECASE)

        if match:
            count = int(match.group(1))
            gpu_name = match.group(2).strip()
        else:
            count = 1
            gpu_name = part

        # Find matching GPU
        gpu_name_lower = gpu_name.lower()
        matched_gpu = None

        # Exact match
        if gpu_name_lower in gpu_map:
            matched_gpu = gpu_map[gpu_name_lower]
        else:
            # Partial match
            for name, gpu in gpu_map.items():
                if gpu_name_lower in name:
                    matched_gpu = gpu
                    break

        if matched_gpu:
            for _ in range(count):
                gpus.append(matched_gpu)
        else:
            raise ValueError(f"GPU '{gpu_name}' not found in database")

    return gpus


def create_multi_gpu_config(
    gpu_names: list[str],
    mode: MultiGPUMode = MultiGPUMode.TENSOR_PARALLEL,
    gpu_database: list[GPU] = None,
) -> MultiGPUConfig:
    """Create a MultiGPUConfig from a list of GPU names.

    Cria uma MultiGPUConfig a partir de uma lista de nomes de GPU.

    Args:
        gpu_names: List of GPU names (e.g., ["RTX 4090", "RTX 4090", "RTX 3090"])
        mode: Parallelism mode
        gpu_database: Available GPUs (uses all GPUs if None)

    Returns:
        MultiGPUConfig instance

    Raises:
        ValueError: If a GPU name is not found
    """
    if gpu_database is None:
        from gpus import get_all_gpus

        gpu_database = get_all_gpus()

    gpu_map = {gpu.name.lower(): gpu for gpu in gpu_database}
    gpus = []

    for name in gpu_names:
        name_lower = name.lower()
        if name_lower in gpu_map:
            gpus.append(gpu_map[name_lower])
        else:
            # Try partial match
            matched = None
            for gpu_name, gpu in gpu_map.items():
                if name_lower in gpu_name:
                    matched = gpu
                    break
            if matched:
                gpus.append(matched)
            else:
                raise ValueError(f"GPU '{name}' not found in database")

    return MultiGPUConfig(gpus=gpus, mode=mode)
