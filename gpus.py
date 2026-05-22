"""
Database of GPUs available in the market.

Includes consumer and datacenter GPUs with their VRAM capacities.

Base de dados de GPUs disponíveis no mercado.
Inclui GPUs consumer e datacenter com suas capacidades de VRAM.
"""

from dataclasses import dataclass
from enum import Enum


class GPUType(Enum):
    """GPU type: consumer or datacenter.

    Tipo de GPU: consumer ou datacenter.
    """

    CONSUMER = "consumer"
    DATACENTER = "datacenter"


@dataclass(frozen=True)
class GPU:
    """Represents a GPU with its VRAM capacity.

    Representa uma GPU com sua capacidade de VRAM.

    Attributes:
        name: GPU model name
        vram_gb: VRAM capacity in GB
        type: GPU type (consumer or datacenter)
        memory_bandwidth_gb_s: Memory bandwidth in GB/s (optional, for future calculations)
        architecture: GPU architecture name (optional, for information)
        pcie_gen: Default PCIe generation (for CPU offload calculations)
    """

    name: str
    vram_gb: int
    type: GPUType
    # Opcional para cálculos futuros / Optional for future calculations
    memory_bandwidth_gb_s: int | None = None
    # Opcional para info / Optional for information
    architecture: str | None = None
    # Geração PCIe padrão (para cálculos de offload de CPU)
    # Default PCIe generation (for CPU offload calculations)
    pcie_gen: str = "4.0"

    @property
    def vram_label(self) -> str:
        """Returns formatted VRAM label (e.g., '24GB').

        Retorna label formatado da VRAM (ex: '24GB').
        """
        return f"{self.vram_gb}GB"


# Hardcoded GPU database
# Base de GPUs hardcoded

GPUS: list[GPU] = [
    # ============================================================
    # CONSUMER GPUs - NVIDIA GeForce RTX 30 Series
    # ============================================================
    GPU(
        name="RTX 3060",
        vram_gb=12,
        type=GPUType.CONSUMER,
        architecture="Ampere",
    ),
    GPU(
        name="RTX 3060 Ti",
        vram_gb=8,
        type=GPUType.CONSUMER,
        architecture="Ampere",
    ),
    GPU(
        name="RTX 3070",
        vram_gb=8,
        type=GPUType.CONSUMER,
        architecture="Ampere",
    ),
    GPU(
        name="RTX 3070 Ti",
        vram_gb=8,
        type=GPUType.CONSUMER,
        architecture="Ampere",
    ),
    GPU(
        name="RTX 3080",
        vram_gb=10,
        type=GPUType.CONSUMER,
        architecture="Ampere",
    ),
    GPU(
        name="RTX 3080 Ti",
        vram_gb=12,
        type=GPUType.CONSUMER,
        architecture="Ampere",
    ),
    GPU(
        name="RTX 3090",
        vram_gb=24,
        type=GPUType.CONSUMER,
        architecture="Ampere",
    ),
    GPU(
        name="RTX 3090 Ti",
        vram_gb=24,
        type=GPUType.CONSUMER,
        architecture="Ampere",
    ),
    # ============================================================
    # CONSUMER GPUs - NVIDIA GeForce RTX 40 Series
    # ============================================================
    GPU(
        name="RTX 4060",
        vram_gb=8,
        type=GPUType.CONSUMER,
        architecture="Ada Lovelace",
    ),
    GPU(
        name="RTX 4060 Ti",
        vram_gb=8,
        type=GPUType.CONSUMER,
        architecture="Ada Lovelace",
    ),
    GPU(
        name="RTX 4060 Ti (16GB)",
        vram_gb=16,
        type=GPUType.CONSUMER,
        architecture="Ada Lovelace",
    ),
    GPU(
        name="RTX 4070",
        vram_gb=12,
        type=GPUType.CONSUMER,
        architecture="Ada Lovelace",
    ),
    GPU(
        name="RTX 4070 Ti",
        vram_gb=12,
        type=GPUType.CONSUMER,
        architecture="Ada Lovelace",
    ),
    GPU(
        name="RTX 4070 Ti Super",
        vram_gb=16,
        type=GPUType.CONSUMER,
        architecture="Ada Lovelace",
    ),
    GPU(
        name="RTX 4080",
        vram_gb=16,
        type=GPUType.CONSUMER,
        architecture="Ada Lovelace",
    ),
    GPU(
        name="RTX 4080 Super",
        vram_gb=16,
        type=GPUType.CONSUMER,
        architecture="Ada Lovelace",
    ),
    GPU(
        name="RTX 4090",
        vram_gb=24,
        type=GPUType.CONSUMER,
        architecture="Ada Lovelace",
    ),
    # ============================================================
    # CONSUMER GPUs - NVIDIA GeForce RTX 50 Series
    # ============================================================
    GPU(
        name="RTX 5050",
        vram_gb=8,
        type=GPUType.CONSUMER,
        architecture="Blackwell",
    ),
    GPU(
        name="RTX 5060",
        vram_gb=8,
        type=GPUType.CONSUMER,
        architecture="Blackwell",
    ),
    GPU(
        name="RTX 5060 Ti",
        vram_gb=16,
        type=GPUType.CONSUMER,
        architecture="Blackwell",
    ),
    GPU(
        name="RTX 5070",
        vram_gb=12,
        type=GPUType.CONSUMER,
        architecture="Blackwell",
    ),
    GPU(
        name="RTX 5070 Ti",
        vram_gb=16,
        type=GPUType.CONSUMER,
        architecture="Blackwell",
    ),
    GPU(
        name="RTX 5080",
        vram_gb=16,
        type=GPUType.CONSUMER,
        architecture="Blackwell",
    ),
    GPU(
        name="RTX 5090",
        vram_gb=32,
        type=GPUType.CONSUMER,
        architecture="Blackwell",
    ),
    # ============================================================
    # CONSUMER GPUs - AMD Radeon RX 7000 Series
    # ============================================================
    GPU(
        name="RX 7600",
        vram_gb=8,
        type=GPUType.CONSUMER,
        architecture="RDNA 3",
    ),
    GPU(
        name="RX 7700 XT",
        vram_gb=16,
        type=GPUType.CONSUMER,
        architecture="RDNA 3",
    ),
    GPU(
        name="RX 7800 XT",
        vram_gb=16,
        type=GPUType.CONSUMER,
        architecture="RDNA 3",
    ),
    GPU(
        name="RX 7900 XT",
        vram_gb=20,
        type=GPUType.CONSUMER,
        architecture="RDNA 3",
    ),
    GPU(
        name="RX 7900 XTX",
        vram_gb=24,
        type=GPUType.CONSUMER,
        architecture="RDNA 3",
    ),
    # ============================================================
    # CONSUMER GPUs - AMD Radeon RX 6000 Series (relevant legacy)
    # GPUs Consumer - AMD Radeon RX 6000 Series (legado relevante)
    # ============================================================
    GPU(
        name="RX 6800",
        vram_gb=16,
        type=GPUType.CONSUMER,
        architecture="RDNA 2",
    ),
    GPU(
        name="RX 6800 XT",
        vram_gb=16,
        type=GPUType.CONSUMER,
        architecture="RDNA 2",
    ),
    GPU(
        name="RX 6900 XT",
        vram_gb=16,
        type=GPUType.CONSUMER,
        architecture="RDNA 2",
    ),
    GPU(
        name="RX 6950 XT",
        vram_gb=16,
        type=GPUType.CONSUMER,
        architecture="RDNA 2",
    ),
    # ============================================================
    # DATACENTER GPUs - NVIDIA
    # ============================================================
    GPU(
        name="A100 (40GB)",
        vram_gb=40,
        type=GPUType.DATACENTER,
        architecture="Ampere",
    ),
    GPU(
        name="A100 (80GB)",
        vram_gb=80,
        type=GPUType.DATACENTER,
        architecture="Ampere",
    ),
    GPU(
        name="A40",
        vram_gb=48,
        type=GPUType.DATACENTER,
        architecture="Ampere",
    ),
    GPU(
        name="A6000",
        vram_gb=48,
        type=GPUType.DATACENTER,
        architecture="Ampere",
    ),
    GPU(
        name="A6000 Ada",
        vram_gb=48,
        type=GPUType.DATACENTER,
        architecture="Ada Lovelace",
    ),
    GPU(
        name="H100 (80GB)",
        vram_gb=80,
        type=GPUType.DATACENTER,
        architecture="Hopper",
    ),
    GPU(
        name="H200 (141GB)",
        vram_gb=141,
        type=GPUType.DATACENTER,
        architecture="Hopper",
    ),
    GPU(
        name="L40S",
        vram_gb=48,
        type=GPUType.DATACENTER,
        architecture="Ada Lovelace",
    ),
    GPU(
        name="L4",
        vram_gb=24,
        type=GPUType.DATACENTER,
        architecture="Ada Lovelace",
    ),
    # ============================================================
    # GOOGLE COLAB GPUs
    # ============================================================
    GPU(
        name="Colab T4",
        vram_gb=16,
        type=GPUType.DATACENTER,
        architecture="Turing",
    ),
    GPU(
        name="Colab V100",
        vram_gb=16,
        type=GPUType.DATACENTER,
        architecture="Volta",
    ),
    GPU(
        name="Colab P100",
        vram_gb=16,
        type=GPUType.DATACENTER,
        architecture="Pascal",
    ),
    # ============================================================
    # DATACENTER GPUs - AMD
    # ============================================================
    GPU(
        name="MI210 (64GB)",
        vram_gb=64,
        type=GPUType.DATACENTER,
        architecture="CDNA 2",
    ),
    GPU(
        name="MI250X (128GB total, 64GB per GCD)",
        vram_gb=128,
        type=GPUType.DATACENTER,
        architecture="CDNA 2",
    ),
    GPU(
        name="MI300X (192GB)",
        vram_gb=192,
        type=GPUType.DATACENTER,
        architecture="CDNA 3",
    ),
]


def get_gpu_by_name(name: str) -> GPU | None:
    """Returns a GPU by exact name.

    Retorna uma GPU pelo nome exato.

    Args:
        name: GPU name to search for

    Returns:
        GPU if found, None otherwise
    """
    for gpu in GPUS:
        if gpu.name.lower() == name.lower():
            return gpu
    return None


def get_consumer_gpus() -> list[GPU]:
    """Returns all consumer GPUs.

    Retorna todas as GPUs consumer.

    Returns:
        List of consumer GPUs
    """
    return [gpu for gpu in GPUS if gpu.type == GPUType.CONSUMER]


def get_datacenter_gpus() -> list[GPU]:
    """Returns all datacenter GPUs.

    Retorna todas as GPUs datacenter.

    Returns:
        List of datacenter GPUs
    """
    return [gpu for gpu in GPUS if gpu.type == GPUType.DATACENTER]


def get_all_gpus() -> list[GPU]:
    """Returns all available GPUs.

    Retorna todas as GPUs disponíveis.

    Returns:
        List of all GPUs
    """
    return GPUS.copy()


def get_gpus_by_vram_min(min_vram_gb: int) -> list[GPU]:
    """Returns GPUs with at least the specified VRAM.

    Retorna GPUs com pelo menos a VRAM especificada.

    Args:
        min_vram_gb: Minimum VRAM in GB

    Returns:
        List of GPUs with VRAM >= min_vram_gb
    """
    return [gpu for gpu in GPUS if gpu.vram_gb >= min_vram_gb]
