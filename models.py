"""
Database of LLM models available in the market.

Each model contains metadata for VRAM usage calculation during inference.

Base de dados de modelos LLM disponíveis no mercado.
Cada modelo possui metadados para cálculo de uso de VRAM em inferência.
"""

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class LLMModel:
    """Represents an LLM model with metadata for inference.

    Represents an LLM model with metadata for inference.

    Attributes:
        name: Model name
        params_billion: Number of parameters in billions
        architecture: Model architecture (e.g., "decoder-only")
        precision_default: Default precision (e.g., "fp16")
        kv_cache_mb_per_token: KV cache in MB per token (conservative FP16 estimate)
    """

    name: str
    params_billion: int
    architecture: str
    precision_default: str
    # KV cache em MB por token (estimativa conservadora para FP16)
    # KV cache in MB per token (conservative FP16 estimate)
    kv_cache_mb_per_token: float

    @property
    def size_label(self) -> str:
        """Returns simplified size label (e.g., '7B', '13B').

        Retorna label simplificado do tamanho (ex: '7B', '13B').
        """
        return f"{self.params_billion}B"


# Hardcoded model database with conservative values
# Base de modelos hardcoded com valores conservadores
# KV cache per token is estimated for decoder-only in FP16
# KV cache por token é estimado para decoder-only em FP16
# Approximate formula: kv_cache_mb_per_token ≈ (2 * layers * hidden_size * 2 bytes) / (1024 * 1024)
# Fórmula aproximada: kv_cache_mb_per_token ≈ (2 * layers * hidden_size * 2 bytes) / (1024 * 1024)
# Values below are approximate and conservative
# Valores abaixo são aproximados e conservadores

LLM_MODELS: List[LLMModel] = [
    # ~7B models (LLaMA, Mistral, Qwen, etc.)
    # Modelos ~7B
    LLMModel(
        name="LLaMA 2 / Mistral / Qwen 7B",
        params_billion=7,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=0.6,
    ),
    # ~8B models (LLaMA 3, etc.)
    # Modelos ~8B
    LLMModel(
        name="LLaMA 3 8B",
        params_billion=8,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=0.7,
    ),
    # ~13B models
    # Modelos ~13B
    LLMModel(
        name="LLaMA 2 13B",
        params_billion=13,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=0.9,
    ),
    # ~14B models
    # Modelos ~14B
    LLMModel(
        name="Qwen 14B / CodeQwen 14B",
        params_billion=14,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=1.0,
    ),
    # ~30-34B models
    # Modelos ~30-34B
    LLMModel(
        name="LLaMA 3.1 34B / Yi 34B",
        params_billion=34,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=1.8,
    ),
    # ~65-70B models
    # Modelos ~65-70B
    # KV cache adjusted for production use (includes padding, alignment, buffers)
    LLMModel(
        name="LLaMA 2 70B / LLaMA 3.1 70B",
        params_billion=70,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=4.27,  # ~35 GB for 8k context in production
    ),
    # ~100B+ models
    # Modelos ~100B+
    LLMModel(
        name="Falcon 180B",
        params_billion=180,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=8.0,
    ),
]


def get_model_by_size(size_billion: int) -> LLMModel | None:
    """Returns a model by its size in billions of parameters.

    Retorna um modelo pelo tamanho em bilhões de parâmetros.

    Args:
        size_billion: Model size in billions of parameters

    Returns:
        LLMModel if found, None otherwise
    """
    for model in LLM_MODELS:
        if model.params_billion == size_billion:
            return model
    return None


def get_all_models() -> List[LLMModel]:
    """Returns all available models.

    Retorna todos os modelos disponíveis.

    Returns:
        List of all LLMModel instances
    """
    return LLM_MODELS.copy()
