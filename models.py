"""
Database of LLM models available in the market.

Each model contains metadata for VRAM usage calculation during inference.
Includes Ollama library models for comprehensive coverage.

Base de dados de modelos LLM disponíveis no mercado.
Cada modelo possui metadados para cálculo de uso de VRAM em inferência.
Inclui modelos da biblioteca Ollama para cobertura abrangente.
"""

import re
from dataclasses import dataclass, field

from formats import ModelFormat


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
        format: Model format (defaults to FP16 for base models)
        context_length_max: Maximum context length in tokens (None if unlimited)
        num_layers: Number of transformer layers (for layer offload calculations)
    """

    name: str
    params_billion: float
    architecture: str
    precision_default: str
    # KV cache em MB por token (estimativa conservadora para FP16)
    # KV cache in MB per token (conservative FP16 estimate)
    kv_cache_mb_per_token: float
    # Model format (optional, defaults to FP16)
    # Formato do modelo (opcional, padrão FP16)
    format: ModelFormat = field(default=ModelFormat.FP16)
    # Maximum context length in tokens (None if unlimited)
    # Comprimento máximo de contexto em tokens (None se ilimitado)
    context_length_max: int | None = None
    # Number of transformer layers (for layer offload calculations, estimated if None)
    # Número de camadas transformer (para cálculos de offload de camadas, estimado se None)
    num_layers: int | None = None

    @property
    def size_label(self) -> str:
        """Returns simplified size label (e.g., '7B', '13B').

        Retorna label simplificado do tamanho (ex: '7B', '13B').
        """
        return f"{self.params_billion:g}B"

    @property
    def estimated_layers(self) -> int:
        """Estimate number of layers based on model size if not specified.

        Estima número de camadas baseado no tamanho do modelo se não especificado.

        Uses typical layer counts for decoder-only models:
        - 7B models: ~32 layers
        - 13B models: ~40 layers
        - 30B+ models: ~60+ layers
        """
        if self.num_layers is not None:
            return self.num_layers

        # Conservative layer estimation based on parameter count
        if self.params_billion <= 1:
            return 12
        elif self.params_billion <= 3:
            return 24
        elif self.params_billion <= 7:
            return 32
        elif self.params_billion <= 13:
            return 40
        elif self.params_billion <= 30:
            return 48
        elif self.params_billion <= 70:
            return 80
        else:
            # For very large models, estimate based on sqrt of params
            return int(80 * (self.params_billion / 70) ** 0.5)


# ============================================================================
# OLLAMA LIBRARY MODELS
# Modelos da biblioteca Ollama
# ============================================================================
# Source: https://ollama.com/library/
# KV cache per token is estimated for decoder-only in FP16
# KV cache por token é estimado para decoder-only em FP16
# Approximate formula: kv_cache_mb_per_token ≈ (2 * layers * hidden_size * 2 bytes) / (1024 * 1024)
# Values below are approximate and conservative
# Valores abaixo são aproximados e conservadores

_RAW_MODEL_GROUPS: tuple[LLMModel, ...] = (
    # ==========================================================================
    # TINY MODELS (< 1B) - Embedding, small tasks
    # Modelos TINY (< 1B) - Embedding, tarefas pequenas
    # ==========================================================================
    LLMModel(
        name="SmolLM2 135M",
        params_billion=0,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=0.01,
    ),
    LLMModel(
        name="all-minilm 22M/33M",
        params_billion=0,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=0.01,
    ),
    LLMModel(
        name="snowflake-arctic-embed 22M/33M/110M/137M",
        params_billion=0,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=0.01,
    ),
    LLMModel(
        name="granite-embedding 30M",
        params_billion=0,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=0.01,
    ),
    LLMModel(
        name="Granite 4 350M",
        params_billion=0,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=0.02,
    ),
    LLMModel(
        name="SmolLM / SmolLM2 360M",
        params_billion=0,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=0.03,
    ),
    LLMModel(
        name="granite-embedding 278M / paraphrase-multilingual 278M",
        params_billion=0,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=0.03,
    ),
    LLMModel(
        name="LFM2.5 Thinking 1.2B",
        params_billion=1,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=0.1,
    ),
    LLMModel(
        name="qwen2.5 0.5B / qwen2 0.5B / Qwen 0.5B",
        params_billion=1,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=0.1,
    ),
    LLMModel(
        name="qwen3 0.6B",
        params_billion=1,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=0.1,
    ),
    LLMModel(
        name="granite3.1-moe 1B / granite3-moe 1B / gemma 2B",
        params_billion=1,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=0.1,
    ),
    LLMModel(
        name="Granite 4 1B",
        params_billion=1,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=0.1,
    ),
    LLMModel(
        name="sailor2 1B",
        params_billion=1,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=0.1,
    ),
    LLMModel(
        name="Gemma 3n e2B",
        params_billion=2,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=0.15,
    ),
    LLMModel(
        name="llama3.2 1B",
        params_billion=1,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=0.1,
    ),
    LLMModel(
        name="qwen2.5 1.5B / qwen2 1.5B / Qwen 1.8B",
        params_billion=2,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=0.15,
    ),
    LLMModel(
        name="qwen3 1.7B",
        params_billion=2,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=0.15,
    ),
    LLMModel(
        name="phi / dolphin-phi 2.7B",
        params_billion=3,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=0.2,
    ),
    LLMModel(
        name="deepseek-r1 1.5B / deepscaler 1.5B / deepcoder 1.5B",
        params_billion=2,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=0.15,
    ),
    LLMModel(
        name="reader-lm 1.5B / opencoder 1.5B / phi4-mini-reasoning",
        params_billion=2,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=0.15,
    ),
    LLMModel(
        name="yi-coder 1.5B / StableLM2 1.6B / llama-pro",
        params_billion=2,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=0.15,
    ),
    LLMModel(
        name="TinyDolphin / TinyLlama 1.1B",
        params_billion=1,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=0.1,
    ),
    LLMModel(
        name="gemma3 270M / functiongemma 270M",
        params_billion=0,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=0.03,
    ),
    LLMModel(
        name="bge-m3 567M / snowflake-arctic-embed2 568M",
        params_billion=1,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=0.1,
    ),
    LLMModel(
        name="qwen3-embedding 0.6B",
        params_billion=1,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=0.1,
    ),
    # ==========================================================================
    # SMALL MODELS (1B - 4B) - Edge devices, laptops
    # Modelos PEQUENOS (1B - 4B) - Dispositivos edge, laptops
    # ==========================================================================
    LLMModel(
        name="llama3.2 3B / qwen2.5 3B / qwen3 4B / gemma3 4B",
        params_billion=3,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=0.25,
    ),
    LLMModel(
        name="phi3.5 / phi4-mini / granite3.3 2B",
        params_billion=4,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=0.3,
    ),
    LLMModel(
        name="falcon3 1B/3B / granite3.1-moe 3B / granite3-moe 3B",
        params_billion=3,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=0.25,
    ),
    LLMModel(
        name="granite3.1-dense 2B / granite3-dense 2B / granite3.2 2B",
        params_billion=2,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=0.2,
    ),
    LLMModel(
        name="granite3-guardian 2B / granite4 3B",
        params_billion=2,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=0.2,
    ),
    LLMModel(
        name="qwen2-math 1.5B / exaone-deep 2.4B / exaone3.5 2.4B",
        params_billion=2,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=0.15,
    ),
    LLMModel(
        name="stablelm-zephyr 3B / smallthinker 3B",
        params_billion=3,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=0.25,
    ),
    LLMModel(
        name="stable-code 3B / starcoder2 3B",
        params_billion=3,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=0.25,
    ),
    LLMModel(
        name="codeqwen 1.5B / qwen2.5-coder 1.5B / qwen2.5-coder 3B",
        params_billion=2,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=0.15,
    ),
    LLMModel(
        name="smollm 1.7B / smollm2 1.7B",
        params_billion=2,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=0.15,
    ),
    LLMModel(
        name="cogito 3B / dolphin3 8B / hermes3 3B",
        params_billion=3,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=0.25,
    ),
    LLMModel(
        name="ministral-3 3B / qwen2.5vl 3B / qwen3-vl 2B/4B",
        params_billion=3,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=0.25,
    ),
    LLMModel(
        name="moondream 1.8B / deepseek-ocr 3B",
        params_billion=2,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=0.15,
    ),
    LLMModel(
        name="llava-phi3 3.8B / nuextract 3.8B",
        params_billion=4,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=0.3,
    ),
    LLMModel(
        name="translategemma 4B",
        params_billion=4,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=0.3,
    ),
    LLMModel(
        name="llama-guard3 1B / shieldgemma 2B",
        params_billion=1,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=0.1,
    ),
    LLMModel(
        name="starcoder 1B/3B / gpt-oss-safeguard 20B",
        params_billion=3,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=0.25,
    ),
    LLMModel(
        name="internlm2 1.8B",
        params_billion=2,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=0.15,
    ),
    # ==========================================================================
    # MEDIUM-SMALL MODELS (7B - 10B) - Consumer GPUs, RTX 3060-4090
    # Modelos MÉDIOS-PEQUENOS (7B - 10B) - GPUs consumer, RTX 3060-4090
    # ==========================================================================
    LLMModel(
        name="llama3 / llama3.1 / mistral / qwen2.5 7B / qwen3 8B",
        params_billion=8,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=0.7,
    ),
    LLMModel(
        name="LLaMA 2 7B / llama2 / mistral 7B / yi 6B",
        params_billion=7,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=0.6,
    ),
    LLMModel(
        name="dolphin3 8B / dolphin-llama3 8B / llama3-chatqa 8B / llama3-gradient 8B",
        params_billion=8,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=0.7,
    ),
    LLMModel(
        name="phi3 3.8B / phi4 14B / phi4-reasoning 14B",
        params_billion=14,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=1.0,
    ),
    LLMModel(
        name="gemma2 9B / gemma3 12B / falcon3 7B/10B / glm4 9B",
        params_billion=9,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=0.8,
    ),
    LLMModel(
        name="gemma 7B / codegemma 2B/7B / gemma3n e4B",
        params_billion=7,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=0.6,
    ),
    LLMModel(
        name="qwen2.5 14B / qwen3 14B / qwen2.5vl 7B/32B / qwen3-vl 8B",
        params_billion=14,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=1.0,
    ),
    LLMModel(
        name="Qwen 14B / CodeQwen 14B",
        params_billion=14,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=1.0,
    ),
    LLMModel(
        name="mixtral 8x7b / nous-hermes2-mixtral / notux / dolphin-mixtral 8x7b",
        params_billion=47,
        architecture="moe-8x7b",
        precision_default="fp16",
        kv_cache_mb_per_token=2.5,
    ),
    LLMModel(
        name="falcon3 7B / codellama 7B / deepseek-llm 7B / deepseek-coder 6.7B",
        params_billion=7,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=0.6,
    ),
    LLMModel(
        name="dolphin-mistral 7B / mistral-nemo 12B / mistral-small 22B/24B",
        params_billion=12,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=0.9,
    ),
    LLMModel(
        name="ministral-3 8B/14B / granite3.3 8B / granite3.1-dense 8B / granite3-dense 8B",
        params_billion=8,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=0.7,
    ),
    LLMModel(
        name="granite3.2 8B / granite3.2-vision 2B / granite3-guardian 8B",
        params_billion=8,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=0.7,
    ),
    LLMModel(
        name="granite-code 3B/8B/20B/34B",
        params_billion=8,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=0.7,
    ),
    LLMModel(
        name="llava 7B / llava-llama3 8B / bakllava 7B / minicpm-v 8B",
        params_billion=8,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=0.7,
    ),
    LLMModel(
        name="codestral 22B / magistral 24B / devstral 24B / devstral-small-2 24B",
        params_billion=24,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=1.5,
    ),
    LLMModel(
        name="mistral-small3.2 24B / mistral-small3.1 24B / solar-pro 22B / falcon2 11B",
        params_billion=24,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=1.5,
    ),
    LLMModel(
        name="starcoder2 7B/15B / starcoder 7B/15B / dolphincoder 7B/15B",
        params_billion=7,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=0.6,
    ),
    LLMModel(
        name="sqlcoder 7B/15B / wizardcoder 33B / codebooga 34B",
        params_billion=15,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=1.0,
    ),
    LLMModel(
        name="neural-chat 7B / nous-hermes 7B / openchat 7B / wizardlm2 7B",
        params_billion=7,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=0.6,
    ),
    LLMModel(
        name="wizard-vicuna / wizard-vicuna-uncensored / wizardlm-uncensored 13B",
        params_billion=13,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=0.9,
    ),
    LLMModel(
        name="openhermes 7B / mistral-openorca 7B / samantha-mistral 7B",
        params_billion=7,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=0.6,
    ),
    LLMModel(
        name="vicuna 7B/13B/33B / xwinlm 7B/13B / orca2 7B/13B",
        params_billion=13,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=0.9,
    ),
    LLMModel(
        name="stable-beluga 7B/13B/70B / wizard-math 7B/13B/70B",
        params_billion=13,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=0.9,
    ),
    LLMModel(
        name="solar 10.7B / nous-hermes2 10.7B / meditron 7B/70B",
        params_billion=11,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=0.8,
    ),
    LLMModel(
        name="command-r7b / command-r7b-arabic / aya 8B / aya-expanse 8B/32B",
        params_billion=8,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=0.7,
    ),
    LLMModel(
        name="tulu3 8B / deepseek-coder-v2 16B / athene-v2 72B",
        params_billion=8,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=0.7,
    ),
    LLMModel(
        name="bakllava 7B / yarn-mistral 7B / yarn-llama2 7B/13B / goliath",
        params_billion=7,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=0.6,
    ),
    LLMModel(
        name="medllama2 7B / llama2-chinese 7B/13B / everythinglm 13B",
        params_billion=7,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=0.6,
    ),
    LLMModel(
        name="codeup 13B / nexusraven 13B / wizard-vicuna 13B / llama-pro",
        params_billion=13,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=0.9,
    ),
    LLMModel(
        name="dbrx 132b / mathstral 7B / magicoder 7B / duckdb-nsql 7B",
        params_billion=7,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=0.6,
    ),
    LLMModel(
        name="mistrallite 7B / starling-lm 7B / notus 7B / bespoke-minicheck 7B",
        params_billion=7,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=0.6,
    ),
    LLMModel(
        name="firefunction-v2 70B / open-orca-platypus2 13B / reflection 70B",
        params_billion=13,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=0.9,
    ),
    LLMModel(
        name="r1-1776 70B / nemotron 70B / llama3-gradient 70B",
        params_billion=13,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=0.9,
    ),
    LLMModel(
        name="nemotron-mini 4B / reader-lm 0.5B/1.5B / llama-guard3 1B/8B",
        params_billion=4,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=0.3,
    ),
    LLMModel(
        name="llama2-uncensored 7B/70B / dolphin-llama3 70B / alfred 40B",
        params_billion=40,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=2.0,
    ),
    LLMModel(
        name="phind-codellama 34B / marco-o1 7B / olmo2 7B/13B / olmo-3 7B/32B",
        params_billion=7,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=0.6,
    ),
    LLMModel(
        name="sailor2 1B/8B/20B / qwen2-math 7B/72B / exaone-deep 7.8B",
        params_billion=7,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=0.6,
    ),
    LLMModel(
        name="rnj-1 8B / cogito 8B / dolphin-mixtral 8x22b / hermes3 8B",
        params_billion=8,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=0.7,
    ),
    LLMModel(
        name="yi 9B / codegeex4 9B / shieldgemma 9B / gemma2 9B",
        params_billion=9,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=0.8,
    ),
    LLMModel(
        name="zephyr 7B/141B / mistral-large 123B / command-r 35B / command-a 111B",
        params_billion=7,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=0.6,
    ),
    LLMModel(
        name="orca-mini 3B/7B/13B/70B / deepcoder 14B / opencoder 1.5B/8B",
        params_billion=7,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=0.6,
    ),
    LLMModel(
        name="megadolphin 120b / gpt-oss 20B/120B / yi-coder 9B",
        params_billion=9,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=0.8,
    ),
    LLMModel(
        name="granite3.2-vision 2B / qwen3-coder 30B/480B / qwen3-next 80B",
        params_billion=9,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=0.8,
    ),
    # ==========================================================================
    # MEDIUM MODELS (13B - 40B) - High-end consumer, datacenter entry
    # Modelos MÉDIOS (13B - 40B) - Consumer high-end, entrada datacenter
    # ==========================================================================
    LLMModel(
        name="LLaMA 2 13B / llama2",
        params_billion=13,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=0.9,
    ),
    LLMModel(
        name="llava 13B / llama2-chinese 13B / llama-pro",
        params_billion=13,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=0.9,
    ),
    LLMModel(
        name="command-r 35B / aya 35B / stablelm2 12B",
        params_billion=35,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=2.0,
    ),
    LLMModel(
        name="LLaMA 3.1 34B / Yi 34B / nous-hermes2 34B",
        params_billion=34,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=1.8,
    ),
    LLMModel(
        name="olmo2 13B / olmo-3 7B/32B / olmo-3.1 32B",
        params_billion=13,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=0.9,
    ),
    LLMModel(
        name="deepseek-coder 33B / deepseek-v2 16B/236B / deepseek-v2.5 236B",
        params_billion=33,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=1.8,
    ),
    LLMModel(
        name="codellama 34B / llava 34B / phind-codellama 34B",
        params_billion=34,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=1.8,
    ),
    LLMModel(
        name="internlm2 20B / qwq 32B / cogito 32B / opencoder 8B",
        params_billion=20,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=1.2,
    ),
    LLMModel(
        name="sailor2 20B / mistral-small 22B/24B / magistral 24B",
        params_billion=24,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=1.5,
    ),
    LLMModel(
        name="codestral 22B / falcon2 11B / devstral 24B / solar-pro 22B",
        params_billion=22,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=1.4,
    ),
    LLMModel(
        name="gemma2 27B / shieldgemma 27B / translategemma 27B",
        params_billion=27,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=1.6,
    ),
    LLMModel(
        name="Qwen3 30B-A3B",
        params_billion=30,
        architecture="moe",
        precision_default="fp16",
        # Qwen3 30B-A3B uses a much smaller attention state than dense 30B models
        # (48 layers, 4 KV heads, head_dim 64), so long contexts are feasible.
        kv_cache_mb_per_token=0.046875,
        context_length_max=262144,
        num_layers=48,
    ),
    LLMModel(
        name="Qwen3.6 27B",
        params_billion=27,
        architecture="moe",
        precision_default="fp16",
        kv_cache_mb_per_token=0.046875,
        context_length_max=262144,
        num_layers=48,
    ),
    LLMModel(
        name="llama3.2-vision 11B / falcon3 10B / exaone3.5 32B",
        params_billion=11,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=0.8,
    ),
    LLMModel(
        name="nemotron-3-nano 30B / nemotron-3 30B / llama3-groq-tool-use 70B",
        params_billion=30,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=1.7,
    ),
    # ==========================================================================
    # LARGE MODELS (70B - 100B) - Datacenter GPUs, A100 40GB-80GB
    # Modelos GRANDES (70B - 100B) - GPUs datacenter, A100 40GB-80GB
    # ==========================================================================
    LLMModel(
        name="LLaMA 2 70B / LLaMA 3.1 70B / llama3 / llama3.3 70B",
        params_billion=70,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=4.27,
    ),
    LLMModel(
        name="llama3.1 405B / hermes3 405B / cogito-2.1 671B",
        params_billion=70,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=4.27,
    ),
    LLMModel(
        name="llama2-uncensored 70B / dolphin-llama3 70B / hermes3 70B",
        params_billion=70,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=4.27,
    ),
    LLMModel(
        name="dolphin3 70B / reflection 70B / r1-1776 70B / r1-1776 671B",
        params_billion=70,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=4.27,
    ),
    LLMModel(
        name="deepseek-r1 70B / deepseek-v3 671B / deepseek-v3.1 671B",
        params_billion=70,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=4.27,
    ),
    LLMModel(
        name="nemotron 70B / llama3-chatqa 70B / llama3-gradient 70B",
        params_billion=70,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=4.27,
    ),
    LLMModel(
        name="llama3-groq-tool-use 70B / firefunction-v2 70B / tulu3 70B",
        params_billion=70,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=4.27,
    ),
    LLMModel(
        name="stable-beluga 70B / wizard-math 70B / orca-mini 70B / meditron 70B",
        params_billion=70,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=4.27,
    ),
    LLMModel(
        name="gemma2 27B / qwen2.5 72B / qwen3-vl 235B / qwen3 235B",
        params_billion=72,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=4.5,
    ),
    LLMModel(
        name="qwen2.5vl 72B / qwen2-math 72B / athene-v2 72B",
        params_billion=72,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=4.5,
    ),
    LLMModel(
        name="command-r-plus 104B / mistral-large 123B / gpt-oss 120B",
        params_billion=104,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=5.5,
    ),
    LLMModel(
        name="gpt-oss-safeguard 120B / devstral-2 123B / zephyr 141B",
        params_billion=123,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=6.0,
    ),
    LLMModel(
        name="command-a 111B / mistral-large 123B / zephyr 141B",
        params_billion=111,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=5.8,
    ),
    LLMModel(
        name="mixtral 8x22b / wizardlm2 8x22b / nous-hermes2-mixtral 8x7b",
        params_billion=141,
        architecture="moe-8x22b",
        precision_default="fp16",
        kv_cache_mb_per_token=6.5,
    ),
    LLMModel(
        name="dbrx 132B / alfred 40B / qwq 32B / cogito 32B",
        params_billion=132,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=6.8,
    ),
    # ==========================================================================
    # XL MODELS (180B+) - Multi-GPU, H100, enterprise
    # Modelos XL (180B+) - Multi-GPU, H100, enterprise
    # ==========================================================================
    LLMModel(
        name="Falcon 180B",
        params_billion=180,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=8.0,
    ),
    LLMModel(
        name="falcon 40B/180B / deepseek-llm 67B",
        params_billion=180,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=8.0,
    ),
    LLMModel(
        name="Qwen 110B / qwen 72B/110B",
        params_billion=110,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=5.0,
    ),
    LLMModel(
        name="llama4 16x17b / llama4 128x17b / deepseek-v3 671B",
        params_billion=272,
        architecture="moe",
        precision_default="fp16",
        kv_cache_mb_per_token=10.0,
    ),
    LLMModel(
        name="deepseek-r1 671B / deepseek-v3 671B / deepseek-v3.1 671B / deepseek-v3.2",
        params_billion=671,
        architecture="moe",
        precision_default="fp16",
        kv_cache_mb_per_token=20.0,
    ),
    LLMModel(
        name="qwen3-coder 480B / qwen3 235B / qwen3-next 80B",
        params_billion=480,
        architecture="moe",
        precision_default="fp16",
        kv_cache_mb_per_token=15.0,
    ),
    LLMModel(
        name="llama3.1 405B / hermes3 405B / cogito-2.1 671B",
        params_billion=405,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=15.0,
    ),
    # ==========================================================================
    # EMBEDDING MODELS (for reference, small VRAM)
    # Modelos de EMBEDDING (para referência, VRAM pequeno)
    # ==========================================================================
    LLMModel(
        name="nomic-embed-text / mxbai-embed-large 335M",
        params_billion=0,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=0.03,
    ),
    LLMModel(
        name="bge-large / snowflake-arctic-embed 335M",
        params_billion=0,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=0.03,
    ),
    LLMModel(
        name="granite-embedding 30M/278M / embeddinggemma 300M",
        params_billion=0,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=0.03,
    ),
    LLMModel(
        name="qwen3-embedding 0.6B/4B/8B / bge-m3 567M",
        params_billion=1,
        architecture="decoder-only",
        precision_default="fp16",
        kv_cache_mb_per_token=0.1,
    ),
)


_SIZE_PATTERN = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)\s*([BbMm])\b")
_MOE_PATTERN = re.compile(r"(\d+)\s*x\s*(\d+(?:\.\d+)?)\s*[Bb]", re.IGNORECASE)
_MOE_PARAM_OVERRIDES = {
    "8x7b": 47.0,
    "8x22b": 141.0,
}


def _format_params(params_billion: float) -> str:
    if params_billion >= 1:
        return f"{params_billion:g}B"
    return f"{params_billion * 1000:g}M"


def _estimate_kv_cache(params_billion: float) -> float:
    """Estimate KV cache MB/token for expanded variants without exact metadata."""
    if params_billion <= 1:
        return 0.05 if params_billion < 1 else 0.1
    if params_billion <= 3:
        return 0.15
    if params_billion <= 7:
        return 0.6
    if params_billion <= 13:
        return 0.9
    if params_billion <= 30:
        return 1.7
    if params_billion <= 70:
        return 4.27
    if params_billion <= 100:
        return 5.0
    return 5.0 * (params_billion / 100) ** 0.5


def _extract_sizes(name: str) -> list[float]:
    sizes: list[float] = []
    moe_match = _MOE_PATTERN.search(name)
    if moe_match is not None:
        experts, expert_size = moe_match.groups()
        expert_size_float = float(expert_size)
        moe_key = f"{experts}x{expert_size_float:g}b"
        sizes.append(_MOE_PARAM_OVERRIDES.get(moe_key, float(experts) * expert_size_float))
    else:
        for match in _SIZE_PATTERN.finditer(name):
            size = float(match.group(1))
            unit = match.group(2).lower()
            sizes.append(size / 1000 if unit == "m" else size)

    unique_sizes: list[float] = []
    for size in sizes:
        if size not in unique_sizes:
            unique_sizes.append(size)
    return unique_sizes


def _name_for_size(raw_name: str, params_billion: float) -> str:
    """Create a concrete display name for one size from a grouped source name."""
    size_label = _format_params(params_billion)
    if size_label.lower() in raw_name.lower() and "/" not in raw_name:
        return raw_name

    if _MOE_PATTERN.search(raw_name):
        return raw_name

    base_match = _SIZE_PATTERN.search(raw_name)
    if base_match is None:
        return raw_name

    base = raw_name[: base_match.start()].strip(" -_/()")
    if not base:
        return raw_name
    return f"{base} {size_label}"


def _expand_group(group: LLMModel) -> list[LLMModel]:
    entries: list[LLMModel] = []
    for raw_name in [part.strip() for part in group.name.split(" / ") if part.strip()]:
        sizes = _extract_sizes(raw_name) or [group.params_billion]
        for size in sizes:
            architecture = "moe" if _MOE_PATTERN.search(raw_name) else group.architecture
            kv_cache = group.kv_cache_mb_per_token if size == group.params_billion else _estimate_kv_cache(size)
            entries.append(
                LLMModel(
                    name=_name_for_size(raw_name, size),
                    params_billion=size,
                    architecture=architecture,
                    precision_default=group.precision_default,
                    kv_cache_mb_per_token=kv_cache,
                    format=group.format,
                    context_length_max=group.context_length_max,
                    num_layers=group.num_layers,
                )
            )
    return entries


def _build_catalog(groups: tuple[LLMModel, ...]) -> tuple[LLMModel, ...]:
    deduped: dict[tuple[str, float], LLMModel] = {}
    for group in groups:
        for entry in _expand_group(group):
            deduped.setdefault((entry.name.lower(), entry.params_billion), entry)
    return tuple(sorted(deduped.values(), key=lambda entry: (entry.params_billion, entry.name.lower())))


LLM_MODELS: tuple[LLMModel, ...] = _build_catalog(_RAW_MODEL_GROUPS)


def get_model_by_size(size_billion: float) -> LLMModel | None:
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


def get_all_models() -> list[LLMModel]:
    """Returns all available models.

    Retorna todos os modelos disponíveis.

    Returns:
        List of all LLMModel instances
    """
    return list(LLM_MODELS)


def get_models_by_size_range(min_billion: int, max_billion: int) -> list[LLMModel]:
    """Returns models within a size range.

    Retorna modelos dentro de uma faixa de tamanho.

    Args:
        min_billion: Minimum size in billions
        max_billion: Maximum size in billions

    Returns:
        List of LLMModel instances within the range
    """
    return [m for m in LLM_MODELS if min_billion <= m.params_billion <= max_billion]


def search_models_by_name(query: str) -> list[LLMModel]:
    """Search models by name (case-insensitive partial match).

    Busca modelos por nome (correspondência parcial case-insensitive).

    Args:
        query: Search query string

    Returns:
        List of matching LLMModel instances
    """
    query_lower = query.lower()
    return [m for m in LLM_MODELS if query_lower in m.name.lower()]
