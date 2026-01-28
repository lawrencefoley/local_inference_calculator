#!/usr/bin/env python3
"""
CLI for LLM local inference viability calculator.

Allows quickly discovering which models run on which GPU
for a given context size.

CLI para calculadora de viabilidade de inferência local de LLMs.
Permite descobrir rapidamente quais modelos rodam em qual GPU
para um determinado tamanho de contexto.
"""

import argparse
import csv
import json
import sys
from typing import List

from models import LLMModel, get_all_models, get_model_by_size
from gpus import GPU, get_all_gpus, get_consumer_gpus, get_datacenter_gpus, get_gpu_by_name
from calculator import (
    VRAMCalculator, Quantization, InferenceResult, Status, CalculationMode,
    LayerOffloadCalculator, LayerOffloadResult,
    CPUOffloadCalculator, CPUOffloadResult,
)
from formats import ModelFormat, detect_gguf_quantization, get_format_from_filename
from multi_gpu import (
    MultiGPUConfig, MultiGPUCalculator, MultiGPUMode,
    parse_gpu_config_string, create_multi_gpu_config,
)


# Terminal colors for enhanced readability
# Cores de terminal para melhor legibilidade
class Colors:
    """ANSI color codes for terminal output.

    Códigos de cores ANSI para saída de terminal.
    """
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Foreground colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Background colors
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"

    @staticmethod
    def ok(text: str) -> str:
        """Green text for success/OK messages."""
        return f"{Colors.GREEN}{text}{Colors.RESET}"

    @staticmethod
    def warning(text: str) -> str:
        """Yellow text for warnings."""
        return f"{Colors.YELLOW}{text}{Colors.RESET}"

    @staticmethod
    def warn(text: str) -> str:
        """Yellow text for warnings (alias)."""
        return f"{Colors.YELLOW}{text}{Colors.RESET}"

    @staticmethod
    def error(text: str) -> str:
        """Red text for errors."""
        return f"{Colors.RED}{text}{Colors.RESET}"

    @staticmethod
    def info(text: str) -> str:
        """Blue text for info."""
        return f"{Colors.BLUE}{text}{Colors.RESET}"

    @staticmethod
    def dim(text: str) -> str:
        """Dim text for less emphasis."""
        return f"{Colors.DIM}{text}{Colors.RESET}"

    @staticmethod
    def bold(text: str) -> str:
        """Bold text."""
        return f"{Colors.BOLD}{text}{Colors.RESET}"


def print_table(
    results: List[InferenceResult],
    group_by_gpu: bool = False,
    show_only_runs: bool = False,
):
    """Prints results in ASCII table format.

    Imprime resultados em formato de tabela ASCII.

    Args:
        results: List of results / Lista de resultados
        group_by_gpu: Group by GPU instead of model / Agrupa por GPU
        show_only_runs: Show only running combinations / Apenas combinações que rodam
    """
    if show_only_runs:
        results = [r for r in results if r.status == Status.RUNS]

    if not results:
        print("\nNo combinations found. / Nenhuma combinação encontrada.")
        return

    # Header / Cabeçalho
    if group_by_gpu:
        header = f"{'GPU':<25} {'VRAM':<8} {'Model':<30} {'Needed':<10} {'Status':<10}"
    else:
        header = f"{'Model':<30} {'Needed':<10} {'GPU':<25} {'VRAM':<8} {'Status':<10}"

    separator = "-" * len(header)

    print(f"\n{header}")
    print(separator)

    # Sort: running first, then by VRAM required
    # Ordenar: roda primeiro, depois por VRAM necessária
    sorted_results = sorted(
        results,
        key=lambda r: (r.status != Status.RUNS, r.required_vram_gb),
    )

    for r in sorted_results:
        status_str = r.status.value
        if r.status == Status.RUNS:
            status_str = f"\033[92m{status_str}\033[0m"  # Green / Verde
        else:
            status_str = f"\033[91m{status_str}\033[0m"  # Red / Vermelho

        if group_by_gpu:
            row = (
                f"{r.gpu_name:<25} "
                f"{r.gpu_vram_gb:<8} "
                f"{f'{r.model_params_billion}B':<30} "
                f"{r.required_vram_gb:<10.1f} "
                f"{status_str:<10}"
            )
        else:
            row = (
                f"{f'{r.model_params_billion}B':<30} "
                f"{r.required_vram_gb:<10.1f} "
                f"{r.gpu_name:<25} "
                f"{r.gpu_vram_gb:<8} "
                f"{status_str:<10}"
            )

        print(row)

        # Show warning if present
        # Mostrar aviso se houver
        if r.warning:
            print(f"  ⚠️  {r.warning}")


def print_summary_by_model(results: List[InferenceResult]):
    """Prints summary grouped by model size.

    Imprime resumo agrupado por tamanho de modelo.

    Shows for each model which GPUs support it.
    Mostra para cada modelo quais GPUs suportam.
    """
    print("\n" + "=" * 70)
    print("SUMMARY BY MODEL / RESUMO POR MODELO")
    print("=" * 70)

    # Group by model size
    # Agrupar por tamanho de modelo
    from collections import defaultdict

    by_model: dict[int, List[InferenceResult]] = defaultdict(list)
    for r in results:
        by_model[r.model_params_billion].append(r)

    for size in sorted(by_model.keys()):
        runnable = [r for r in by_model[size] if r.status == Status.RUNS]
        not_runnable = [r for r in by_model[size] if r.status != Status.RUNS]

        print(f"\nModel {size}B:")

        if runnable:
            print(f"  ✓ RUNS on: / RODA em: {', '.join(sorted(set(r.gpu_name for r in runnable)))}")
        else:
            print(f"  ✗ Doesn't run on any listed GPU / Não roda em nenhuma GPU listada")

        if not_runnable:
            closest = min(not_runnable, key=lambda r: r.required_vram_gb - r.gpu_vram_gb)
            print(f"  ⚠️  Closest: / Mais próximo: {closest.gpu_name} (needs / precisa de {closest.required_vram_gb:.1f} GB)")


def print_summary_by_gpu(results: List[InferenceResult]):
    """Prints summary grouped by GPU.

    Imprime resumo agrupado por GPU.

    Shows for each GPU which models it supports.
    Mostra para cada GPU quais modelos suporta.
    """
    print("\n" + "=" * 70)
    print("SUMMARY BY GPU / RESUMO POR GPU")
    print("=" * 70)

    # Group by GPU
    # Agrupar por GPU
    from collections import defaultdict

    by_gpu: dict[str, List[InferenceResult]] = defaultdict(list)
    for r in results:
        by_gpu[r.gpu_name].append(r)

    # Sort GPUs by VRAM
    # Ordenar GPUs por VRAM
    gpu_vram = {r.gpu_name: r.gpu_vram_gb for r in results}
    sorted_gpus = sorted(by_gpu.keys(), key=lambda g: gpu_vram[g])

    for gpu_name in sorted_gpus:
        gpu_results = by_gpu[gpu_name]
        runnable = [r for r in gpu_results if r.status == Status.RUNS]

        vram = gpu_results[0].gpu_vram_gb
        print(f"\n{gpu_name} ({vram} GB):")

        if runnable:
            models = sorted(set(r.model_params_billion for r in runnable))
            print(f"  ✓ Supports: / Suporta: {', '.join(f'{m}B' for m in models)}")
        else:
            print(f"  ✗ Doesn't support any listed model / Não suporta nenhum modelo listado")


def print_layer_offload_result(result: LayerOffloadResult, model: LLMModel, gpu: GPU):
    """Prints layer offload configuration result.

    Imprime resultado da configuração de offload de camadas.

    Args:
        result: Layer offload calculation result
        model: LLM model
        gpu: GPU being used
    """
    print("\n" + "=" * 70)
    print(f"{Colors.BOLD}OPTIMAL LAYER OFFLOAD CONFIGURATION{Colors.RESET}")
    print("=" * 70)

    print(f"\n{Colors.CYAN}Model:{Colors.RESET} {model.name} ({model.params_billion}B parameters)")
    print(f"{Colors.CYAN}GPU:{Colors.RESET}   {gpu.name} ({gpu.vram_gb} GB VRAM)")
    print(f"{Colors.CYAN}Total Layers:{Colors.RESET} {result.total_layers}")

    print(f"\n{Colors.CYAN}Layer Distribution:{Colors.RESET}")
    print(f"  Layers on GPU:  {Colors.ok(str(result.layers_on_gpu))}")
    print(f"  Layers on CPU:  {Colors.warning(str(result.layers_on_cpu)) if result.layers_on_cpu > 0 else Colors.dim(str(result.layers_on_cpu))}")
    print(f"  Offload Ratio:  {result.offload_ratio:.1%}")

    print(f"\n{Colors.CYAN}Memory Usage:{Colors.RESET}")
    print(f"  GPU VRAM used:  {result.gpu_vram_used:.2f} GB / {gpu.vram_gb} GB")
    print(f"  CPU RAM used:   {result.cpu_ram_used:.2f} GB")
    if gpu.vram_gb > 0:
        gpu_util = (result.gpu_vram_used / gpu.vram_gb) * 100
        print(f"  GPU utilization: {gpu_util:.1f}%")

    print(f"\n{Colors.CYAN}Performance Impact:{Colors.RESET}")
    if result.status == "full_gpu":
        print(f"  {Colors.ok('✓ Full GPU acceleration - no performance impact')}")
    elif result.status == "cpu_only":
        print(f"  {Colors.error('✗ CPU only inference - ~10-20x slower')}")
    else:
        print(f"  Estimated slowdown: {Colors.warning(f'{result.performance_impact:.1f}%')}")
        if result.performance_impact < 20:
            print(f"  {Colors.info('Minimal impact - good for interactive use')}")
        elif result.performance_impact < 50:
            print(f"  {Colors.warning('Moderate impact - usable with some patience')}")
        else:
            print(f"  {Colors.error('Significant impact - consider more VRAM or smaller model')}")

    print(f"\n{Colors.CYAN}Recommended Configuration:{Colors.RESET}")
    print(f"  llama.cpp:  {Colors.bold(f'--gpu-layers {result.layers_on_gpu}')}")
    print(f"  AutoGPTQ:   {Colors.bold(f'--gpu-memory {result.gpu_vram_used:.1f}G')}")

    if result.status == "partial_offload":
        print(f"\n{Colors.DIM}Note: Layers on CPU are accessed via PCIe, which is slower than GPU VRAM.{Colors.RESET}")
        print(f"{Colors.DIM}      Consider quantization (INT4) to fit more layers on GPU.{Colors.RESET}")

    print("=" * 70)


def print_cpu_offload_result(result: CPUOffloadResult, model: LLMModel):
    """Prints CPU offload configuration result.

    Imprime resultado da configuração de offload de CPU.

    Args:
        result: CPU offload calculation result
        model: LLM model
    """
    print("\n" + "=" * 70)
    print(f"{Colors.BOLD}CPU OFFLOAD ANALYSIS{Colors.RESET}")
    print("=" * 70)

    print(f"\n{Colors.CYAN}System Requirements:{Colors.RESET}")
    print(f"  System RAM required: {result.system_ram_required:.2f} GB")
    print(f"  System RAM available: {result.system_ram_available:.2f} GB")
    if result.fits_in_ram:
        print(f"  Status: {Colors.ok('✓ Fits in system RAM')}")
    else:
        print(f"  Status: {Colors.error(f'✗ Need {result.system_ram_required - result.system_ram_available:.2f} GB more RAM')}")

    print(f"\n{Colors.CYAN}PCIe Configuration:{Colors.RESET}")
    print(f"  Generation: PCIe {result.pcie_generation}")
    pcie_bandwidth = {"3.0": 12, "4.0": 24, "5.0": 48}.get(result.pcie_generation, 24)
    print(f"  Bandwidth: ~{pcie_bandwidth} GB/s effective")

    print(f"\n{Colors.CYAN}Performance Estimate:{Colors.RESET}")
    print(f"  Token speed: ~{result.estimated_token_speed:.1f} tokens/second")
    if result.speed_vs_full_gpu >= 0.8:
        print(f"  Speed ratio: {Colors.ok(f'{result.speed_vs_full_gpu:.1%} of full GPU')}")
    elif result.speed_vs_full_gpu >= 0.3:
        print(f"  Speed ratio: {Colors.warning(f'{result.speed_vs_full_gpu:.1%} of full GPU')}")
    else:
        print(f"  Speed ratio: {Colors.error(f'{result.speed_vs_full_gpu:.1%} of full GPU')}")

    # Print layer offload details
    offload = result.offload_config
    print(f"\n{Colors.CYAN}Layer Distribution:{Colors.RESET}")
    print(f"  Layers on GPU:  {Colors.ok(str(offload.layers_on_gpu))}")
    print(f"  Layers on CPU:  {Colors.warning(str(offload.layers_on_cpu)) if offload.layers_on_cpu > 0 else Colors.dim(str(offload.layers_on_cpu))}")
    print(f"  Offload Ratio:  {offload.offload_ratio:.1%}")
    print(f"  GPU VRAM used:  {offload.gpu_vram_used:.2f} GB")
    print(f"  CPU RAM used:   {offload.cpu_ram_used:.2f} GB")

    if offload.status == "partial_offload":
        print(f"\n{Colors.DIM}Note: Layers on CPU are accessed via PCIe, which is slower than GPU VRAM.{Colors.RESET}")
        print(f"{Colors.DIM}      Consider quantization (INT4) to fit more layers on GPU.{Colors.RESET}")

    print("=" * 70)


def print_multi_gpu_result(result, model: LLMModel):
    """Prints multi-GPU configuration result.

    Imprime resultado da configuração multi-GPU.

    Args:
        result: MultiGPUResult from MultiGPUCalculator
        model: LLM model
    """
    print("\n" + "=" * 70)
    print(f"{Colors.BOLD}MULTI-GPU CONFIGURATION{Colors.RESET}")
    print("=" * 70)

    print(f"\n{Colors.CYAN}Model:{Colors.RESET} {model.name} ({model.params_billion}B parameters)")

    status_text = Colors.ok('RUNS') if result.status == 'runs' else Colors.error("DOESN'T RUN")
    print(f"{Colors.CYAN}Status:{Colors.RESET} {status_text}")

    if result.bottleneck_gpu:
        print(f"  {Colors.warning(f'Bottleneck: {result.bottleneck_gpu}')}")
    if result.communication_overhead_gb > 0:
        print(f"  Communication overhead: {result.communication_overhead_gb:.2f} GB")

    print(f"\n{Colors.CYAN}Per-GPU Allocation:{Colors.RESET}")
    for gpu_name, alloc in result.per_gpu_allocation.items():
        status = Colors.ok("✓") if alloc.vram_used_gb <= alloc.gpu.vram_gb else Colors.error("✗")
        print(f"  {status} {gpu_name:<20} {alloc.vram_used_gb:6.2f} GB / {alloc.gpu.vram_gb} GB")

        if alloc.layer_count > 0:
            print(f"      Layers: {alloc.layer_count}")
        if alloc.shard_ratio > 0:
            print(f"      Shard: {alloc.shard_ratio:.1%}")

    print(f"\n{Colors.CYAN}Framework Configuration:{Colors.RESET}")
    for framework, config in result.recommended_framework_config.items():
        print(f"  {framework}: {config}")

    print("=" * 70)


def print_summary_by_gpu(results: List[InferenceResult]):
    """Prints summary grouped by GPU.

    Imprime resumo agrupado por GPU.

    Shows for each GPU which models it supports.
    Mostra para cada GPU quais modelos suporta.
    """
    print("\n" + "=" * 70)
    print("SUMMARY BY GPU / RESUMO POR GPU")
    print("=" * 70)

    # Group by GPU
    # Agrupar por GPU
    from collections import defaultdict

    by_gpu: dict[str, List[InferenceResult]] = defaultdict(list)
    for r in results:
        by_gpu[r.gpu_name].append(r)

    # Sort GPUs by VRAM
    # Ordenar GPUs por VRAM
    gpu_vram = {r.gpu_name: r.gpu_vram_gb for r in results}
    sorted_gpus = sorted(by_gpu.keys(), key=lambda g: gpu_vram[g])

    for gpu_name in sorted_gpus:
        gpu_results = by_gpu[gpu_name]
        runnable = [r for r in gpu_results if r.status == Status.RUNS]

        vram = gpu_results[0].gpu_vram_gb
        print(f"\n{gpu_name} ({vram} GB):")

        if runnable:
            models = sorted(set(r.model_params_billion for r in runnable))
            print(f"  ✓ Supports: / Suporta: {', '.join(f'{m}B' for m in models)}")
        else:
            print(f"  ✗ Doesn't support any listed model / Não suporta nenhum modelo listado")


def export_csv(results: List[InferenceResult], filepath: str):
    """Exports results to CSV.

    Exporta resultados para CSV.
    """
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Model", "Params_B", "GPU", "GPU_VRAM_GB",
            "VRAM_Required_GB", "Status", "VRAM_Free_%", "Quantization", "Warning"
        ])
        for r in results:
            writer.writerow([
                r.model_name,
                r.model_params_billion,
                r.gpu_name,
                r.gpu_vram_gb,
                round(r.required_vram_gb, 2),
                r.status.value,
                round(r.vram_free_percent, 1),
                r.quantization.value,
                r.warning or "",
            ])
    print(f"\n✓ Results exported to: / Resultados exportados para: {filepath}")


def export_json(results: List[InferenceResult], filepath: str, context_tokens: int, quantization: Quantization):
    """Exports results to JSON.

    Exporta resultados para JSON.
    """
    data = {
        "context_tokens": context_tokens,
        "quantization": quantization.value,
        "results": [r.to_dict() for r in results],
    }
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Results exported to: / Resultados exportados para: {filepath}")


def list_models():
    """Prints all available models.

    Imprime todos os modelos disponíveis.
    """
    models = get_all_models()
    print("\n" + "=" * 70)
    print("AVAILABLE MODELS / MODELOS DISPONÍVEIS")
    print("=" * 70)

    for model in models:
        print(f"\n  [{model.params_billion}B] {model.name}")
        print(f"     Architecture: {model.architecture}")
        print(f"     Default precision: {model.precision_default}")
        print(f"     KV cache: {model.kv_cache_mb_per_token} MB/token (FP16 baseline)")

    print("\n" + "=" * 70)
    print("\nUsage: python main.py --model <size>  (e.g., --model 7)")


def print_model_vram_breakdown(model: LLMModel, context_tokens: int, quantization: Quantization, calculation_mode: CalculationMode = CalculationMode.CONSERVATIVE):
    """Prints detailed VRAM breakdown for a specific model.

    Imprime breakdown detalhado de VRAM para um modelo específico.

    Args:
        model: LLM model to analyze
        context_tokens: Context size in tokens
        quantization: Quantization type
        calculation_mode: Calculation mode
    """
    from calculator import VRAMCalculator, BYTES_PER_PARAM, KV_CACHE_MULTIPLIER

    calc = VRAMCalculator(quantization=quantization, calculation_mode=calculation_mode)
    breakdown = calc.calculate_total_vram(model, context_tokens)

    # Calculate real-world estimates
    # Idle: model loaded, no active generation
    # Peak: during token generation (KV cache fully allocated)
    idle_estimate = breakdown.params_memory_gb + breakdown.overhead_gb
    peak_estimate = breakdown.total_vram_gb

    # Determine if close to 24GB limit for warning color
    is_tight_24gb = 22 <= breakdown.total_vram_gb <= 24

    print("\n" + "=" * 70)
    print(f"{Colors.BOLD}VRAM BREAKDOWN: {model.name}{Colors.RESET}")
    print("=" * 70)
    print(f"\n{Colors.CYAN}Configuration:{Colors.RESET}")
    print(f"  Batch size:           1 (inference only)")
    print(f"  Context:              {context_tokens:,} tokens")
    print(f"  Quantization backend: {Colors.bold(quantization.value.upper())} ({BYTES_PER_PARAM[quantization]} bytes/param)")
    print(f"  KV cache precision:   FP16 (default) | Quantized (experimental)")
    print(f"  Calculation mode:     {calculation_mode.value}")
    print(f"  Memory allocator:     PyTorch-style (HF Transformers, vLLM)")

    print(f"\n{Colors.CYAN}Memory Breakdown:{Colors.RESET}")
    print(f"  Model parameters:     {breakdown.params_memory_gb:.2f} GB")
    print(f"  Overhead (30%):       {Colors.dim(f'{breakdown.overhead_gb:.2f} GB')}")
    print(f"  Model + overhead:     {breakdown.model_with_overhead_gb:.2f} GB")
    print(f"  KV cache (FP16):      {Colors.warning(f'{breakdown.kv_cache_gb:.2f} GB')} ({calculation_mode.value} mode)")
    print(f"  " + "-" * 40)
    print(f"  {Colors.BOLD}TOTAL VRAM:{Colors.RESET:15} {Colors.bold(f'{breakdown.total_vram_gb:.2f} GB')}")

    print(f"\n{Colors.CYAN}Real-World Usage Estimates:{Colors.RESET}")
    print(f"  Idle (model loaded):      {idle_estimate:.2f} GB")
    print(f"  Peak (generation):        {Colors.warning(f'{peak_estimate:.2f} GB')}")

    print(f"\nMinimum GPU VRAM required: {breakdown.total_vram_gb:.1f} GB")
    print(f"Recommended (with margin): {Colors.ok(f'{breakdown.total_vram_gb * 1.1:.1f} GB')}")

    # Show assumptions for production mode
    print(f"\n{Colors.DIM}  Assumptions:{Colors.RESET}")
    print(f"    • batch_size = 1 (no batching)")
    print(f"    • No LoRA adapters active")
    print(f"    • No speculative decoding")
    print(f"    • No tool calling overhead")
    print(f"    • PyTorch allocator (TensorRT-LLM / llama.cpp may vary)")
    print(f"    • KV cache in FP16 (quantized KV cache is experimental)")
    print(f"      → Weights INT4 ≠ KV cache INT4 in most frameworks")

    print(f"\n{Colors.DIM}  Scaling notes:{Colors.RESET}")
    print(f"    • KV cache scales linearly with context length")
    print(f"      → 16k context ≈ {breakdown.kv_cache_gb * 2:.1f} GB KV cache")
    print(f"      → 32k context ≈ {breakdown.kv_cache_gb * 4:.1f} GB KV cache")
    print(f"    • KV cache scales linearly with batch size")
    print(f"      → batch_size = 4 ≈ +{breakdown.kv_cache_gb * 3:.1f} GB KV cache")
    print(f"    • VRAM calculations do not account for throughput or latency")
    print(f"      → This tool measures {Colors.bold('capacity')}, not speed")

    # Warning for 24GB GPUs near limit
    if is_tight_24gb:
        print(f"\n  {Colors.BG_RED}{Colors.WHITE} ⚠️  WARNING: 24GB GPUs run at the limit.{Colors.RESET}")
        print(f"     Any batching, adapters (LoRA), or additional features may cause OOM.")

    print("=" * 70)


def estimate_kv_cache(params_billion: int) -> float:
    """Estimate KV cache per token based on model size.

    Estima KV cache por token baseado no tamanho do modelo.

    Uses a conservative formula based on decoder-only architecture.
    Usa uma fórmula conservadora baseada em arquitetura decoder-only.

    Args:
        params_billion: Model size in billions of parameters

    Returns:
        Estimated KV cache in MB per token (FP16)
    """
    # Approximate KV cache scaling based on model size
    # Escalonamento aproximado de KV cache baseado no tamanho do modelo
    # Formula: kv_cache ≈ 0.6 * sqrt(params_billion / 7)
    # This is a rough approximation for decoder-only models
    if params_billion <= 1:
        return 0.05
    elif params_billion <= 3:
        return 0.15
    elif params_billion <= 7:
        return 0.4
    elif params_billion <= 13:
        return 0.6
    elif params_billion <= 30:
        return 1.0
    elif params_billion <= 70:
        return 2.0
    elif params_billion <= 100:
        return 3.0
    else:
        # For very large models, KV cache grows roughly with sqrt of params
        return 3.0 * (params_billion / 100) ** 0.5


def parse_args():
    """Parse CLI arguments.

    Parse argumentos da CLI.
    """
    parser = argparse.ArgumentParser(
        description="LLM Local Inference Viability Calculator / "
                    "Calculadora de viabilidade de inferência local de LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples / Exemplos:
  python main.py --context 4096
  python main.py -c 8192 --gpu-type consumer
  python main.py -c 4096 --only-runs --export-json results.json
  python main.py -c 16384 --group-gpu
  python main.py -c 8192 --quantization int4
  python main.py --list-models
  python main.py --model 7 --context 8192
  python main.py -m 70 -c 16384 -q int4
  python main.py -m 70 -c 8192 -q int4 --mode production

Generic model / Modelo genérico:
  python main.py --params-b 405 --context 8192 --quantization int4
  python main.py --params-b 405 --kv-cache 15.0 --context 8192
  python main.py --params-b 405 --model-name "Llama 3.1 405B" -c 8192

Available precisions / Precisões disponíveis:
  fp32 - Float32 (4 bytes/param) - Original precision, highest quality
  fp16 - Float16 (2 bytes/param) - Half VRAM, excellent quality
  int8 - Int8 (1 byte/param) - Quarter VRAM, small quality loss
  int4 - Int4 (0.5 byte/param) - Eighth VRAM, noticeable quality loss

Calculation modes / Modos de cálculo:
  theoretical - Ideal minimum (batch=1, no padding/alignment)
  conservative - Default mode with 10%% buffer (minimal overhead)
  production  - Real-world serving (batch>1, fragmentation) with 25%% buffer
        """,
    )

    parser.add_argument(
        "-c", "--context",
        type=int,
        default=4096,
        help="Context size in tokens / Tamanho do contexto em tokens (default: 4096)",
    )

    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all available models / Listar todos os modelos disponíveis",
    )

    parser.add_argument(
        "-m", "--model",
        type=float,
        metavar="SIZE",
        help="Model size in billions of parameters (e.g., 0.6, 7, 13, 70) / "
             "Tamanho do modelo em bilhões de parâmetros",
    )

    parser.add_argument(
        "--gpu-type",
        choices=["consumer", "datacenter", "all"],
        default="all",
        help="GPU type to consider / Tipo de GPU a considerar (default: all)",
    )

    parser.add_argument(
        "--only-runs",
        action="store_true",
        help="Show only running combinations / Mostrar apenas combinações que rodam",
    )

    parser.add_argument(
        "--group-gpu",
        action="store_true",
        help="Group results by GPU instead of model / "
             "Agrupar resultados por GPU em vez de por modelo",
    )

    parser.add_argument(
        "--summary",
        choices=["model", "gpu", "both", "none"],
        default="both",
        help="Summary type to show / Tipo de resumo a mostrar (default: both)",
    )

    parser.add_argument(
        "--export-csv",
        metavar="FILE",
        help="Export results to CSV / Exportar resultados para CSV",
    )

    parser.add_argument(
        "--export-json",
        metavar="FILE",
        help="Export results to JSON / Exportar resultados para JSON",
    )

    parser.add_argument(
        "-q", "--quantization",
        choices=["fp32", "fp16", "int8", "int4"],
        default="fp16",
        help="Model precision/quantization / Precisão do modelo (fp32, fp16, int8, int4)",
    )

    parser.add_argument(
        "--mode",
        choices=["theoretical", "conservative", "production"],
        default="conservative",
        help="Calculation mode / Modo de cálculo "
             "(theoretical=ideal minimum, conservative=default, production=real-world serving)",
    )

    # Generic model parameters / Parâmetros de modelo genérico
    parser.add_argument(
        "--params-b",
        type=int,
        metavar="BILLIONS",
        help="Generic model: parameters in billions (e.g., 8, 70, 405) / "
             "Modelo genérico: parâmetros em bilhões (ex: 8, 70, 405)",
    )

    parser.add_argument(
        "--kv-cache",
        type=float,
        metavar="MB_PER_TOKEN",
        help="Generic model: KV cache in MB per token FP16 (e.g., 0.6, 1.0, 4.27) / "
             "Modelo genérico: KV cache em MB por token FP16 (ex: 0.6, 1.0, 4.27)",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        metavar="NAME",
        help="Generic model: custom name for display / "
             "Modelo genérico: nome personalizado para exibição",
    )

    # -----------------------------------------------------------------------
    # NEW: Advanced configuration options
    # NOVOS: Opções de configuração avançadas
    # -----------------------------------------------------------------------

    parser.add_argument(
        "--optimize-config",
        action="store_true",
        help="Show optimal layer offload configuration / "
             "Mostrar configuração ótima de offload de camadas",
    )

    parser.add_argument(
        "--cpu-offload",
        action="store_true",
        help="Enable CPU offload calculations / "
             "Habilitar cálculos de offload de CPU",
    )

    parser.add_argument(
        "--system-ram",
        type=float,
        default=32.0,
        metavar="GB",
        help="System RAM available in GB (for CPU offload) / "
             "RAM do sistema disponível em GB (para offload de CPU) (default: 32.0)",
    )

    parser.add_argument(
        "--pcie-gen",
        choices=["3.0", "4.0", "5.0"],
        default="4.0",
        help="PCIe generation for bandwidth estimation / "
             "Geração PCIe para estimativa de largura de banda (default: 4.0)",
    )

    parser.add_argument(
        "--multi-gpu",
        action="store_true",
        help="Enable multi-GPU mode / "
             "Habilitar modo multi-GPU",
    )

    parser.add_argument(
        "--gpu-config",
        type=str,
        metavar="CONFIG",
        help="Multi-GPU configuration (e.g., '2x4090,1x3090') / "
             "Configuração multi-GPU (ex: '2x4090,1x3090')",
    )

    parser.add_argument(
        "--multi-gpu-mode",
        choices=["tensor", "pipeline"],
        default="tensor",
        help="Multi-GPU parallelism mode / "
             "Modo de paralelismo multi-GPU (default: tensor)",
    )

    parser.add_argument(
        "--gguf-file",
        type=str,
        metavar="FILENAME",
        help="GGUF filename to auto-detect quantization / "
             "Nome de arquivo GGUF para auto-detectar quantização",
    )

    parser.add_argument(
        "--format",
        choices=["fp16", "gguf", "exl2", "gptq", "awq"],
        default="fp16",
        help="Model format for overhead calculation / "
             "Formato do modelo para cálculo de overhead (default: fp16)",
    )

    return parser.parse_args()


def main():
    """Main CLI function.

    Função principal da CLI.
    """
    args = parse_args()

    # Handle --list-models
    if args.list_models:
        list_models()
        sys.exit(0)

    # Validate context
    # Validar contexto
    if args.context <= 0:
        print("Error: context_tokens must be positive / Erro: context_tokens deve ser positivo",
              file=sys.stderr)
        sys.exit(1)

    # Map quantization (needed early for GGUF detection)
    # Mapear quantização (necessário cedo para detecção GGUF)
    quant_map = {
        "fp32": Quantization.FP32,
        "fp16": Quantization.FP16,
        "int8": Quantization.INT8,
        "int4": Quantization.INT4,
    }
    quantization = quant_map[args.quantization]

    # Map calculation mode
    # Mapear modo de cálculo
    mode_map = {
        "theoretical": CalculationMode.THEORETICAL,
        "conservative": CalculationMode.CONSERVATIVE,
        "production": CalculationMode.PRODUCTION,
    }
    calculation_mode = mode_map[args.mode]

    # Map model format
    # Mapear formato do modelo
    format_map = {
        "fp16": ModelFormat.FP16,
        "gguf": ModelFormat.GGUF,
        "exl2": ModelFormat.EXL2,
        "gptq": ModelFormat.GPTQ,
        "awq": ModelFormat.AWQ,
    }
    model_format = format_map[args.format]

    # Handle GGUF auto-detection
    # Lidar com auto-detecção GGUF
    if args.gguf_file:
        gguf_info = detect_gguf_quantization(args.gguf_file)
        if gguf_info.is_gguf:
            print(f"\n{Colors.info('GGUF file detected:')} {args.gguf_file}")
            print(f"  Quantization: {gguf_info.quant_name}")
            print(f"  Effective bits: {gguf_info.bits_per_param:.2f} bits/param")
            # Update quantization based on GGUF detection
            if gguf_info.quant_type != "fp16":
                quantization = quant_map[gguf_info.quant_type]
                print(f"  Using quantization: {quantization.value}")
            model_format = ModelFormat.GGUF

    # -----------------------------------------------------------------------
    # Handle advanced configuration modes
    # Lidar com modos de configuração avançados
    # -----------------------------------------------------------------------

    # Multi-GPU mode
    # Modo multi-GPU
    if args.multi_gpu and args.gpu_config:
        try:
            multi_gpus = parse_gpu_config_string(args.gpu_config, get_all_gpus())

            # Get or create model
            model = None
            if args.params_b is not None:
                kv_cache = args.kv_cache if args.kv_cache else estimate_kv_cache(args.params_b)
                model_name = args.model_name if args.model_name else f"Custom Model {args.params_b}B"
                model = LLMModel(
                    name=model_name,
                    params_billion=args.params_b,
                    architecture="decoder-only",
                    precision_default="fp16",
                    kv_cache_mb_per_token=kv_cache,
                    format=model_format,
                )
            elif args.model:
                model = get_model_by_size(args.model)
                if not model:
                    kv_cache = estimate_kv_cache(args.model)
                    model = LLMModel(
                        name=f"Generic Model {args.model}B",
                        params_billion=args.model,
                        architecture="decoder-only",
                        precision_default="fp16",
                        kv_cache_mb_per_token=kv_cache,
                        format=model_format,
                    )

            if not model:
                print("Error: Please specify --model or --params-b with --multi-gpu", file=sys.stderr)
                sys.exit(1)

            # Create multi-GPU config
            mode_map_gpu = {"tensor": MultiGPUMode.TENSOR_PARALLEL, "pipeline": MultiGPUMode.PIPELINE_PARALLEL}
            multi_gpu_mode = mode_map_gpu[args.multi_gpu_mode]
            config = MultiGPUConfig(gpus=multi_gpus, mode=multi_gpu_mode)

            # Calculate
            calc = MultiGPUCalculator(quantization=quantization, model_format=model_format)
            result = calc.calculate(model, config, args.context)

            # Print result
            print_multi_gpu_result(result, model)
            sys.exit(0)

        except ValueError as e:
            print(f"Error parsing GPU config: {e}", file=sys.stderr)
            sys.exit(1)

    # Handle --model (specific model) or --params-b (generic model)
    # Lidar com --model (modelo específico) ou --params-b (modelo genérico)
    model = None
    use_generic = False

    # Generic model takes precedence / Modelo genérico tem precedência
    if args.params_b is not None:
        # Create generic model / Criar modelo genérico
        params_b = args.params_b
        kv_cache = args.kv_cache if args.kv_cache else estimate_kv_cache(params_b)
        model_name = args.model_name if args.model_name else f"Custom Model {params_b}B"

        model = LLMModel(
            name=model_name,
            params_billion=params_b,
            architecture="decoder-only",
            precision_default="fp16",
            kv_cache_mb_per_token=kv_cache,
        )
        use_generic = True

    elif args.model:
        model = get_model_by_size(args.model)
        if not model:
            # Model not found in database - offer to use as generic
            # Modelo não encontrado no banco - oferecer usar como genérico
            print(f"\n{Colors.warn('Model size ' + str(args.model) + 'B not found in database.')}")
            print(f"{Colors.dim('Using generic model with estimated KV cache.')}")
            print(f"Use --params-b {args.model} --kv-cache <value> for custom KV cache.")
            print(f"Or use --list-models to see all available models.\n")

            # Create generic model as fallback / Criar modelo genérico como fallback
            kv_cache = args.kv_cache if args.kv_cache else estimate_kv_cache(args.model)
            model = LLMModel(
                name=f"Generic Model {args.model}B",
                params_billion=args.model,
                architecture="decoder-only",
                precision_default="fp16",
                kv_cache_mb_per_token=kv_cache,
            )
            use_generic = True

    if model:
        # Show VRAM breakdown for the specific model
        print_model_vram_breakdown(model, args.context, quantization, calculation_mode)

        # Determine which GPU to use for advanced calculations
        target_gpu = None
        if args.optimize_config or args.cpu_offload:
            # Use largest GPU or allow user to specify via --gpu flag
            # For now, use the largest available GPU
            gpus = get_all_gpus()
            target_gpu = max(gpus, key=lambda g: g.vram_gb)

            # For CPU offload, get target GPU from user or default to largest
            if args.cpu_offload:
                cpu_calc = CPUOffloadCalculator(
                    quantization=quantization,
                    system_ram_gb=args.system_ram,
                    pcie_generation=args.pcie_gen,
                    model_format=model_format,
                )
                cpu_result = cpu_calc.calculate_offload(model, target_gpu, args.context)
                print_cpu_offload_result(cpu_result, model)
                sys.exit(0)

            # For layer offload optimization
            if args.optimize_config:
                layer_calc = LayerOffloadCalculator(
                    quantization=quantization,
                    model_format=model_format,
                )
                layer_result = layer_calc.calculate_optimal_offload(model, target_gpu, args.context)
                print_layer_offload_result(layer_result, model, target_gpu)

                # Show offload options for all GPUs
                print(f"\n{Colors.CYAN}Offload Options for All GPUs:{Colors.RESET}")
                for gpu in sorted(get_all_gpus(), key=lambda g: g.vram_gb, reverse=True):
                    result = layer_calc.calculate_optimal_offload(model, gpu, args.context)
                    status_color = Colors.ok if result.status == "full_gpu" else Colors.warning
                    print(f"  {gpu.name:<25} ({gpu.vram_gb:3} GB): "
                          f"{result.layers_on_gpu}/{result.total_layers} layers on GPU "
                          f"({status_color(result.status)}){Colors.RESET}")

                print("\n" + "=" * 70)
                sys.exit(0)

        # Show which GPUs can run this model
        gpus = get_all_gpus()
        calculator = VRAMCalculator(quantization=quantization, calculation_mode=calculation_mode)
        results = []
        for gpu in gpus:
            result = calculator.evaluate_pair(model, gpu, args.context)
            results.append(result)

        print(f"\nGPU COMPATIBILITY ({args.context:,} tokens, {quantization.value.upper()}):")
        print("-" * 70)

        runnable = [r for r in results if r.status == Status.RUNS]
        not_runnable = [r for r in results if r.status != Status.RUNS]

        if runnable:
            print(f"\n✓ RUNS on these GPUs:")
            for r in sorted(runnable, key=lambda x: x.gpu_vram_gb):
                free_pct = r.vram_free_percent
                print(f"  {r.gpu_name:<25} ({r.gpu_vram_gb:3} GB) - "
                      f"{free_pct:4.1f}% free")

        if not_runnable:
            print(f"\n✗ DOESN'T RUN - needs more VRAM:")
            sorted_by_need = sorted(not_runnable, key=lambda x: -(x.required_vram_gb - x.gpu_vram_gb))[:5]
            for r in sorted_by_need:
                print(f"  {r.gpu_name:<25} ({r.gpu_vram_gb:3} GB) - "
                      f"needs {r.required_vram_gb:.1f} GB")

            # Suggest layer offload option
            print(f"\n{Colors.info('💡 Tip: Use --optimize-config to see layer offload options')}")
            print(f"{Colors.dim('   Some layers can run on GPU while others use system RAM.')}")

        print("\n" + "=" * 70)
        sys.exit(0)

    # Original flow: show all combinations
    # Fluxo original: mostrar todas as combinações

    # Quantization info
    # Info sobre quantização
    bytes_per_param = quantization.bytes_per_param
    kv_mult = quantization.kv_cache_multiplier
    if args.quantization != "fp16":
        print(
            f"\nℹ️  Using {args.quantization.upper()}: / Usando {args.quantization.upper()}: "
            f"{bytes_per_param} bytes/param, "
            f"KV cache ×{kv_mult}"
        )

    # Select GPUs
    # Selecionar GPUs
    if args.gpu_type == "consumer":
        gpus = get_consumer_gpus()
        gpu_type_label = "Consumer"
    elif args.gpu_type == "datacenter":
        gpus = get_datacenter_gpus()
        gpu_type_label = "Datacenter"
    else:
        gpus = get_all_gpus()
        gpu_type_label = "All / Todas"

    # Select models
    # Selecionar modelos
    models = get_all_models()

    # Calculate
    # Calcular
    calculator = VRAMCalculator(quantization=quantization, calculation_mode=calculation_mode)
    results = []
    for model in models:
        for gpu in gpus:
            result = calculator.evaluate_pair(model, gpu, args.context)
            results.append(result)

    # Header
    print("\n" + "=" * 70)
    print("LLM LOCAL INFERENCE VIABILITY CALCULATOR")
    print("CALCULADORA DE VIABILIDADE DE INFERÊNCIA LOCAL DE LLMs")
    print("=" * 70)
    print(f"\nConfiguration / Configuração:")
    print(f"  • Batch size: 1 (inference)")
    print(f"  • Context: / Contexto: {args.context:,} tokens")
    print(f"  • Quantization: / Quantização: {Colors.bold(args.quantization.upper())}")
    print(f"  • Mode: / Modo: {args.mode}")
    print(f"  • GPUs: {gpu_type_label} ({len(gpus)} models / modelos)")
    print(f"  • LLM Models: / Modelos LLM: {len(models)} sizes / tamanhos")

    # Main table
    # Tabela principal
    print_table(
        results,
        group_by_gpu=args.group_gpu,
        show_only_runs=args.only_runs,
    )

    # Summaries
    # Resumos
    if args.summary in ("model", "both"):
        print_summary_by_model(results)

    if args.summary in ("gpu", "both"):
        print_summary_by_gpu(results)

    # Exports
    # Exportações
    if args.export_csv:
        export_csv(results, args.export_csv)

    if args.export_json:
        export_json(results, args.export_json, args.context, quantization)

    # Warning for 24GB GPUs running near limit
    # Aviso para GPUs de 24GB rodando no limite
    tight_24gb = [
        r for r in results
        if r.status == Status.RUNS
        and r.gpu_vram_gb == 24
        and 22 <= r.required_vram_gb <= 24
    ]
    if tight_24gb:
        print("\n" + "⚠️  " * 12)
        print(f"\n{Colors.BOLD}NOTICE: 24GB GPUs running at the limit:{Colors.RESET}")
        print(f"  {Colors.warning('• Any batching (batch_size > 1) may cause OOM')}")
        print(f"  {Colors.warning('• LoRA adapters add ~0.5-2GB per adapter')}")
        print(f"  {Colors.warning('• Speculative decoding adds ~30-50% memory')}")
        print(f"  {Colors.warning('• Tool calling / function calling adds overhead')}")
        print(f"\n{Colors.DIM}  Assumptions for calculations above:{Colors.RESET}")
        print(f"    Memory model: PyTorch allocator (HF Transformers, vLLM)")
        print(f"    KV cache: FP16 (quantized KV is experimental/exclusive)")
        print(f"    batch_size = 1 (no batching)")
        print(f"    No LoRA adapters active")
        print(f"    No speculative decoding")
        print(f"    No tool calling overhead")
        print(f"  {Colors.DIM}Note: TensorRT-LLM, llama.cpp, EXL2 may have different behavior{Colors.RESET}")
        print()

    print("\n" + "=" * 70)
    print()


if __name__ == "__main__":
    main()
