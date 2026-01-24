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

from models import LLMModel, get_all_models
from gpus import GPU, get_all_gpus, get_consumer_gpus, get_datacenter_gpus
from calculator import VRAMCalculator, Quantization, InferenceResult, Status


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

Available precisions / Precisões disponíveis:
  fp32 - Float32 (4 bytes/param) - Original precision, highest quality
  fp16 - Float16 (2 bytes/param) - Half VRAM, excellent quality
  int8 - Int8 (1 byte/param) - Quarter VRAM, small quality loss
  int4 - Int4 (0.5 byte/param) - Eighth VRAM, noticeable quality loss
        """,
    )

    parser.add_argument(
        "-c", "--context",
        type=int,
        default=4096,
        help="Context size in tokens / Tamanho do contexto em tokens (default: 4096)",
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

    return parser.parse_args()


def main():
    """Main CLI function.

    Função principal da CLI.
    """
    args = parse_args()

    # Validate context
    # Validar contexto
    if args.context <= 0:
        print("Error: context_tokens must be positive / Erro: context_tokens deve ser positivo",
              file=sys.stderr)
        sys.exit(1)

    # Map quantization
    # Mapear quantização
    quant_map = {
        "fp32": Quantization.FP32,
        "fp16": Quantization.FP16,
        "int8": Quantization.INT8,
        "int4": Quantization.INT4,
    }
    quantization = quant_map[args.quantization]

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
    calculator = VRAMCalculator(quantization=quantization)
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
    print(f"  • Context: / Contexto: {args.context:,} tokens")
    print(f"  • Quantization: / Quantização: {args.quantization.upper()}")
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

    print("\n" + "=" * 70)
    print()


if __name__ == "__main__":
    main()
