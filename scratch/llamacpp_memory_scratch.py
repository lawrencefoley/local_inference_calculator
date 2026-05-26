"""Scratch llama.cpp memory calculator for iterating on formulas.

Hardcoded and exploratory. Goal: match llama.cpp/llama-bench/amd-smi logs
first, then move the useful formula into llmfit.

Key llama.cpp finding for Qwen3.5/Qwen3.6:
- `full_attention_interval = 4`
- only every 4th layer allocates per-token KV
- other layers allocate recurrent state (RS), mostly fixed with context

Important distinction:
- server idle/load memory = model + allocated KV + RS + compute/misc
- llama-server allocates the needed buffers at load time; amd-smi stays essentially flat during small inference requests
- llama-bench can show much higher transient prompt-processing peaks and is not the right proxy for steady llama-server VRAM
"""

from __future__ import annotations

from dataclasses import dataclass

BYTES_PER_MIB = 1024**2
BYTES_PER_GIB = 1024**3


@dataclass(frozen=True)
class Observation:
    ctx: int
    kv_mib: float
    rs_mib: float
    compute_mib: float


@dataclass(frozen=True)
class AmdSmiObservation:
    ctx: int
    server_idle_delta_mb: float | None = None
    server_inference_peak_delta_mb: float | None = None
    llama_bench_peak_delta_mb: float | None = None


@dataclass(frozen=True)
class QwenHybridGgufModel:
    name: str
    model_size_bytes: int
    params: int
    total_layers: int
    full_attention_interval: int
    kv_heads: int
    key_length: int
    value_length: int
    ssm_inner_size: int
    ssm_state_size: int
    ssm_time_step_rank: int
    native_context: int
    observed_gpu_model_mib: float
    observed_cpu_mapped_mib: float
    observations: tuple[Observation, ...]
    amd_smi: tuple[AmdSmiObservation, ...]

    @property
    def full_attention_layers(self) -> int:
        return self.total_layers // self.full_attention_interval

    @property
    def recurrent_layers(self) -> int:
        return self.total_layers - self.full_attention_layers

    @property
    def baseline_observation(self) -> Observation:
        return self.observations[0]


@dataclass(frozen=True)
class LlamaCppRun:
    gpu_vram_gib: float
    context_tokens: int
    kv_bytes: int
    no_mmproj: bool
    observed_vram_fraction: float | None = None


MODELS = [
    QwenHybridGgufModel(
        name="Qwen3.5-0.8B Q4_K_M",
        model_size_bytes=696_328_192,
        params=821_839_104,
        total_layers=24,
        full_attention_interval=4,
        kv_heads=2,
        key_length=256,
        value_length=256,
        ssm_inner_size=2048,
        ssm_state_size=128,
        ssm_time_step_rank=16,
        native_context=262_144,
        observed_gpu_model_mib=497.40,
        observed_cpu_mapped_mib=198.93,
        observations=(
            Observation(ctx=512, kv_mib=6.00, rs_mib=19.27, compute_mib=489.00),
            Observation(ctx=4096, kv_mib=48.00, rs_mib=19.27, compute_mib=497.00),
            Observation(ctx=32768, kv_mib=384.00, rs_mib=19.27, compute_mib=594.01),
        ),
        amd_smi=(AmdSmiObservation(ctx=32768, server_idle_delta_mb=1724, llama_bench_peak_delta_mb=5592),),
    ),
    QwenHybridGgufModel(
        name="Qwen3.5-9B Q4_K_M",
        model_size_bytes=5_669_554_176,
        params=8_953_803_264,
        total_layers=32,
        full_attention_interval=4,
        kv_heads=4,
        key_length=256,
        value_length=256,
        ssm_inner_size=4096,
        ssm_state_size=128,
        ssm_time_step_rank=32,
        native_context=262_144,
        observed_gpu_model_mib=4861.28,
        observed_cpu_mapped_mib=545.62,
        observations=(
            Observation(ctx=512, kv_mib=16.00, rs_mib=50.25, compute_mib=506.01),
            Observation(ctx=4096, kv_mib=128.00, rs_mib=50.25, compute_mib=501.00),
            Observation(ctx=32768, kv_mib=1024.00, rs_mib=50.25, compute_mib=1132.01),
        ),
        amd_smi=(
            AmdSmiObservation(
                ctx=32768,
                server_idle_delta_mb=6868,
                server_inference_peak_delta_mb=6869,
                llama_bench_peak_delta_mb=14884,
            ),
        ),
    ),
    QwenHybridGgufModel(
        name="Qwen3.6-27B Q4_K_M",
        model_size_bytes=16_806_250_496,
        params=26_895_998_464,
        total_layers=64,
        full_attention_interval=4,
        kv_heads=4,
        key_length=256,
        value_length=256,
        ssm_inner_size=6144,
        ssm_state_size=128,
        ssm_time_step_rank=48,
        native_context=262_144,
        observed_gpu_model_mib=15345.66,
        observed_cpu_mapped_mib=682.03,
        observations=(
            Observation(ctx=512, kv_mib=32.00, rs_mib=149.62, compute_mib=510.01),
            Observation(ctx=4096, kv_mib=256.00, rs_mib=149.62, compute_mib=505.00),
            Observation(ctx=32768, kv_mib=2048.00, rs_mib=149.62, compute_mib=1660.01),
        ),
        amd_smi=(
            AmdSmiObservation(ctx=32768, server_idle_delta_mb=18794, llama_bench_peak_delta_mb=30740),
            AmdSmiObservation(ctx=212_992, server_idle_delta_mb=30218, server_inference_peak_delta_mb=30219),
        ),
    ),
]

RUN = LlamaCppRun(
    gpu_vram_gib=32.0,
    context_tokens=212_992,
    kv_bytes=2,  # f16 KV
    no_mmproj=True,
    observed_vram_fraction=0.93,
)

# Calibrated to server-idle amd-smi observations. This is *not* prompt eval peak.
SERVER_MISC_OVERHEAD_MIB = 512.0
MM_PROJ_MIB = 0.0 if RUN.no_mmproj else 1024.0


def mib(bytes_value: float) -> float:
    return bytes_value / BYTES_PER_MIB


def gib_from_mib(mib_value: float) -> float:
    return mib_value / 1024


def kv_mib_per_token(model: QwenHybridGgufModel, kv_bytes: int) -> float:
    """llama.cpp Qwen3.5/Qwen3.6 KV cache for full-attention layers only."""
    bytes_per_token = model.full_attention_layers * model.kv_heads * (model.key_length + model.value_length) * kv_bytes
    return mib(bytes_per_token)


def recurrent_state_s_only_mib(model: QwenHybridGgufModel) -> float:
    return mib(model.recurrent_layers * model.ssm_inner_size * model.ssm_state_size * 4)


def server_idle_total_mib(model: QwenHybridGgufModel, run: LlamaCppRun) -> float:
    kv_mib = kv_mib_per_token(model, run.kv_bytes) * run.context_tokens
    return (
        model.observed_gpu_model_mib
        + kv_mib
        + model.baseline_observation.rs_mib
        + model.baseline_observation.compute_mib
        + SERVER_MISC_OVERHEAD_MIB
        + MM_PROJ_MIB
    )


def max_server_idle_context(model: QwenHybridGgufModel, run: LlamaCppRun) -> int:
    fixed_mib = (
        model.observed_gpu_model_mib
        + model.baseline_observation.rs_mib
        + model.baseline_observation.compute_mib
        + SERVER_MISC_OVERHEAD_MIB
        + MM_PROJ_MIB
    )
    available_for_kv_mib = run.gpu_vram_gib * 1024 - fixed_mib
    return int(available_for_kv_mib / kv_mib_per_token(model, run.kv_bytes))


def print_observation_check(model: QwenHybridGgufModel, run: LlamaCppRun) -> None:
    kv_per_token = kv_mib_per_token(model, run.kv_bytes)
    for obs in model.observations:
        predicted_kv = kv_per_token * obs.ctx
        kv_delta = predicted_kv - obs.kv_mib
        print(
            f"  llama-bench ctx {obs.ctx:>7,}: KV predicted {predicted_kv:>8.2f} MiB, "
            f"actual {obs.kv_mib:>8.2f}, delta {kv_delta:>7.2f}; "
            f"RS {obs.rs_mib:>7.2f}; compute {obs.compute_mib:>7.2f}"
        )


def print_amd_smi_check(model: QwenHybridGgufModel, run: LlamaCppRun) -> None:
    for obs in model.amd_smi:
        estimate = server_idle_total_mib(model, LlamaCppRun(run.gpu_vram_gib, obs.ctx, run.kv_bytes, run.no_mmproj))
        if obs.server_idle_delta_mb is not None:
            print(
                f"  amd-smi server ctx {obs.ctx:>7,}: estimated idle {estimate:>8.0f} MiB, "
                f"observed delta {obs.server_idle_delta_mb:>8.0f} MB, "
                f"diff {estimate - obs.server_idle_delta_mb:>7.0f}"
            )
        if obs.server_inference_peak_delta_mb is not None:
            print(
                f"  amd-smi infer  ctx {obs.ctx:>7,}: server inference peak delta "
                f"{obs.server_inference_peak_delta_mb:>8.0f} MB, "
                f"diff from idle {obs.server_inference_peak_delta_mb - (obs.server_idle_delta_mb or 0):>5.0f}"
            )
        if obs.llama_bench_peak_delta_mb is not None:
            print(
                f"  amd-smi bench  ctx {obs.ctx:>7,}: llama-bench transient peak delta "
                f"{obs.llama_bench_peak_delta_mb:>8.0f} MB"
            )


def main() -> None:
    print(f"VRAM budget: {RUN.gpu_vram_gib:.2f} GiB")
    print(f"Target context: {RUN.context_tokens:,} tokens")
    if RUN.observed_vram_fraction is not None:
        print(
            f"Observed VRAM usage: {RUN.observed_vram_fraction:.1%} "
            f"({RUN.gpu_vram_gib * RUN.observed_vram_fraction:.2f} GiB)"
        )
    print()

    for model in MODELS:
        kv_per_token = kv_mib_per_token(model, RUN.kv_bytes)
        kv_at_target = kv_per_token * RUN.context_tokens
        estimated = server_idle_total_mib(model, RUN)
        rs_s_only = recurrent_state_s_only_mib(model)
        model_gib = gib_from_mib(model.observed_gpu_model_mib)
        file_gib = model.model_size_bytes / BYTES_PER_GIB

        print(model.name)
        print("-" * len(model.name))
        print(
            f"  layers: {model.total_layers} total, {model.full_attention_layers} full-attn, {model.recurrent_layers} recurrent"
        )
        print(f"  file size: {file_gib:.2f} GiB; GPU model buffer: {model_gib:.2f} GiB")
        print(
            f"  KV formula: {model.full_attention_layers} * {model.kv_heads} * ({model.key_length}+{model.value_length}) * {RUN.kv_bytes} bytes"
        )
        print(f"  KV: {kv_per_token:.8f} MiB/token")
        print_observation_check(model, RUN)
        print_amd_smi_check(model, RUN)
        print(
            f"  RS observed baseline: {model.baseline_observation.rs_mib:.2f} MiB; S-only formula: {rs_s_only:.2f} MiB"
        )
        print(f"  KV @ {RUN.context_tokens:,}: {gib_from_mib(kv_at_target):.2f} GiB")
        print(f"  estimated server idle @ target: {gib_from_mib(estimated):.2f} GiB")
        print(f"  max server-idle context with current knobs: {max_server_idle_context(model, RUN):,} tokens")
        print()


if __name__ == "__main__":
    main()
