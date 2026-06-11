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

MTP (multi-token prediction) finding:
- MTP heads add ~180 MB of fixed weights but do NOT multiply KV allocation
- KV slope for MTP models is identical to the base model (same attention architecture)
- The --spec-draft-n-max 3 flag affects generation, not buffer allocation at load time

Dense sweep finding (Qwen3.5-9B-MTP, 18 points byte-precision, ctx 128–262144, 260527):
- VRAM vs context is perfectly linear from ctx=512 onwards (residuals <1%)
- ctx=128 measurement underestimates the true zero-KV baseline by ~370 MB — llama-server
  uses a minimum buffer at tiny contexts that does not scale linearly
- Old slope (128→32768) overestimates by ~31% due to this ctx=128 anomaly
- Improved approach: anchor at ctx=32768 measurement, slope from (32768→max_verified_ctx)

n_parallel finding (Qwen3.5-9B-MTP, np=1 vs np=4, 260527):
- llama-server default n_parallel=4 allocates 3 extra per-slot buffers vs np=1
- The np=4 vs np=1 difference is a CONSTANT +150 MiB across ALL context sizes (±0.5 MiB)
- KV slopes are identical: 34,172 bytes/tok (np=1) vs 34,175 bytes/tok (np=4)
- Production llama-swap uses -np 1; 9B-MTP measurements below are from np=1
- Other models in this catalogue were measured with np=4 (default auto); subtract ~150 MiB
  from their absolute levels for production accuracy, but slopes are unaffected

Backend-specific np overhead (per additional slot beyond np=1), 260529:
- ROCm:   ~50 MB/slot  (derived from 9B-MTP: +150 MiB for np=4 vs np=1 = 150/3)
- Vulkan: ~188 MB/slot (empirically derived: 27B-MTP np=3 at ctx=131,072 measured
  27,720 MB; vs np=1 formula baseline of ~19,350 MB weights + 7,995 MB KV = 27,345 MB;
  overhead = (27,720 - 27,345) / (3-1) ≈ 188 MB/slot)
- Use np_overhead_mb(np, backend) for estimates; add on top of vram_at_ctx()
"""

from __future__ import annotations

from dataclasses import dataclass

BYTES_PER_MIB = 1024**2
BYTES_PER_GIB = 1024**3

VRAM_BUDGET_MB = 32_000  # practical budget (32 GB - OS overhead)

# Per-slot np overhead beyond np=1 (compute buffers scale with parallel slots)
NP_OVERHEAD_MB_PER_SLOT: dict[str, float] = {
    "rocm":   50.0,   # measured: +150 MiB for np=4 vs np=1 on 9B-MTP
    "vulkan": 188.0,  # empirically derived from 27B-MTP np=3 on Atlas
}
BACKEND = "vulkan"  # default backend for Atlas


def np_overhead_mb(np: int, backend: str = BACKEND) -> float:
    """Additional VRAM overhead for np > 1 parallel slots."""
    if np <= 1:
        return 0.0
    return (np - 1) * NP_OVERHEAD_MB_PER_SLOT[backend]


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


# ── Empirically measured models ────────────────────────────────────────────────
# All measurements via measure_llama_server_vram.py on Atlas (AMD AI Pro R9700,
# 32 GB GDDR6). loaded_delta_mb = amd-smi VRAM after server ready minus baseline.
# weights_mb = delta@ctx=128 (true model weight footprint; KV negligible at 128 tokens)
# mmproj_mb = delta@128_with_mmproj - delta@128_no_mmproj (fixed vision encoder cost)
#
# KV slope: when high-ctx verified data (ctx > 32768, no mmproj) is available, slope is
# computed from (delta@32768 → delta@max_verified_ctx). This avoids the ctx=128 anomaly
# where llama-server under-allocates KV buffers (~370 MB below the true linear baseline),
# which inflates the naive (128→32768) slope by ~31%. For models without high-ctx data,
# falls back to (delta@128 → delta@32768) / (32768 − 128).


@dataclass(frozen=True)
class MeasuredModel:
    """Linear VRAM model from llama-server amd-smi measurements.

    Slope is anchored at ctx=32768 when high-ctx verified data is available
    (more accurate than the naive ctx=128 anchor). vram_at_ctx(32768) always
    returns delta_at_32k_mb exactly for models with high-ctx data.
    """

    name: str
    hf_repo: str
    weights_mb: float  # loaded_delta at ctx=128 — true model weight footprint
    delta_at_32k_mb: float  # loaded_delta at ctx=32768
    mmproj_mb: float = 0.0  # additional fixed cost of vision encoder (0 = not measured / N/A)
    mtp: bool = False  # True if model uses MTP draft heads
    note: str = ""
    verified: tuple[tuple[int, float, bool], ...] = ()  # (ctx, measured_mb, with_mmproj)

    def _high_ctx_text_pts(self) -> list[tuple[int, float]]:
        """Non-mmproj verified points above ctx=32768, used for improved slope."""
        return [(c, mb) for c, mb, mm in self.verified if not mm and c > 32768]

    @property
    def kv_mb_per_tok(self) -> float:
        pts = self._high_ctx_text_pts()
        if pts:
            ctx_hi, mb_hi = max(pts, key=lambda x: x[0])
            return (mb_hi - self.delta_at_32k_mb) / (ctx_hi - 32768)
        return (self.delta_at_32k_mb - self.weights_mb) / (32768 - 128)

    def vram_at_ctx(self, ctx: int, with_mmproj: bool = False, np: int = 1, backend: str = BACKEND) -> float:
        """Estimated loaded VRAM in MB at given context size, np, and backend."""
        pts = self._high_ctx_text_pts()
        if pts:
            base_mb = self.delta_at_32k_mb + self.kv_mb_per_tok * (ctx - 32768)
        else:
            base_mb = self.weights_mb + self.kv_mb_per_tok * (ctx - 128)
        return base_mb + (self.mmproj_mb if with_mmproj else 0.0) + np_overhead_mb(np, backend)

    def max_solo_ctx(self, budget_mb: float = VRAM_BUDGET_MB, with_mmproj: bool = False, np: int = 1, backend: str = BACKEND) -> int:
        pts = self._high_ctx_text_pts()
        anchor_ctx = 32768 if pts else 128
        anchor_mb = self.delta_at_32k_mb if pts else self.weights_mb
        overhead = (self.mmproj_mb if with_mmproj else 0.0) + np_overhead_mb(np, backend)
        available = budget_mb - anchor_mb - overhead
        if available <= 0:
            return 0
        raw = int(available / self.kv_mb_per_tok) + anchor_ctx
        # Round down to nearest power-of-2-friendly size
        for snap in range(262144, 0, -4096):
            if snap <= raw:
                return snap
        return raw


# ── Measured model catalogue (260527) ────────────────────────────────────────
# Note: 9B-MTP uses np=1 (production) measurements. All other models used np=4 (auto default)
# and run ~150 MiB higher than production. Slopes are unaffected; subtract ~150 MiB from
# absolute levels of non-9B-MTP models for production-accurate VRAM estimates.

MEASURED = [
    MeasuredModel(
        name="Qwen3.5-0.8B Q4_K_M",
        hf_repo="unsloth/Qwen3.5-0.8B-GGUF",
        weights_mb=979,
        delta_at_32k_mb=1725,
        verified=((131072, 2876, False),),
    ),
    MeasuredModel(
        name="Qwen3.5-2B Q4_K_M",
        hf_repo="unsloth/Qwen3.5-2B-GGUF",
        weights_mb=1694,
        delta_at_32k_mb=2488,
        verified=((131072, 3595, False),),
    ),
    MeasuredModel(
        name="Qwen3.5-4B Q4_K_M",
        hf_repo="unsloth/Qwen3.5-4B-GGUF",
        weights_mb=3217,
        delta_at_32k_mb=4660,
        verified=((131072, 7675, False),),
    ),
    MeasuredModel(
        name="Qwen3.5-9B Q4_K_M",
        hf_repo="unsloth/Qwen3.5-9B-GGUF",
        weights_mb=5482,
        delta_at_32k_mb=6870,
        verified=((131072, 9942, False),),
    ),
    MeasuredModel(
        name="Qwen3.5-9B-MTP Q4_K_M",
        hf_repo="unsloth/Qwen3.5-9B-MTP-GGUF",
        weights_mb=5514,
        delta_at_32k_mb=6903,
        mtp=True,
        note=(
            "MTP heads add ~182 MB fixed vs base 9B (using np=1 measurements). "
            "Byte-precision sweep (18 pts, ctx 128–262144, np=1) confirmed perfectly linear "
            "from ctx=512 (residuals <1%). ctx=128 is ~370 MB below linear baseline. "
            "np=4 vs np=1 = constant +150 MiB; slopes identical (34,172 vs 34,175 bytes/tok)."
        ),
        verified=(
            # Byte-precision sweep 260527, np=1 (production match)
            (512, 5899, False),
            (1024, 5912, False),
            (2048, 5949, False),
            (4096, 6006, False),
            (8192, 6137, False),
            (16384, 6390, False),
            (65536, 8011, False),
            (98304, 8946, False),
            (131072, 9968, False),
            (196608, 12140, False),
            (262144, 14378, False),
        ),
    ),
    MeasuredModel(
        name="Qwen3-VL-2B Q4_K_M (no mmproj)",
        hf_repo="Qwen/Qwen3-VL-2B-Instruct-GGUF",
        weights_mb=1427,
        delta_at_32k_mb=5209,
        mmproj_mb=779,  # 2206 - 1427
        note="Dense attention; KV slope ~5x higher than Qwen3.5-2B",
        verified=(
            (65536, 9572, True),
            (196608, 24212, True),
            (245760, 29731.6, True),  # pass2 sweep 260527
            (262144, 31571.6, True),  # pass3 sweep 260527 — solo max (native ctx)
        ),
    ),
    MeasuredModel(
        name="Qwen3-VL-4B Q4_K_M (no mmproj)",
        hf_repo="Qwen/Qwen3-VL-4B-Instruct-GGUF",
        weights_mb=2762,
        delta_at_32k_mb=7560,
        mmproj_mb=786,  # 3548 - 2762
        verified=(
            (65536, 12954, True),
            (131072, 22294, True),
            (180224, 29349.2, True),  # pass2 sweep 260527
            (200704, 32289.4, True),  # pass3 sweep 260527 — solo max
        ),
    ),
    MeasuredModel(
        name="Qwen3-VL-8B Q4_K_M (no mmproj)",
        hf_repo="Qwen/Qwen3-VL-8B-Instruct-GGUF",
        weights_mb=4848,
        delta_at_32k_mb=9647,
        mmproj_mb=1120,  # 5968 - 4848
        verified=(
            (65536, 15375, True),
            (131072, 24727, True),
            (163840, 29431.2, True),  # pass2 sweep 260527
            (184320, 32370.4, True),  # pass3 sweep 260527 — solo max
        ),
    ),
    MeasuredModel(
        name="Qwen3.6-27B Q4_K_M (no mmproj)",
        hf_repo="unsloth/Qwen3.6-27B-GGUF",
        weights_mb=16395,
        delta_at_32k_mb=18787,
        note="Hybrid attention; KV slope ~0.063 MB/tok (improved: from 32768→229376 anchor)",
        verified=(
            (131072, 24930, False),
            (196608, 29148, False),
            (229376, 30839.4, False),  # pass2/pass3 sweep 260527 — solo max (253952 @ 32444 MB is over budget)
        ),
    ),
    MeasuredModel(
        name="Qwen3.6-27B-MTP Q4_K_M (no mmproj)",
        hf_repo="unsloth/Qwen3.6-27B-MTP-GGUF",
        weights_mb=16672,
        delta_at_32k_mb=19065,
        mtp=True,
        note="MTP heads add ~277 MB fixed; KV slope identical to base 27B",
        verified=(
            (131072, 25208, False),
            (196608, 29423, False),
            (212992, 32343.2, False),  # pass3 sweep 260527 — solo max
        ),
    ),
    MeasuredModel(
        name="Gemma-4-31B Q4_K_M (no mmproj)",
        hf_repo="unsloth/gemma-4-31B-it-GGUF",
        weights_mb=18121,
        delta_at_32k_mb=24491,
        note="Dense attention; KV slope ~0.053 MB/tok (improved: from 32768→163840 anchor); 172032 OOM",
        verified=(
            (65536, 26994, False),
            (131072, 29708.6, False),  # pass1 sweep 260527
            (147456, 30986.3, False),  # pass2 sweep 260527
            (163840, 32309, False),    # direct test 260527 — solo max
        ),
    ),
    MeasuredModel(
        name="DeepSeek-R1-Distill-32B Q4_K_M",
        hf_repo="bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF:Q4_K_M",
        weights_mb=18956,
        delta_at_32k_mb=27383.9,
        note="Dense attention (Qwen-32B base); high KV slope ~0.258 MB/tok limits max ctx to ~49k",
        verified=(
            (49152, 31502.9, False),  # measure_new_models.py pass2 260527 — solo max
        ),
    ),
    MeasuredModel(
        name="Hy-MT2-7B Q4_K_M",
        hf_repo="tencent/Hy-MT2-7B-GGUF:Q4_K_M",
        weights_mb=4780,
        delta_at_32k_mb=9035.1,
        note="Translation model; dense attention; KV slope ~0.127 MB/tok",
        verified=(
            (208896, 31437.2, False),  # measure_new_models.py pass2 260527 — solo max
        ),
    ),
    # GLM-OCR: fixed ctx=12000 (upstream limitation); mmproj auto-detected; flash-attn must be OFF
    # weights_mb ≈ 2708 at ctx=12000 (KV negligible at 12k relative to model size)
    # Not added as MeasuredModel — fixed-ctx model doesn't benefit from slope extrapolation
]

# Index by short key for co-load calculations
_M = {m.hf_repo: m for m in MEASURED}


# ── Legacy Qwen hybrid dataclass models (for llama-bench cross-checks) ────────

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


def print_measured_summary() -> None:
    """Print solo VRAM usage and max context for all empirically measured models."""
    print("=" * 90)
    print("MEASURED MODELS — Solo VRAM (Atlas: AMD AI Pro R9700, 32 GB)")
    print("=" * 90)
    print(f"{'Model':<40} {'weights':>8} {'kv/tok':>8} {'@32k':>8} {'@131k':>8} {'max_ctx':>10}")
    print("-" * 90)
    for m in MEASURED:
        v_32k = m.vram_at_ctx(32768)
        v_131k = m.vram_at_ctx(131072)
        max_ctx = m.max_solo_ctx()
        print(
            f"{m.name:<40} {m.weights_mb:>7.0f}  {m.kv_mb_per_tok:>7.4f}  {v_32k:>7.0f}  {v_131k:>7.0f}  {max_ctx:>10,}"
        )
    print()


def print_coload_analysis() -> None:
    """Print co-load feasibility for candidate pairs on 32 GB."""
    print("=" * 90)
    print("CO-LOAD ANALYSIS — Candidate pairs (budget = 32,000 MB)")
    print("=" * 90)

    pairs: list[tuple[str, int, bool, str, int, bool]] = [
        # (repo_a, ctx_a, mmproj_a, repo_b, ctx_b, mmproj_b)
        # Current matrix pair: mtp27 & vl8
        ("unsloth/Qwen3.6-27B-MTP-GGUF", 65_536, False, "Qwen/Qwen3-VL-8B-Instruct-GGUF", 16_384, True),
        ("unsloth/Qwen3.6-27B-MTP-GGUF", 65_536, False, "Qwen/Qwen3-VL-8B-Instruct-GGUF", 32_768, True),
        # New candidate: mtp9 & vl8
        ("unsloth/Qwen3.5-9B-MTP-GGUF", 262_144, False, "Qwen/Qwen3-VL-8B-Instruct-GGUF", 16_384, True),
        ("unsloth/Qwen3.5-9B-MTP-GGUF", 262_144, False, "Qwen/Qwen3-VL-8B-Instruct-GGUF", 32_768, True),
        # mtp27 & vl4
        ("unsloth/Qwen3.6-27B-MTP-GGUF", 65_536, False, "Qwen/Qwen3-VL-4B-Instruct-GGUF", 32_768, True),
        # mtp27 & vl2
        ("unsloth/Qwen3.6-27B-MTP-GGUF", 65_536, False, "Qwen/Qwen3-VL-2B-Instruct-GGUF", 32_768, True),
    ]

    for repo_a, ctx_a, mm_a, repo_b, ctx_b, mm_b in pairs:
        ma = _M[repo_a]
        mb = _M[repo_b]
        vram_a = ma.vram_at_ctx(ctx_a, with_mmproj=mm_a)
        vram_b = mb.vram_at_ctx(ctx_b, with_mmproj=mm_b)
        total = vram_a + vram_b
        headroom = VRAM_BUDGET_MB - total
        fits = "✓" if headroom >= 0 else "✗"
        mm_a_str = "+mmproj" if mm_a else "       "
        mm_b_str = "+mmproj" if mm_b else "       "
        print(f"  {fits}  {ma.name.split()[0]:<25} ctx={ctx_a // 1024:>4}k {mm_a_str}  {vram_a:>7.0f} MB")
        print(f"     + {mb.name.split()[0]:<25} ctx={ctx_b // 1024:>4}k {mm_b_str}  {vram_b:>7.0f} MB")
        print(f"       total {total:>7.0f} MB  headroom {headroom:>+7.0f} MB")
        print()


def main() -> None:
    print_measured_summary()
    print_coload_analysis()

    print("=" * 90)
    print("LEGACY QWEN HYBRID MODELS — llama-bench KV formula check")
    print("=" * 90)
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
