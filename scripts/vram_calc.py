"""Simple CLI VRAM calculator for Atlas (AMD AI Pro R9700, 32 GB GDDR6).

Uses the same measured model data as atlas-vram-calc.html.

Modes:
  Given ctx  → reports VRAM usage
  Given vram → reports max context

Examples:
  uv run scripts/vram_calc.py q27b-mtp --ctx 196608 --np 3
  uv run scripts/vram_calc.py q27b-mtp --vram 31832
  uv run scripts/vram_calc.py vl8b --ctx 32768 --mmproj
  uv run scripts/vram_calc.py --list
"""

from __future__ import annotations

import argparse

# ── Model catalogue (mirrors atlas-vram-calc.html) ───────────────────────────
# w = weights MB, slope = MB/tok (anchored at ctx=32768 for high-ctx models,
# or ctx=128 for models without high-ctx verified data)

MODELS: dict[str, dict] = {
    # Text
    "q0.8b":     {"name": "Qwen3.5-0.8B",       "quant": "Q4_K_M", "w": 979,   "slope": 0.0117, "max": 262144},
    "q2b":       {"name": "Qwen3.5-2B",          "quant": "Q4_K_M", "w": 1694,  "slope": 0.0113, "max": 262144},
    "q4b":       {"name": "Qwen3.5-4B",          "quant": "Q4_K_M", "w": 3217,  "slope": 0.0307, "max": 262144},
    "q9b":       {"name": "Qwen3.5-9B",          "quant": "Q4_K_M", "w": 5482,  "slope": 0.0312, "max": 262144},
    "q9b-mtp":   {"name": "Qwen3.5-9B-MTP",      "quant": "Q4_K_M", "w": 5514,  "slope": 0.0312, "max": 262144,
                  # MTP: draft context buffers at np=1 derived from measurement vs non-MTP baseline.
                  # mtp_vulkan_np_overhead_per_slot: not yet measured for np>1 on Vulkan; use estimate.
                  "mtp": True, "mtp_base_mb": 250, "mtp_per_slot_mb": 55,
                  "mtp_vulkan_np_overhead_per_slot": 350},  # rough estimate — measure to calibrate
    "q27b":      {"name": "Qwen3.6-27B",         "quant": "Q4_K_M", "w": 16395, "slope": 0.0613, "max": 229376},
    "q27b-mtp":  {"name": "Qwen3.6-27B-MTP",     "quant": "Q4_K_M", "w": 16672, "slope": 0.0613, "max": 212992,
                  # MTP np=1: measured delta over non-MTP baseline at ctx=196k (ROCm): 976 MB
                  # MTP np>1 Vulkan: combined overhead calibrated from two Atlas measurements:
                  #   155,648 ctx np=3 → 29,345 MB measured → 1,078 MB/slot above np=1 formula
                  #   196,608 ctx np=3 → 31,671 MB measured → 986 MB/slot above np=1 formula
                  #   average: ~1,030 MB/slot (replaces general np_overhead for Vulkan+MTP)
                  "mtp": True, "mtp_base_mb": 976, "mtp_per_slot_mb": 149,
                  "mtp_vulkan_np_overhead_per_slot": 1030},
    # Vision
    "vl2b":      {"name": "Qwen3-VL-2B",         "quant": "Q4_K_M", "w": 2206,  "slope": 0.1159, "max": 262144, "mmproj": 1180},
    "vl4b":      {"name": "Qwen3-VL-4B",         "quant": "Q4_K_M", "w": 3548,  "slope": 0.1470, "max": 200704, "mmproj": 1180},
    "vl8b":      {"name": "Qwen3-VL-8B",         "quant": "Q4_K_M", "w": 5968,  "slope": 0.1470, "max": 184320, "mmproj": 1180},
    # Specialty
    "gemma31b":  {"name": "Gemma-4-31B",         "quant": "Q4_K_M", "w": 18121, "slope": 0.0596, "max": 163840},
    "r1-32b":    {"name": "DeepSeek-R1-32B",     "quant": "Q4_K_M", "w": 18956, "slope": 0.2514, "max": 49152},
    "hymt2":     {"name": "Hy-MT2-7B",           "quant": "Q4_K_M", "w": 4780,  "slope": 0.1272, "max": 208896},
    # Voice (fixed size, no KV)
    "whisper":   {"name": "Whisper STT",         "quant": "small.en", "w": 726,  "slope": 0.0,    "max": 0, "fixed": True},
    "kokoro":    {"name": "Kokoro TTS",          "quant": "CPU ONNX", "w": 0,    "slope": 0.0,    "max": 0, "fixed": True},
}

# Per-slot np overhead (compute buffers), measured on Atlas
NP_OVERHEAD_MB_PER_SLOT = {"rocm": 50.0, "vulkan": 188.0}

TOTAL_VRAM_MB = 32624
BUDGET_MB = 32400  # practical budget


def np_overhead(np: int, backend: str) -> float:
    return (np - 1) * NP_OVERHEAD_MB_PER_SLOT[backend] if np > 1 else 0.0


def mtp_overhead(m: dict, np: int, backend: str) -> float:
    """MTP draft context buffer overhead: base (np=1) + extra per additional slot.

    For Vulkan+MTP with np>1, uses empirically calibrated combined overhead
    (replaces general np_overhead — do not double-count).
    """
    if not m.get("mtp"):
        return 0.0
    base = m.get("mtp_base_mb", 0.0)
    if np <= 1:
        return base
    if backend == "vulkan" and "mtp_vulkan_np_overhead_per_slot" in m:
        return base + (np - 1) * m["mtp_vulkan_np_overhead_per_slot"]
    return base + (np - 1) * m.get("mtp_per_slot_mb", 0.0)


def uses_vulkan_mtp_np_overhead(m: dict, np: int, backend: str) -> bool:
    """True when the Vulkan+MTP combined overhead is used (general np_overhead skipped)."""
    return m.get("mtp", False) and np > 1 and backend == "vulkan" and "mtp_vulkan_np_overhead_per_slot" in m


def vram_at_ctx(model_id: str, ctx: int, mmproj: bool, np: int, backend: str) -> float:
    m = MODELS[model_id]
    if m.get("fixed"):
        return float(m["w"])
    kv = m["slope"] * max(0, ctx - 128)
    mp = m.get("mmproj", 0) if mmproj else 0.0
    # For Vulkan+MTP, the combined overhead replaces the general np_overhead
    extra_np = 0.0 if uses_vulkan_mtp_np_overhead(m, np, backend) else np_overhead(np, backend)
    return m["w"] + kv + mp + extra_np + mtp_overhead(m, np, backend)


def max_ctx_for_budget(model_id: str, budget: float, mmproj: bool, np: int, backend: str) -> int:
    m = MODELS[model_id]
    if m.get("fixed") or m["slope"] == 0:
        return 0
    mp = m.get("mmproj", 0) if mmproj else 0.0
    extra_np = 0.0 if uses_vulkan_mtp_np_overhead(m, np, backend) else np_overhead(np, backend)
    fixed = m["w"] + mp + extra_np + mtp_overhead(m, np, backend) + 128 * m["slope"]
    available = budget - fixed
    if available <= 0:
        return 0
    raw = int(available / m["slope"]) + 128
    raw = min(raw, m["max"] if m["max"] else raw)
    return (raw // 4096) * 4096


def fmt_mb(mb: float) -> str:
    if mb >= 1000:
        return f"{mb/1024:.1f} GB ({mb:,.0f} MB)"
    return f"{mb:,.0f} MB"


def fmt_ctx(ctx: int) -> str:
    if ctx >= 1024:
        return f"{ctx//1024}k ({ctx:,} tokens)"
    return str(ctx)


def main() -> None:
    parser = argparse.ArgumentParser(description="Atlas VRAM calculator")
    parser.add_argument("model", nargs="?", help="Model ID (use --list to see all)")
    parser.add_argument("--ctx", type=int, help="Context size → output VRAM usage")
    parser.add_argument("--vram", type=float, help="Available VRAM in MB → output max context")
    parser.add_argument("--np", type=int, default=1, help="n-parallel slots (default 1)")
    parser.add_argument("--mmproj", action="store_true", help="Include vision encoder (VL models)")
    parser.add_argument("--backend", choices=["vulkan", "rocm"], default="vulkan",
                        help="Inference backend (default: vulkan)")
    parser.add_argument("--budget", type=float, default=BUDGET_MB,
                        help=f"Total VRAM budget in MB (default: {BUDGET_MB})")
    parser.add_argument("--list", action="store_true", help="List all models")
    args = parser.parse_args()

    if args.list:
        print(f"{'ID':<12} {'Name':<22} {'Quant':<14} {'Weights':>9}  {'Slope':>8}  {'Max ctx':>8}")
        print("-" * 80)
        for mid, m in MODELS.items():
            flags = ""
            if m.get("mtp"):   flags += " MTP"
            if m.get("fixed"): flags += " fixed"
            print(f"{mid:<12} {m['name']:<22} {m['quant']:<14} {m['w']:>8.0f}MB"
                  f"  {m['slope']:>7.4f}/tok  {fmt_ctx(m['max'])}{flags}")
        return

    if not args.model:
        parser.print_help()
        return

    if args.model not in MODELS:
        print(f"Unknown model '{args.model}'. Use --list to see available models.")
        return

    m = MODELS[args.model]
    ovhd = np_overhead(args.np, args.backend)
    mp = m.get("mmproj", 0) if args.mmproj else 0.0

    mtp_ovhd = mtp_overhead(m, args.np, args.backend)

    print(f"\nModel:    {m['name']} ({m['quant']})")
    print(f"Backend:  {args.backend}  |  np={args.np}")
    print(f"Weights:  {m['w']:,.0f} MB")
    if args.mmproj and mp:
        print(f"mmproj:   +{mp:.0f} MB")
    if mtp_ovhd:
        if uses_vulkan_mtp_np_overhead(m, args.np, args.backend):
            extra = (args.np - 1) * m.get("mtp_vulkan_np_overhead_per_slot", 0)
            print(f"MTP ctx:  +{mtp_ovhd:.0f} MB  ({m.get('mtp_base_mb',0):.0f} MB base + {extra:.0f} MB Vulkan+MTP np overhead [{args.np-1}×{m.get('mtp_vulkan_np_overhead_per_slot',0):.0f}])")
        else:
            extra = max(0, args.np - 1) * m.get("mtp_per_slot_mb", 0)
            print(f"MTP ctx:  +{mtp_ovhd:.0f} MB  ({m.get('mtp_base_mb',0):.0f} MB base + {extra:.0f} MB extra slots)")
    if ovhd and not uses_vulkan_mtp_np_overhead(m, args.np, args.backend):
        print(f"np ovhd:  +{ovhd:.0f} MB  ({args.np-1} extra slot{'s' if args.np>2 else ''} × {NP_OVERHEAD_MB_PER_SLOT[args.backend]:.0f} MB)")
    print()

    if args.ctx is not None:
        total = vram_at_ctx(args.model, args.ctx, args.mmproj, args.np, args.backend)
        kv = m["slope"] * max(0, args.ctx - 128)
        headroom = args.budget - total
        per_slot = args.ctx // args.np if args.np > 1 else args.ctx

        print(f"Context:  {fmt_ctx(args.ctx)}" + (f"  ({fmt_ctx(per_slot)}/slot)" if args.np > 1 else ""))
        print(f"KV cache: {kv:,.0f} MB")
        print(f"Total:    {fmt_mb(total)}")
        print(f"Headroom: {fmt_mb(headroom)}" + (" ⚠ OVER BUDGET" if headroom < 0 else
                                                   " ⚠ tight" if headroom < 1000 else " ✓"))

    elif args.vram is not None:
        ctx = max_ctx_for_budget(args.model, args.vram, args.mmproj, args.np, args.backend)
        if ctx == 0:
            print(f"Model does not fit in {args.vram:,.0f} MB (weights alone: {m['w'] + mp + ovhd:,.0f} MB)")
        else:
            total = vram_at_ctx(args.model, ctx, args.mmproj, args.np, args.backend)
            per_slot = ctx // args.np if args.np > 1 else ctx
            print(f"Budget:   {fmt_mb(args.vram)}")
            print(f"Max ctx:  {fmt_ctx(ctx)}" + (f"  ({fmt_ctx(per_slot)}/slot)" if args.np > 1 else ""))
            print(f"VRAM at max ctx: {fmt_mb(total)}")
    else:
        # No ctx or vram given — show solo max
        ctx = max_ctx_for_budget(args.model, args.budget, args.mmproj, args.np, args.backend)
        total = vram_at_ctx(args.model, ctx, args.mmproj, args.np, args.backend) if ctx else m["w"] + mp + ovhd
        per_slot = ctx // args.np if args.np > 1 else ctx

        print(f"Budget:   {fmt_mb(args.budget)}")
        if ctx:
            print(f"Max ctx:  {fmt_ctx(ctx)}" + (f"  ({fmt_ctx(per_slot)}/slot)" if args.np > 1 else ""))
            print(f"VRAM:     {fmt_mb(total)}")
        else:
            print(f"Does not fit (weights: {fmt_mb(m['w'] + mp + ovhd)})")


if __name__ == "__main__":
    main()
