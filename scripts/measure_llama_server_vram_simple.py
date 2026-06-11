#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["click", "rich"]
# ///

"""Measure llama-server VRAM usage via the llama-swap Prometheus metrics endpoint.

Starts a llama-server container on atlas via ``docker run``, polls
http://atlas.lan:8078/metrics for ``llamaswap_gpu_memory_used_bytes``, and
reports baseline, loaded, and inference-peak VRAM.

Usage:
    ./measure_llama_server_vram_simple.py "-hf unsloth/Qwen3-8B-GGUF -c 32768"
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import threading
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

import click
from rich.console import Console
from rich.table import Table

METRICS_URL = "http://atlas.lan:8078/metrics"
METRIC_NAME = "llamaswap_gpu_memory_used_bytes"
SSH_HOST = "atlas"
DEFAULT_IMAGE = "ghcr.io/ggml-org/llama.cpp:full-vulkan"  # same as llama-bench-vulkan
HF_CACHE_SRC = "/mnt/cache1/huggingface"  # shared cache used by llama-swap
CONTAINER_PORT = 8080

MiB = 1024 * 1024


# ── helpers ──────────────────────────────────────────────────────────────


def run_ssh(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a single command on the remote host via SSH."""
    return subprocess.run(["ssh", SSH_HOST, *cmd], text=True, capture_output=True)


def stop_container(container_name: str) -> None:
    """Remove any leftover container with the given name."""
    result = run_ssh(["docker", "rm", "-f", container_name])
    if result.returncode == 0 and result.stdout.strip():
        click.echo(f"Stopped previous container: {result.stdout.strip()}")
    time.sleep(2)


def start_container(
    container_name: str,
    image: str,
    host_port: int,
    suffix: str,
    hf_cache: bool,
) -> None:
    """Start a llama-server container on the remote host."""
    cmd: list[str] = [
        "docker", "run", "-d", "--rm",
        "--name", container_name,
        "-p", f"{host_port}:{CONTAINER_PORT}",
        # AMD GPU (Vulkan) — same device mounts as llama-bench-vulkan
        "--device", "/dev/dri:/dev/dri",
        "--group-add", "44",  # video GID
        "--group-add", "993",  # render GID
    ]
    if hf_cache:
        cmd.extend(["-v", f"{HF_CACHE_SRC}:/root/.cache/huggingface:ro"])

    cmd.extend(["--entrypoint", "/app/llama-server", image])

    # Static args
    cmd.extend(["-ngl", "99", "--host", "0.0.0.0", "--port", str(CONTAINER_PORT)])

    # User-supplied suffix (model + extra flags)
    cmd.extend(suffix.split())

    click.echo(f"Starting container…  {' '.join(cmd)}")
    result = run_ssh(cmd)
    if result.returncode != 0:
        click.echo(f"docker run failed:\n{result.stderr}", err=True)
        sys.exit(result.returncode)
    click.echo(f"Container started: {result.stdout.strip()}")


# ── metrics ──────────────────────────────────────────────────────────────


def get_metric() -> int | None:
    """Fetch GPU memory used from the Prometheus metrics endpoint."""
    try:
        with urlopen(METRICS_URL, timeout=5) as resp:  # noqa: S310
            body = resp.read().decode()
    except (URLError, OSError) as exc:
        click.echo(f"Failed to fetch {METRICS_URL}: {exc}", err=True)
        return None

    # Prometheus format: metric_name{labels} value (may be scientific notation)
    pattern = re.compile(rf"^{re.escape(METRIC_NAME)}\{{[^}}]*\}}\s+([\d.eE+-]+)")
    for line in body.splitlines():
        if (m := pattern.match(line)):
            return int(float(m.group(1)))
    return None


def wait_ready(container_name: str, host_port: int, timeout: float, interval: float) -> None:
    """Wait until the container's llama-server responds to HTTP or times out."""
    started = time.time()
    while time.time() - started < timeout:
        # Health-check our container's API (not the metrics endpoint)
        try:
            with urlopen(f"http://{SSH_HOST}:{host_port}/health", timeout=3) as resp:  # noqa: S310
                if resp.status == 200:
                    click.echo("Server is ready (health check passed).")
                    return
        except (URLError, OSError):
            pass

        # Also check via docker logs for "server listening"
        result = run_ssh(["docker", "logs", container_name, "2>&1"])
        combined = result.stdout + result.stderr
        if "server is listening" in combined or "HTTP server is listening" in combined:
            click.echo("Server is ready (log: server listening).")
            return

        # Container still alive?
        running = run_ssh(["docker", "inspect", "-f", "{{.State.Running}}", container_name])
        if running.stdout.strip() == "false":
            logs = run_ssh(["docker", "logs", container_name])
            click.echo("Container exited prematurely.", err=True)
            click.echo(logs.stderr[:2000], err=True)
            sys.exit(1)

        time.sleep(interval)

    click.echo(f"Timeout waiting for server ({timeout}s).", err=True)
    sys.exit(1)


def send_completion(host_port: int, max_retries: int = 2) -> None:
    """Send a small completion request to trigger inference VRAM usage."""
    payload = json.dumps({"prompt": "Hello", "n_predict": 64, "temperature": 0.0})
    for attempt in range(1, max_retries + 1):
        try:
            req = Request(
                f"http://{SSH_HOST}:{host_port}/completion",
                data=payload.encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(req, timeout=120) as resp:  # noqa: S310
                resp.read()
            return
        except Exception as exc:
            if attempt < max_retries:
                click.echo(f"Completion request attempt {attempt} failed, retrying…: {exc}", err=True)
                time.sleep(2)
            else:
                click.echo(f"Completion request failed (non-fatal): {exc}", err=True)


# ── cli ──────────────────────────────────────────────────────────────────


@click.command(
    help="Measure llama-server VRAM via the llama-swap metrics endpoint.",
    epilog=(
        "Examples:\n"
        '  ./measure_llama_server_vram_simple.py -- "-hf unsloth/Qwen3-8B-GGUF -c 32768"\n'
        '  ./measure_llama_server_vram_simple.py -- "-hf unsloth/Qwen3.5-2B-GGUF:Q8_0 --ctx-size 4096"\n'
    ),
)
@click.argument("suffix", metavar="CMD_SUFFIX")
@click.option("--container-name", default="vram-measure", help="Docker container name.")
@click.option("--image", default=DEFAULT_IMAGE, help="Docker image for llama.cpp.")
@click.option("--host-port", type=int, default=18080, help="Host port mapped to the container API.")
@click.option("--no-hf-cache", is_flag=True, help="Skip Hugging Face cache volume mount.")
@click.option("--sample-interval", type=float, default=0.5, help="Metrics polling interval in seconds.")
@click.option("--startup-timeout", type=float, default=120.0, help="Max seconds to wait for server startup.")
@click.option("--json", "json_only", is_flag=True, help="Output raw JSON only.")
def main(
    suffix: str,
    container_name: str,
    image: str,
    host_port: int,
    no_hf_cache: bool,
    sample_interval: float,
    startup_timeout: float,
    json_only: bool,
) -> None:
    """Measure VRAM for a llama-server run.

    CMD_SUFFIX is passed directly after the static llama-server arguments
    (model path, context size, quantization flags, etc.).
    """

    # 1. Cleanup
    stop_container(container_name)

    # 2. Baseline
    click.echo("Sampling baseline VRAM…")
    time.sleep(1)
    baseline = get_metric()
    if baseline is None:
        click.echo("Could not read metrics — aborting.", err=True)
        sys.exit(1)
    click.echo(f"  Baseline: {baseline / MiB:.2f} MiB")

    # 3. Start container
    start_container(container_name, image, host_port, suffix, hf_cache=not no_hf_cache)

    # 4. Wait for model to load (HTTP/log readiness)
    click.echo(f"Waiting for server (timeout {startup_timeout}s)…")
    wait_ready(container_name, host_port, startup_timeout, sample_interval)

    # 5. Wait for VRAM to stabilize — model may still be loading into GPU memory
    click.echo("Waiting for VRAM to stabilize…")
    prev: int | None = None
    stable_count = 0
    stable_threshold = 3  # consecutive readings within tolerance
    tolerance_mb = 2  # MiB tolerance between readings
    while stable_count < stable_threshold:
        if (v := get_metric()) is not None:
            if prev is not None and abs(v - prev) > tolerance_mb * MiB:
                stable_count = 0  # still changing
            else:
                stable_count += 1
            click.echo(f"  {v / MiB:.0f} MiB (stable={stable_count}/{stable_threshold})")
            prev = v
        else:
            stable_count = 0
        time.sleep(1)

    # 6. Loaded VRAM
    loaded = get_metric()
    if loaded is None:
        click.echo("Could not read loaded VRAM.", err=True)
        stop_container(container_name)
        sys.exit(1)
    click.echo(f"  Loaded:   {loaded / MiB:.2f} MiB")

    # 7. Inference VRAM — sample in a thread while sending a completion request
    click.echo("Running inference test…")
    samples: list[int] = []
    inference_duration = 10.0

    def _sample() -> None:
        end = time.time() + inference_duration
        while time.time() < end:
            if (v := get_metric()) is not None:
                samples.append(v)
            time.sleep(sample_interval)

    t = threading.Thread(target=_sample)
    t.start()
    send_completion(host_port)
    t.join()

    peak = max(samples) if samples else loaded
    click.echo(f"  Peak:     {peak / MiB:.2f} MiB")

    # 8. Stop container
    click.echo("Stopping container…")
    stop_container(container_name)
    time.sleep(1)

    final = get_metric() or 0

    # 9. Results
    result = {
        "suffix": suffix,
        "container_name": container_name,
        "baseline_bytes": baseline,
        "baseline_mb": round(baseline / MiB, 4),
        "loaded_bytes": loaded,
        "loaded_mb": round(loaded / MiB, 4),
        "loaded_delta_bytes": loaded - baseline,
        "loaded_delta_mb": round((loaded - baseline) / MiB, 4),
        "inference_peak_bytes": peak,
        "inference_peak_mb": round(peak / MiB, 4),
        "inference_peak_delta_bytes": peak - baseline,
        "inference_peak_delta_mb": round((peak - baseline) / MiB, 4),
        "inference_peak_minus_loaded_bytes": peak - loaded,
        "inference_peak_minus_loaded_mb": round((peak - loaded) / MiB, 4),
        "final_bytes": final,
        "final_mb": round(final / MiB, 4),
        "n_samples": len(samples),
    }

    if json_only:
        click.echo(json.dumps(result, indent=2))
    else:
        console = Console()
        console.print()
        console.print(f"[bold]Command suffix:[/bold] {suffix!r}")
        console.print(f"[bold]Samples:[/bold] {result['n_samples']}")
        console.print()

        table = Table(show_header=True, header_style="bold cyan", box=None)
        table.add_column("Metric", style="white")
        table.add_column("MiB", justify="right", style="magenta")
        table.add_column("Bytes", justify="right", style="dim")

        table.add_row("Baseline", f"{result['baseline_mb']:.1f}", f"{result['baseline_bytes']:,}")
        table.add_row("Loaded", f"{result['loaded_mb']:.1f}", f"{result['loaded_bytes']:,}")
        table.add_row("Loaded delta", f"{result['loaded_delta_mb']:.1f}", f"{result['loaded_delta_bytes']:,}")
        table.add_row("Inference peak", f"{result['inference_peak_mb']:.1f}", f"{result['inference_peak_bytes']:,}")
        table.add_row("Peak delta", f"{result['inference_peak_delta_mb']:.1f}", f"{result['inference_peak_delta_bytes']:,}")
        table.add_row("Peak − loaded", f"{result['inference_peak_minus_loaded_mb']:.1f}", f"{result['inference_peak_minus_loaded_bytes']:,}")
        table.add_row("Final (cleanup)", f"{result['final_mb']:.1f}", f"{result['final_bytes']:,}")

        console.print(table)


if __name__ == "__main__":
    main()
