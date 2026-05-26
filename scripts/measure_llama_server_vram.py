#!/usr/bin/env python3
"""Measure llama-server VRAM usage with amd-smi.

Runs llama-server in the Atlas Docker Compose stack, waits for the model to load,
samples VRAM with `amd-smi`, sends a small completion request, then reports the
loaded and inference-time VRAM deltas.

Default settings match the Atlas llama-bench service, which shares the
llama-swap Hugging Face cache.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from typing import Any

REMOTE_SCRIPT = r"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

opts = json.loads(sys.argv[1])
model = opts["model"]
ctx = str(opts["ctx"])
port = str(opts["port"])
container_name = opts["container_name"]
compose_file = opts["compose_file"]
compose_project = opts["compose_project"]
compose_service = opts["compose_service"]
request_prompt = opts["prompt"]
n_predict = str(opts["n_predict"])
sample_interval = float(opts["sample_interval"])
startup_timeout = float(opts["startup_timeout"])
request_timeout = float(opts["request_timeout"])
extra_args = opts["extra_args"]
no_mmproj = bool(opts["no_mmproj"])

safe_model = model.replace("/", "_").replace(":", "_")
log_path = Path(f"/tmp/llmfit-server-{safe_model}-{ctx}.log")
samples_path = Path(f"/tmp/llmfit-server-{safe_model}-{ctx}.mem")


def run(command: list[str], check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=True, check=check)


def used_vram_mb() -> int:
    data = subprocess.check_output(["amd-smi", "metric", "-m", "--json"], text=True)
    payload = json.loads(data)
    return int(payload["gpu_data"][0]["mem_usage"]["used_vram"]["value"])


def cleanup() -> None:
    run(["docker", "rm", "-f", container_name])


cleanup()
time.sleep(2)
baseline_mb = used_vram_mb()

server_args = [
    "-hf",
    model,
    "-ngl",
    str(opts["gpu_layers"]),
    "-c",
    ctx,
    "--host",
    "0.0.0.0",
    "--port",
    "8080",
]
if no_mmproj:
    server_args.append("--no-mmproj")
server_args.extend(extra_args)

cmd = [
    "docker",
    "compose",
    "-p",
    compose_project,
    "-f",
    compose_file,
    "run",
    "-d",
    "--rm",
    "--name",
    container_name,
    "-p",
    f"{port}:8080",
    "--entrypoint",
    "/app/llama-server",
    compose_service,
    *server_args,
]
container_id = subprocess.check_output(cmd, text=True).strip()

loaded = False
startup_error = None
started_at = time.time()
while time.time() - started_at < startup_timeout:
    logs = run(["docker", "logs", container_name]).stdout + run(["docker", "logs", container_name]).stderr
    if "server is listening" in logs or "HTTP server is listening" in logs:
        loaded = True
        break
    inspect = run(["docker", "inspect", "-f", "{{.State.Running}}", container_name])
    if inspect.returncode != 0 or inspect.stdout.strip() == "false":
        startup_error = "container exited before server was ready"
        break
    time.sleep(1)

# Give ROCm/llama-server a moment to settle after startup.
time.sleep(3)
loaded_mb = used_vram_mb()

samples: list[int] = []
stop_sampling = False


def monitor() -> None:
    with samples_path.open("w") as file:
        while not stop_sampling:
            try:
                used = used_vram_mb()
            except Exception:
                used = 0
            samples.append(used)
            file.write(f"{time.time():.3f} {used}\n")
            file.flush()
            time.sleep(sample_interval)

request_payload = json.dumps(
    {
        "prompt": request_prompt,
        "n_predict": int(n_predict),
        "temperature": 0.0,
    }
)

thread = threading.Thread(target=monitor)
thread.start()
request = subprocess.run(
    [
        "curl",
        "-s",
        "--max-time",
        str(int(request_timeout)),
        "-X",
        "POST",
        f"http://127.0.0.1:{port}/completion",
        "-H",
        "Content-Type: application/json",
        "--data-binary",
        "@-",
    ],
    input=request_payload,
    text=True,
    capture_output=True,
    check=False,
)
stop_sampling = True
thread.join()

after_request_mb = used_vram_mb()
logs = run(["docker", "logs", container_name]).stdout + run(["docker", "logs", container_name]).stderr
log_path.write_text(logs)
cleanup()
time.sleep(2)
final_mb = used_vram_mb()

interesting_needles = [
    "model buffer size",
    "llama_context: n_ctx",
    "KV buffer size",
    "llama_kv_cache: size =",
    "RS buffer size",
    "compute buffer size",
    "server is listening",
    "slot launch_slot",
    "prompt eval time",
    "eval time",
]
interesting_logs = [line for line in logs.splitlines() if any(needle in line for needle in interesting_needles)]

peak_mb = max(samples) if samples else after_request_mb
result = {
    "model": model,
    "ctx": int(ctx),
    "container_id": container_id,
    "loaded": loaded,
    "startup_error": startup_error,
    "baseline_mb": baseline_mb,
    "loaded_mb": loaded_mb,
    "loaded_delta_mb": loaded_mb - baseline_mb,
    "inference_peak_mb": peak_mb,
    "inference_peak_delta_mb": peak_mb - baseline_mb,
    "inference_peak_minus_loaded_mb": peak_mb - loaded_mb,
    "after_request_mb": after_request_mb,
    "after_request_minus_loaded_mb": after_request_mb - loaded_mb,
    "final_mb": final_mb,
    "request_returncode": request.returncode,
    "request_stderr": request.stderr[:1000],
    "response_prefix": request.stdout[:500],
    "log_path": str(log_path),
    "samples_path": str(samples_path),
    "interesting_logs": interesting_logs[-80:],
}
print(json.dumps(result, indent=2))
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure llama-server VRAM usage on Atlas with amd-smi.")
    parser.add_argument("model", help="Hugging Face GGUF repo, e.g. unsloth/Qwen3.6-27B-GGUF")
    parser.add_argument("--ctx", type=int, default=32768, help="llama-server context size")
    parser.add_argument("--ssh-host", default="atlas", help="SSH host to run measurements on")
    parser.add_argument("--compose-file", default="/opt/compose/compose.yaml")
    parser.add_argument("--compose-project", default="main")
    parser.add_argument("--compose-service", default="llama-bench")
    parser.add_argument("--container-name", default="llmfit-measure-server")
    parser.add_argument("--port", type=int, default=18080, help="Temporary host port mapped to llama-server")
    parser.add_argument("--gpu-layers", type=int, default=99)
    parser.add_argument("--with-mmproj", action="store_true", help="Allow llama.cpp to auto-load mmproj if available")
    parser.add_argument("--n-predict", type=int, default=256)
    parser.add_argument("--prompt", default="Write a short paragraph about memory estimation.")
    parser.add_argument("--sample-interval", type=float, default=0.1)
    parser.add_argument("--startup-timeout", type=float, default=120.0)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument(
        "--server-arg",
        action="append",
        default=[],
        help="Extra llama-server argument. Repeat for multiple tokens, e.g. --server-arg=-ctk --server-arg=q8_0",
    )
    parser.add_argument("--json", action="store_true", help="Print raw JSON result only")
    return parser.parse_args()


def run_remote(options: dict[str, Any], ssh_host: str) -> dict[str, Any]:
    remote_options = shlex.quote(json.dumps(options))
    command = ["ssh", ssh_host, f"python3 - {remote_options}"]
    completed = subprocess.run(command, input=REMOTE_SCRIPT, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        print(completed.stdout, end="", file=sys.stdout)
        print(completed.stderr, end="", file=sys.stderr)
        raise SystemExit(completed.returncode)
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError:
        print(completed.stdout)
        print(completed.stderr, file=sys.stderr)
        raise


def print_summary(result: dict[str, Any]) -> None:
    print(f"Model: {result['model']}")
    print(f"Context: {result['ctx']:,}")
    print(f"Loaded: {result['loaded']}")
    if result.get("startup_error"):
        print(f"Startup error: {result['startup_error']}")
    print()
    print("amd-smi VRAM:")
    print(f"  baseline:              {result['baseline_mb']:>8} MB")
    print(f"  loaded:                {result['loaded_mb']:>8} MB")
    print(f"  loaded delta:          {result['loaded_delta_mb']:>8} MB")
    print(f"  inference peak:        {result['inference_peak_mb']:>8} MB")
    print(f"  inference peak delta:  {result['inference_peak_delta_mb']:>8} MB")
    print(f"  peak - loaded:         {result['inference_peak_minus_loaded_mb']:>8} MB")
    print(f"  after request:         {result['after_request_mb']:>8} MB")
    print(f"  final after cleanup:   {result['final_mb']:>8} MB")
    print()
    print(f"Request rc: {result['request_returncode']}")
    if result.get("request_stderr"):
        print(f"Request stderr: {result['request_stderr']}")
    print(f"Response prefix: {result.get('response_prefix', '')[:160]!r}")
    print()
    print(f"Remote log: {result['log_path']}")
    print(f"Remote samples: {result['samples_path']}")
    print()
    print("Relevant llama-server logs:")
    for line in result["interesting_logs"]:
        print(f"  {line}")


def main() -> None:
    args = parse_args()
    options = {
        "model": args.model,
        "ctx": args.ctx,
        "port": args.port,
        "container_name": args.container_name,
        "compose_file": args.compose_file,
        "compose_project": args.compose_project,
        "compose_service": args.compose_service,
        "gpu_layers": args.gpu_layers,
        "no_mmproj": not args.with_mmproj,
        "n_predict": args.n_predict,
        "prompt": args.prompt,
        "sample_interval": args.sample_interval,
        "startup_timeout": args.startup_timeout,
        "request_timeout": args.request_timeout,
        "extra_args": args.server_arg,
    }
    result = run_remote(options, args.ssh_host)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print_summary(result)


if __name__ == "__main__":
    main()
