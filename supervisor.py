#!/usr/bin/env python3
"""
Supervisor process that runs both the Moshi inference server and the Discord bot
inside a single Docker container.

Responsibilities:
  1. Start the Moshi server (python -m moshi.server) as a subprocess
  2. Wait for the server to become reachable on localhost:8998
  3. Start the Discord bot (python src/bot.py) as a subprocess
  4. Monitor both — if either exits, log and terminate the other
  5. Handle SIGTERM/SIGINT for graceful Docker stop
"""

import subprocess
import os
import sys
import time
import signal
import socket
import random

# Blackwell (RTX 5090) Bridge: Force Hopper compatibility paths
os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"
os.environ["CUDA_MODULE_LOADING"] = "LAZY"
# 5090 Throughput Optimizations
os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"
os.environ["TORCH_CUDNN_SDPA_ENABLED"] = "1"
os.environ["TORCH_XLA_FLAGS"] = "--xla_gpu_enable_high_throughput_pipeline=true"
# Suppress the sm_120 warning since we are handling it via the bridge
os.environ["PYTHONWARNINGS"] = "ignore:NVIDIA GeForce RTX 5090 with CUDA capability sm_120"


def wait_for_server(host="127.0.0.1", port=10000, timeout=600, label="server"):
    """Poll until the server is accepting TCP connections."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection((host, port), timeout=2):
                return True
        except (ConnectionRefusedError, OSError):
            time.sleep(2)
    return False


def main():
    # Preflight checks
    if not os.getenv("DISCORD_TOKEN"):
        print("[supervisor] ERROR: DISCORD_TOKEN is not set.")
        sys.exit(1)

    model_type = os.getenv("MODEL_TYPE", "moshi").lower()
    print(f"[supervisor] Configuring for MODEL_TYPE: {model_type}")

    if model_type == "moshi":
        if not os.getenv("HF_TOKEN"):
            print("[supervisor] ERROR: HF_TOKEN is required for Moshi weights.")
            sys.exit(1)
        server_cmd = [sys.executable, "-m", "moshi.server", "--host", "0.0.0.0", "--static", "none"]
        port = 8998
    elif model_type == "glm-4":
        server_cmd = [sys.executable, "src/bridge/glm_server.py", "--host", "127.0.0.1", "--port", "10000"]
        port = 10000
    elif model_type == "mini-omni":
        server_cmd = [sys.executable, "omni/server.py", "--ip", "0.0.0.0", "--port", "60808"]
        port = 60808
    else:
        print(f"[supervisor] ERROR: Unknown MODEL_TYPE '{model_type}'")
        sys.exit(1)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    moshi_proc = subprocess.Popen(
        server_cmd,
        stdout=sys.stdout,
        stderr=sys.stderr,
        env=env,
    )

    if not wait_for_server(port=port, timeout=600, label=model_type.upper()):
        print(f"[supervisor] ERROR: {model_type.upper()} server did not become ready!")
        moshi_proc.terminate()
        sys.exit(1)

    bot_proc = subprocess.Popen(
        [sys.executable, "src/bot.py"],
        stdout=sys.stdout,
        stderr=sys.stderr,
    )

    # Graceful shutdown handler
    def handle_signal(sig, frame):
        print(f"[supervisor] Received signal {sig}, shutting down...")
        bot_proc.terminate()
        moshi_proc.terminate()
        bot_proc.wait(timeout=10)
        moshi_proc.wait(timeout=10)
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    # Monitor both processes
    while True:
        if moshi_proc.poll() is not None:
            print(f"[supervisor] {model_type.upper()} server exited with code {moshi_proc.returncode}")
            bot_proc.terminate()
            sys.exit(1)
        if bot_proc.poll() is not None:
            print(f"[supervisor] Discord bot exited with code {bot_proc.returncode}")
            moshi_proc.terminate()
            sys.exit(1)
        time.sleep(1)


if __name__ == "__main__":
    main()
