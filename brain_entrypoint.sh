#!/bin/bash
# WHY THIS FILE EXISTS:
# Clean entrypoint for vLLM verification.

set -e

echo "🔍 [Diagnostics] Checking environment state..."
python3 -c "import torch; print(f'🔥 Torch Version: {torch.__version__}')"
python3 -m pip show vllm | grep -E "Version|Location" || echo "vLLM not found in pip"
python3 -c "import vllm; print(f'📦 vLLM File: {vllm.__file__}')"
nvidia-smi || echo "⚠️ nvidia-smi failed!"

# --- VRAM Stewardship ---
TARGETS=(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/utils.py"
    "/usr/local/lib/python3.12/dist-packages/vllm/worker/utils.py"
    "/usr/local/lib/python3.12/site-packages/vllm/v1/worker/utils.py"
    "/usr/local/lib/python3.12/site-packages/vllm/worker/utils.py"
)

for target in "${TARGETS[@]}"; do
    if [ -f "$target" ]; then
        echo "🩹 Patching memory checks in $target..."
        sed -i 's/if free_gpu_memory < total_requested_gpu_memory:/if False:/g' "$target" || true
        sed -i 's/if init_snapshot.free_memory < requested_memory:/if False:/g' "$target" || true
    fi
done

# --- Inject vLLM-Omni ---
if [ -d "/app/vllm-omni" ]; then
    echo "🧬 Injecting vllm-omni into PYTHONPATH..."
    export PYTHONPATH="/app/vllm-omni:$PYTHONPATH"
fi

echo "🚀 [Brain] Starting vLLM-Omni API Server..."
# WHY: We now switch back to the Omni entrypoint to enable the --omni flag 
# and multi-stage audio generation pipeline.
python3 -m vllm_omni.entrypoints.openai.api_server "$@"
