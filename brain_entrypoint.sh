#!/bin/bash
# WHY THIS FILE EXISTS:
# The Runtime Surgeon. 
# 
# WHY:
# Patching vLLM inside the Dockerfile is slow because it requires a 
# rebuild for every tweak. By patching at RUNTIME, we can iterate on 
# memory validation logic instantly.

echo "💾 [Stewardship] Initializing Runtime Surgeon..."

# --- THE TOTAL LOBOTOMY (Runtime Edition) ---
# We target both V0 and V1 worker utils to bypass phantom VRAM usage reporting.
TARGETS=(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/utils.py"
    "/usr/local/lib/python3.12/dist-packages/vllm/worker/utils.py"
)

for target in "${TARGETS[@]}"; do
    if [ -f "$target" ]; then
        echo "🩹 Patching $target..."
        # Bypass V0: free_gpu_memory < total_requested_gpu_memory
        sed -i 's/if free_gpu_memory < total_requested_gpu_memory:/if False:/g' "$target"
        # Bypass V1: init_snapshot.free_memory < requested_memory
        sed -i 's/if init_snapshot.free_memory < requested_memory:/if False:/g' "$target"
    fi
done

# --- EXTENSION PATCHES ---
if [ -d "/app/vllm-omni" ]; then
    echo "🩹 Patching vllm-omni extensions..."
    find /app/vllm-omni -name "utils.py" -exec sed -i 's/if free_gpu_memory < total_requested_gpu_memory:/if False:/g' {} +
    find /app/vllm-omni -name "utils.py" -exec sed -i 's/if init_snapshot.free_memory < requested_memory:/if False:/g' {} +
fi

echo "🐟 Installing Fish Speech codec dependencies from Github..."
apt-get update && apt-get install -y portaudio19-dev
# We use --no-deps to prevent pip from upgrading torch/torchaudio and breaking vLLM's ABI
python3 -m pip install --no-cache-dir --no-deps git+https://github.com/fishaudio/fish-speech.git
# Manually install the necessary dependencies that DON'T break torch
python3 -m pip install --no-cache-dir "gradio>5.0.0" "transformers<=4.57.3" "lightning>=2.1.0" "hydra-core>=1.3.2" "librosa>=0.10.1" "descript-audio-codec" "pyaudio" "natsort" "einops" "pydub"

echo "🔍 [Diagnostics] Checking environment state before launch..."
python3 -c "import torch; print(f'🔥 Torch Version: {torch.__version__}')"
python3 -m pip show vllm | grep Version
echo "🧬 [Surgery] Performing Total Preload of Torch Core Libraries..."
TORCH_LIB_PATH=$(python3 -c 'import torch, os; print(os.path.dirname(torch.__file__) + "/lib")')
export LD_LIBRARY_PATH=$TORCH_LIB_PATH:$LD_LIBRARY_PATH
# WHY: Preload the holy trinity of torch libs to resolve C++ and Python symbols.
# IMPORTANT: We do NOT export this globally to avoid poisoning system tools (like sh) that worker procs use.
PRELOAD_LIBS="$TORCH_LIB_PATH/libc10.so:$TORCH_LIB_PATH/libc10_cuda.so:$TORCH_LIB_PATH/libtorch_python.so"
echo "🧬 [Surgery] Preload Prepared: $PRELOAD_LIBS"

echo "🚀 [Brain] Starting vLLM-Omni API Server (Internalized Surgery)..."
# WHY: We use a one-liner to unset LD_PRELOAD immediately after Python starts.
# This ensures the main process is patched but its children (like sh) stay clean.
LD_PRELOAD=$PRELOAD_LIBS python3 -c "import os, sys, runpy; os.environ.pop('LD_PRELOAD', None); sys.argv=['vllm_omni.entrypoints.openai.api_server'] + sys.argv[1:]; runpy.run_module('vllm_omni.entrypoints.openai.api_server', run_name='__main__')" "$@"
