#!/bin/bash
set -e

echo "👂 [Ears] Initializing Acoustic Surgery (Brain Sync)..."

# WHY: We must match the exact library resolution pattern of the vllm-brain
# to ensure ABI compatibility with the 5090's Torch 2.8.0 core.
TORCH_LIB_PATH=$(python3 -c 'import torch, os; print(os.path.dirname(torch.__file__) + "/lib")')
export LD_LIBRARY_PATH=$TORCH_LIB_PATH:$LD_LIBRARY_PATH

# Preload the holy trinity of torch libs to resolve C++ and Python symbols
PRELOAD_LIBS="$TORCH_LIB_PATH/libc10.so:$TORCH_LIB_PATH/libc10_cuda.so:$TORCH_LIB_PATH/libtorch_python.so"

echo "🧬 [Surgery] Preload Prepared (In-process only)"

export MODELSCOPE_CACHE=/app/model_cache
export HF_HOME=/app/model_cache

# We avoid 'find' or 'head' here to prevent symbol lookup errors in shell tools
# WHY: Apply surgery ONLY to the python process to avoid poisoning system tools like 'sh' or 'pip'.
echo "🚀 [Ears] Starting SenseVoice ASR Server..."
LD_PRELOAD=$PRELOAD_LIBS exec python3 /app/bot_ears_server.py
