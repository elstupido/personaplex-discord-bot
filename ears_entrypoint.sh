#!/bin/bash
# WHY THIS FILE EXISTS:
# Clean entrypoint for bot-ears (SenseVoice ASR).
# No LD_PRELOAD hacks or symbol pre-resolution.

set -e

export MODELSCOPE_CACHE=/app/model_cache
export HF_HOME=/app/model_cache

echo "🚀 [Ears] Starting SenseVoice ASR Server..."
# WHY: On a clean CUDA 13+ base, we trust the dynamic linker to find the correct torch/cuda symbols.
exec python3 /app/bot_ears_server.py
