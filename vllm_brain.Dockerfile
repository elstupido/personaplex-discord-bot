# MANDATORY: Original vLLM base that contains the engine core
FROM vllm/vllm-openai:v0.20.0

# BLACKWELL COMPATIBILITY BRIDGE
ENV TORCH_CUDA_ARCH_LIST="9.0"
ENV CUDA_MODULE_LOADING=LAZY
ENV TORCH_CUDNN_V8_API_ENABLED=1
ENV TORCH_CUDNN_SDPA_ENABLED=1
ENV PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# 1. System dependencies
RUN apt-get update && apt-get install -y \
    git build-essential portaudio19-dev ffmpeg curl \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Blackwell Wheels
# WHY: These are pre-compiled for sm_120 (RTX 5090).
COPY dist/*.whl /tmp/
    pip install --no-cache-dir /tmp/transformer_engine-*.whl && \
    pip install --no-cache-dir /tmp/torchaudio-*.whl && \
    rm -rf /tmp/*.whl

# 3. Install vLLM-Omni
# WHY: We install vLLM-Omni on top of the validated Blackwell foundation.
RUN git clone https://github.com/vllm-project/vllm-omni.git /app/vllm-omni && \
    cd /app/vllm-omni && \
    python3 -c "import sys; f=open('vllm_omni/entrypoints/stage_utils.py', 'r'); content=f.read(); f.close(); new_content=content.replace('device_list = _map_device_list(stage_id, device_list, visible_device_list)', 'device_list = visible_device_list # FORCED MAPPING'); f=open('vllm_omni/entrypoints/stage_utils.py', 'w'); f.write(new_content); f.close()" && \
    pip install --no-deps -e .

# 4. Install Audio Codecs and Engine Dependencies
RUN pip install --no-cache-dir --no-deps \
    descript-audio-codec \
    librosa \
    soundfile \
    "fish-speech @ git+https://github.com/fishaudio/fish-speech.git"

# 5. Final Blueprint Mapping
COPY qwen2_5_omni_5090.yaml /app/stage_config.yaml

# 6. Preload Protection Script
COPY brain_entrypoint.sh /app/brain_entrypoint.sh
RUN chmod +x /app/brain_entrypoint.sh

ENTRYPOINT ["/app/brain_entrypoint.sh"]
