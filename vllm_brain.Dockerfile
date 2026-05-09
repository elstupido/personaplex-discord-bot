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

# 2. [DEFERRED] Blackwell Wheels will be installed last to ensure they aren't overridden.

# 3. Install vLLM-Omni
# WHY: We install vLLM-Omni on top of the validated Blackwell foundation.
RUN git clone https://github.com/vllm-project/vllm-omni.git /app/vllm-omni && \
    cd /app/vllm-omni && \
    python3 -c "import sys; f=open('vllm_omni/entrypoints/stage_utils.py', 'r'); content=f.read(); f.close(); new_content=content.replace('device_list = _map_device_list(stage_id, device_list, visible_device_list)', 'device_list = visible_device_list # FORCED MAPPING'); f=open('vllm_omni/entrypoints/stage_utils.py', 'w'); f.write(new_content); f.close()" && \
    pip install --no-deps -e .

# 4. Install Unified Dependency Stack
# WHY: We manually satisfy the full dependency tree for vLLM-Omni and Fish Speech
# to avoid 'ABI Rape' (generic torch upgrades) while ensuring no 'ModuleNotFoundError's.

RUN pip install --no-cache-dir --no-deps \
    av>=14.0.0 \
    omegaconf>=2.3.0 \
    diffusers>=0.36.0 \
    accelerate==1.12.0 \
    soundfile>=0.13.1 \
    cache-dit==1.3.0 \
    tqdm>=4.66.0 \
    torchsde>=0.2.6 \
    openai-whisper>=20250625 \
    imageio[ffmpeg]>=2.37.2 \
    x-transformers>=2.12.2 \
    einops>=0.8.1 \
    prettytable>=3.8.0 \
    aenum==3.1.16 \
    pyzmq>=25.0.0 \
    janus>=1.0.0 \
    pydub \
    onnxruntime-gpu>=1.23.2 \
    fa3-fwd==0.0.3 \
    "gradio>5.0.0" \
    "transformers<=4.57.3" \
    "lightning>=2.1.0" \
    "hydra-core>=1.3.2" \
    "librosa>=0.10.1" \
    "descript-audio-codec" \
    "pyaudio" \
    "natsort" \
    "soxr" \
    "lazy_loader" \
    "num2words" \
    "huggingface-hub==0.25.2" \
    "fish-speech @ git+https://github.com/fishaudio/fish-speech.git"

# 5. THE FINAL SURGERY: Force Blackwell Wheels
# WHY: We overwrite whatever generic torch/torchaudio the previous steps pulled in.
COPY dist/*.whl /tmp/
RUN pip install --no-cache-dir --force-reinstall --no-deps /tmp/transformer_engine-*.whl /tmp/torchaudio-*.whl && \
    rm -rf /tmp/*.whl

# 5. Final Blueprint Mapping
COPY qwen2_5_omni_5090.yaml /app/stage_config.yaml

# 6. Preload Protection Script
COPY brain_entrypoint.sh /app/brain_entrypoint.sh
RUN chmod +x /app/brain_entrypoint.sh

ENTRYPOINT ["/app/brain_entrypoint.sh"]
