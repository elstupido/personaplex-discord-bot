# WHY THIS FILE EXISTS:
# Clean implementation of the vLLM-Omni brain container relying on native CUDA 13+ support 
# for Blackwell (RTX 5090). 

FROM vllm/vllm-openai:latest

ENV PYTHONUNBUFFERED=1
# WHY: We are trusting the official vLLM base environment entirely.
# No wheel overrides, no forced ABI mappings.

WORKDIR /app

# 1. System dependencies
RUN apt-get update && apt-get install -y \
    git build-essential portaudio19-dev ffmpeg curl \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# 2. Add vLLM-Omni and Basic Dependencies
# WHY: We clone the repo but do NOT run pip install yet to protect the base vLLM core.
# We manually install the small, safe dependencies that Omni needs to initialize its patcher.
RUN git clone https://github.com/vllm-project/vllm-omni.git /app/vllm-omni && \
    pip install --no-cache-dir \
    aenum \
    pyzmq \
    janus \
    pydub \
    prettytable \
    omegaconf \
    diffusers \
    accelerate \
    einops \
    soundfile \
    tqdm \
    x-transformers \
    torchsde \
    "imageio[ffmpeg]" \
    cache-dit

# 3. Final Blueprint Mapping
COPY qwen2_5_omni_5090.yaml /app/stage_config.yaml

# 4. Entrypoint
COPY brain_entrypoint.sh /app/brain_entrypoint.sh
RUN chmod +x /app/brain_entrypoint.sh

ENTRYPOINT ["/app/brain_entrypoint.sh"]
