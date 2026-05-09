# MANDATORY: Original vLLM base for ABI parity
FROM vllm/vllm-openai:v0.20.0

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV PYTHONUNBUFFERED=1

# BLACKWELL COMPATIBILITY BRIDGE
ENV TORCH_CUDA_ARCH_LIST="9.0"

RUN apt-get update && apt-get install -y ffmpeg libsndfile1 portaudio19-dev \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1. [DEFERRED] Blackwell Wheels
COPY dist/*.whl /tmp/

# 2. Install Unified Dependency Stack
# WHY: Total parity with vllm-brain. We manually satisfy the tree to avoid ABI rape.
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
    "fish-speech @ git+https://github.com/fishaudio/fish-speech.git" \
    funasr \
    modelscope \
    torch_complex \
    websockets \
    scipy \
    tensorboard \
    pydantic \
    kaldiio \
    editdistance \
    aliyun-python-sdk-core \
    aliyun-python-sdk-kms

# 3. THE FINAL SURGERY: Force Blackwell Wheels
RUN pip install --no-cache-dir --force-reinstall --no-deps /tmp/transformer_engine-*.whl /tmp/torchaudio-*.whl && \
    rm -rf /tmp/*.whl

COPY ears_entrypoint.sh /app/ears_entrypoint.sh
RUN chmod +x /app/ears_entrypoint.sh

CMD ["python", "/app/bot_ears_server.py"]
