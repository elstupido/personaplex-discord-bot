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

# 1. Install Blackwell Wheels
# WHY: We must maintain ABI parity with vllm-brain.
COPY dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/transformer_engine-*.whl && \
    pip install --no-cache-dir /tmp/torchaudio-*.whl && \
    rm -rf /tmp/*.whl

# 2. Install Audio Processing Stack
RUN pip install --no-cache-dir --no-deps funasr modelscope
RUN pip install --no-cache-dir omegaconf torch_complex websockets scipy tensorboard pydantic pydub librosa kaldiio soundfile editdistance pyyaml aliyun-python-sdk-core aliyun-python-sdk-kms hydra-core

COPY ears_entrypoint.sh /app/ears_entrypoint.sh
RUN chmod +x /app/ears_entrypoint.sh

CMD ["python", "/app/bot_ears_server.py"]
