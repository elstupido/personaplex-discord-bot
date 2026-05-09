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

# 2. Install Audio Processing Stack
# WHY: Allow dependencies to be satisfied normally.
RUN pip install --no-cache-dir \
    librosa \
    soundfile \
    "fish-speech @ git+https://github.com/fishaudio/fish-speech.git" \
    funasr \
    modelscope

RUN pip install --no-cache-dir omegaconf torch_complex websockets scipy tensorboard pydantic pydub librosa kaldiio soundfile editdistance pyyaml aliyun-python-sdk-core aliyun-python-sdk-kms hydra-core

# 3. THE FINAL SURGERY: Force Blackwell Wheels
RUN pip install --no-cache-dir --force-reinstall --no-deps /tmp/transformer_engine-*.whl /tmp/torchaudio-*.whl && \
    rm -rf /tmp/*.whl

COPY ears_entrypoint.sh /app/ears_entrypoint.sh
RUN chmod +x /app/ears_entrypoint.sh

CMD ["python", "/app/bot_ears_server.py"]
