# WHY THIS FILE EXISTS:
# Clean implementation of the standalone SenseVoice ASR container relying on the 
# native CUDA 13+ support provided by the latest vLLM base for Blackwell hardware.

FROM vllm/vllm-openai:latest

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# 1. System dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg libsndfile1 portaudio19-dev curl \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# 2. Install ASR Dependencies
# WHY: We rely on standard pip resolution over the stable CUDA 13+ base environment.
RUN pip install --no-cache-dir \
    funasr \
    modelscope \
    torch_complex \
    websockets \
    scipy \
    pydantic \
    kaldiio \
    editdistance \
    aliyun-python-sdk-core \
    aliyun-python-sdk-kms \
    soundfile \
    pyaudio

# 3. Preload Protection Script
COPY ears_entrypoint.sh /app/ears_entrypoint.sh
RUN chmod +x /app/ears_entrypoint.sh

CMD ["python", "/app/bot_ears_server.py"]
