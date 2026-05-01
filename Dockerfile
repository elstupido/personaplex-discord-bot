# MANDATORY: NVIDIA 25.01 for RTX 5090 (sm_120) support
# This image is ~15GB, ensure WSL2 has enough disk space.
ARG BASE_IMAGE="nvcr.io/nvidia/pytorch"
ARG BASE_IMAGE_TAG="25.01-py3" 

FROM ${BASE_IMAGE}:${BASE_IMAGE_TAG}
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# BLACKWELL COMPATIBILITY BRIDGE
ENV TORCH_CUDA_ARCH_LIST="9.0"
ENV CUDA_MODULE_LOADING=LAZY
# 5090 Throughput & VRAM Optimizations
ENV TORCH_CUDNN_V8_API_ENABLED=1
ENV TORCH_CUDNN_SDPA_ENABLED=1
ENV PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV PYTHONUNBUFFERED=1

# 1. System dependencies with caching
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    pkg-config libopus-dev libopus0 ffmpeg git wget ca-certificates \
    cmake ninja-build build-essential libsox-dev libasound2-dev ccache \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Parallel Clone Strategy (Combined into one layer to reduce overhead)
RUN git clone --depth 1 https://github.com/THUDM/GLM-4-Voice.git /app/glm && \
    git clone --depth 1 https://github.com/gpt-omni/mini-omni2.git /app/omni && \
    git clone --depth 1 https://github.com/NVIDIA/personaplex.git /tmp/model && \
    cd /tmp/model && git fetch --depth 1 origin pull/72/head && git checkout FETCH_HEAD && \
    cp -r /tmp/model/moshi /app/moshi && \
    rm -rf /app/glm/.git /app/omni/.git /tmp/model

# 3. Create PYNINI SHIM (Prevents massive build-time compilation of C++ bindings)
RUN mkdir -p /usr/local/lib/python3.12/dist-packages/pynini && \
    touch /usr/local/lib/python3.12/dist-packages/pynini/__init__.py && \
    echo 'def Far(*args, **kwargs): return None' >> /usr/local/lib/python3.12/dist-packages/pynini/__init__.py && \
    echo 'def Fst(*args, **kwargs): return None' >> /usr/local/lib/python3.12/dist-packages/pynini/__init__.py && \
    mkdir -p /usr/local/lib/python3.12/dist-packages/pynini-2.1.5.dist-info && \
    echo 'Metadata-Version: 2.1\nName: pynini\nVersion: 2.1.5' > /usr/local/lib/python3.12/dist-packages/pynini-2.1.5.dist-info/METADATA

# 4. Critical Build Tools & Fast Transfer
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -q --no-cache-dir num2words Cython wheel "packaging<=24.2" "six<=1.16" hf_transfer

# 5. Install Pre-built Wheels (TransformerEngine and Torchaudio)
# Expects wheels to be in ./dist/ directory on host
COPY dist/*.whl /tmp/
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir /tmp/transformer_engine-*.whl && \
    pip install --no-cache-dir /tmp/torchaudio-*.whl && \
    python -c "import torchaudio; import transformer_engine; print('SUCCESS: Blackwell Wheels Verified')" && \
    rm -rf /tmp/*.whl

# 6. Optimized Dependency Installation
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -q "ruamel.yaml<0.18.0" && \
    sed -i 's/onnxruntime-gpu==1.16.0/onnxruntime-gpu>=1.20.0/g' /app/glm/requirements.txt && \
    sed -i 's/numpy==1.24.4/numpy>=1.26.0/g' /app/glm/requirements.txt && \
    sed -i 's/grpcio==1.57.0/grpcio>=1.62.0/g' /app/glm/requirements.txt && \
    sed -i 's/grpcio-tools==1.57.0/grpcio-tools>=1.62.0/g' /app/glm/requirements.txt && \
    sed -i '/torch/d; /torchaudio/d; /pynini/d; /WeTextProcessing/d; /flash-attn/d; /xformers/d; /triton/d' /app/glm/requirements.txt && \
    pip install -q --no-cache-dir --no-build-isolation -r /app/glm/requirements.txt && \
    sed -i '/torch/d; /torchaudio/d; /flash-attn/d; /xformers/d; /triton/d' /app/omni/requirements.txt && \
    pip install -q --no-cache-dir --no-build-isolation -r /app/omni/requirements.txt

# 7. Final App Packages
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -U pip setuptools wheel && \
    pip install "Cython<3.0.0" "numpy<2.0.0" && \
    if [ -f /app/moshi/requirements.txt ]; then sed -i '/torch/d; /torchaudio/d; /pynini/d' /app/moshi/requirements.txt; fi && \
    pip install -q --no-cache-dir --no-deps /app/moshi/. && \
    pip install -q --no-cache-dir --prefer-binary --no-build-isolation \
    python-dotenv PyNaCl accelerate "bitsandbytes>=0.45.0" aiohttp \
    matcha-tts setuptools faster-whisper
RUN pip install --no-cache-dir --force-reinstall \
    "py-cord[voice] @ git+https://github.com/Pycord-Development/pycord.git@refs/pull/3159/head"

# 8. PRE-DOWNLOAD DECODER WEIGHTS (High-Speed HF Transfer)
RUN HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download THUDM/glm-4-voice-decoder --local-dir /app/glm/glm-4-voice-decoder

COPY supervisor.py /app/supervisor.py
COPY src/ /app/src/

RUN mkdir -p /app/ssl
ENV HF_HOME=/app/model_cache
ENV TORCH_HOME=/app/model_cache

EXPOSE 8998
EXPOSE 10000
CMD ["python", "supervisor.py"]
