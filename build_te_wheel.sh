#!/bin/bash
set -e

# Use the same base image as the Dockerfile for 100% compatibility
BASE_IMAGE="nvcr.io/nvidia/pytorch:25.01-py3"

# RESOURCE TUNING: You have 32 cores but 15GB RAM. 
# CUDA compilation (cicc) uses ~3-4GB per job.
# We limit to 4 jobs to prevent Signal 9 (OOM) kills.
MAX_JOBS=4

echo "Starting Docker-based build for TransformerEngine..."
echo "Using base image: $BASE_IMAGE"
echo "Resource Limit: MAX_JOBS=$MAX_JOBS"

docker run --rm \
    -v "$(pwd):/project" \
    -v "/root/.cache/ccache:/root/.cache/ccache" \
    $BASE_IMAGE \
    /bin/bash -c "
        apt-get update && \
        apt-get install -y --no-install-recommends cmake ninja-build pkg-config libsox-dev libopus-dev ffmpeg ccache git zip unzip && \
        
        # Build flags for maximum performance on Blackwell
        export FORCE_CUDA=1 \
        export MAX_JOBS=$MAX_JOBS \
        export CMAKE_C_COMPILER_LAUNCHER=ccache \
        export CMAKE_CXX_COMPILER_LAUNCHER=ccache \
        export CMAKE_CUDA_COMPILER_LAUNCHER=ccache && \
        
        echo 'Cloning TransformerEngine...' && \
        git clone --branch main --depth 1 --recursive https://github.com/NVIDIA/TransformerEngine.git /tmp/te && \
        cd /tmp/te && \
        
        echo 'Building TransformerEngine wheel (Jobs: $MAX_JOBS)...' && \
        python3 setup.py bdist_wheel && \
        
        mkdir -p /project/dist && \
        cp dist/*.whl /project/dist/ && \
        echo 'Build successful! TransformerEngine wheel saved to /project/dist/'
    "
