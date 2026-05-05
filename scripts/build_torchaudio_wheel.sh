#!/bin/bash
set -e

# Use the same base image as the Dockerfile for 100% compatibility
BASE_IMAGE="nvcr.io/nvidia/pytorch:25.01-py3"

# RESOURCE TUNING: You have 32 cores but 15GB RAM. 
# CUDA compilation (cicc) uses ~3-4GB per job.
# We limit to 4 jobs to prevent Signal 9 (OOM) kills.
MAX_JOBS=4

echo "Starting Docker-based build for torchaudio..."
echo "Using base image: $BASE_IMAGE"
echo "Resource Limit: MAX_JOBS=$MAX_JOBS"

docker run --rm \
    -v "$(pwd):/project" \
    -v "/root/.cache/ccache:/root/.cache/ccache" \
    $BASE_IMAGE \
    /bin/bash -c "
        apt-get update && \
        apt-get install -y --no-install-recommends cmake ninja-build pkg-config libsox-dev libopus-dev ffmpeg ccache git zip unzip && \
        
        # Verify the PyTorch version inside the container
        TORCH_VERSION=\$(python3 -c 'import torch; print(torch.__version__)') && \
        echo 'Detected PyTorch version inside container: \$TORCH_VERSION' && \

        git clone --branch release/2.6 --depth 1 --recursive https://github.com/pytorch/audio.git /tmp/audio && \
        cd /tmp/audio && \
        
        # Build flags
        export FORCE_CUDA=1 \
        export MAX_JOBS=$MAX_JOBS \
        export CMAKE_C_COMPILER_LAUNCHER=ccache \
        export CMAKE_CXX_COMPILER_LAUNCHER=ccache \
        export CMAKE_CUDA_COMPILER_LAUNCHER=ccache && \
        export BUILD_KALDI=0 BUILD_RNNT=0 BUILD_RIR=0 BUILD_CUDA_CTC_DECODER=0 && \
        
        # Build the wheel
        echo 'Building torchaudio wheel (Jobs: $MAX_JOBS)...' && \
        python3 setup.py bdist_wheel && \
        
        # PATCH METADATA: Strip the strict version requirement to satisfy pip inside the container
        echo 'Patching wheel metadata to remove strict version requirement...' && \
        cd dist && \
        WHL_FILE=\$(ls *.whl) && \
        unzip \$WHL_FILE 'torchaudio-*.dist-info/METADATA' && \
        sed -i 's/Requires-Dist: torch ==.*/Requires-Dist: torch/' torchaudio-*.dist-info/METADATA && \
        zip -u \$WHL_FILE torchaudio-*.dist-info/METADATA && \
        cd .. && \
        
        mkdir -p /project/dist && \
        rm -f /project/dist/*.whl && \
        cp dist/*.whl /project/dist/ && \
        echo 'Build successful! Compatible wheel saved to /project/dist/'
    "
