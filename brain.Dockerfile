# WHY THIS FILE EXISTS:
# Optimized vLLM-Omni Brain Image. 🧠🧬
#
# WHY:
# We use the official vLLM OpenAI image as a base to ensure ABI parity 
# with the RTX 5090/Blackwell architecture, but we bake in the Fish Speech 
# codecs to avoid 'pip tsunami' at runtime.

FROM vllm/vllm-openai:v0.20.0

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Fish Speech and its heavy ML sub-dependencies
# We use --no-deps on the main git install to prevent Torch downgrades,
# then manually satisfy the rest of the requirement tree.
RUN python3 -m pip install --no-cache-dir --no-deps git+https://github.com/fishaudio/fish-speech.git
RUN python3 -m pip install --no-cache-dir \
    "gradio>5.0.0" \
    "transformers<=4.57.3" \
    "lightning>=2.1.0" \
    "hydra-core>=1.3.2" \
    "librosa>=0.10.1" \
    "descript-audio-codec" \
    "pyaudio" \
    "natsort" \
    "einops" \
    "pydub"

WORKDIR /app
