import sys
import os

# Add paths as in config.py
GLM_PATHS = [
    "/app/glm",
    "/app/glm/cosyvoice",
    "/app/glm/third_party/Matcha-TTS"
]
for p in GLM_PATHS:
    if p not in sys.path:
        sys.path.append(p)

try:
    from flow_inference import AudioDecoder
    import inspect
    import torch

    # Try to find the decoder directory
    flow_path = "/app/glm/glm-4-voice-decoder"
    config_path = os.path.join(flow_path, "config.yaml")
    
    # Just inspect the class if we can't load it
    print("--- AudioDecoder.token2wav Signature ---")
    print(inspect.signature(AudioDecoder.token2wav))
    
    # If we can load it, inspect internal model
    # (Assuming we are on a GPU node)
    if os.path.exists(config_path):
        # We don't necessarily need the weights to see the shapes
        # but let's see if we can find the config
        with open(config_path, 'r') as f:
            print("--- Decoder Config ---")
            print(f.read())
            
except Exception as e:
    print(f"Error: {e}")
