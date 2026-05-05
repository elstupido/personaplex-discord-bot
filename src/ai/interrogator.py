"""
WHY THIS FILE EXISTS:
The Identity Interrogator is a diagnostic tool used to systematically sweep 
acoustic conditioning parameters. It helps find the 'Golden Settings' for 
voice cloning by generating multiple audio probes and saving them for review.

WHY THE LOG10 SWEEP?
The model's internal Mel representation often follows a logarithmic distribution. 
By sweeping Mean and Std in log-space, we can find the exact normalization 
statistics needed to align the reference voice embedding with the model's 
training data.
"""
import os
import json
import base64
import requests
import numpy as np
import wave
from tqdm import tqdm
import time

# --- Configuration ---
SERVER_URL = "http://127.0.0.1:10000"
# Absolute paths for container execution
REF_AUDIO_PATH = "/app/src/assets/reference_voice.wav"
OUTPUT_DIR = "/app/src/ai/probe_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# A standard set of VQ tokens for a short phrase.
TARGET_TOKENS = [
    524, 12, 853, 23, 9, 154, 32, 11, 45, 128, 9, 321, 10, 44, 12, 88
]

# --- Parameter Grid ---
# Switching to Log10 as it aligns better with the observed distribution.
# Target: log10(1e-5) = -5.0. To get Mean 0.0, we need Offset approx +5.0
SWEEP_GRID = {
    "mel_mean": [4.0, 5.0, 6.0],
    "mel_std":  [1.0],
    "mel_scale": ["log10"],
    "emb_scalar": [25.0, 50.0]
}

def load_audio_b64(path):
    """Load WAV using standard wave module."""
    with wave.open(path, 'rb') as wf:
        params = wf.getparams()
        frames = wf.readframes(params.nframes)
        samples = np.frombuffer(frames, dtype=np.int16)
        
        # If stereo, take one channel
        if params.nchannels == 2:
            samples = samples[0::2]
        
        # Simple resample (if needed, but assuming 16k for now)
        if params.framerate != 16000:
            print(f"Warning: {path} is {params.framerate}Hz, not 16000Hz. Results may be poor.")
            
        return base64.b64encode(samples.tobytes()).decode()

def save_wav(path, pcm_bytes, sample_rate=22050):
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2) # int16
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)

def run_probe(pcm_b64, tokens, params):
    endpoint = f"{SERVER_URL}/debug/probe_identity"
    payload = {
        "pcm_b64": pcm_b64,
        "target_tokens": tokens,
        "alignment_params": params,
        "sample_rate": 16000
    }
    try:
        resp = requests.post(endpoint, json=payload, timeout=30)
        if resp.status_code != 200:
            print(f"Server Error: {resp.text}")
            return None
        return resp.json()
    except Exception as e:
        print(f"Connection Error: {e}")
        return None

def main():
    print(f"Starting Simplified Identity Interrogation...")
    
    if not os.path.exists(REF_AUDIO_PATH):
        print(f"Error: Reference audio not found at {REF_AUDIO_PATH}")
        return

    pcm_b64 = load_audio_b64(REF_AUDIO_PATH)
    
    combinations = []
    for mean in SWEEP_GRID["mel_mean"]:
        for std in SWEEP_GRID["mel_std"]:
            for scale in SWEEP_GRID["mel_scale"]:
                for scalar in SWEEP_GRID["emb_scalar"]:
                    combinations.append({
                        "mel_mean": mean,
                        "mel_std": std,
                        "mel_scale": scale,
                        "emb_scalar": scalar
                    })

    print(f"Testing {len(combinations)} combinations...")
    
    for i, params in enumerate(tqdm(combinations)):
        res = run_probe(pcm_b64, TARGET_TOKENS, params)
        
        if res and res.get("status") == "ok":
            audio_b64 = res["audio_pcm_b64"]
            audio_bytes = base64.b64decode(audio_b64)
            
            filename = f"probe_{i:03d}_m{params['mel_mean']}_s{params['mel_std']}_esc{params['emb_scalar']}.wav"
            filepath = os.path.join(OUTPUT_DIR, filename)
            save_wav(filepath, audio_bytes)
            
        time.sleep(0.5) # Be gentle

    print(f"\nDone. Check {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
