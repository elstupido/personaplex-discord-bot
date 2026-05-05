import requests
import base64
import wave
import numpy as np
import json
from tqdm import tqdm

URL = "http://127.0.0.1:10000"
REF_WAV = "/app/src/assets/reference_voice.wav"

def load_wav(path):
    # This must run on the host, but the path is for the container.
    # I'll use docker exec to get the base64.
    import subprocess
    cmd = ["docker", "exec", "ai-voice-bot", "python3", "-c", f"""
import base64, wave, numpy as np
with wave.open('{path}', 'rb') as wf:
    pcm = np.frombuffer(wf.readframes(wf.getparams().nframes), dtype=np.int16)
    if wf.getparams().nchannels == 2: pcm = pcm[0::2]
    print(base64.b64encode(pcm.tobytes()).decode())
"""]
    return subprocess.check_output(cmd).decode().strip()

def hunt():
    print("🎯 HUNTING FOR IDENTITY PARAMETERS...")
    pcm_b64 = load_wav(REF_WAV)
    
    # Grid search for distribution alignment
    scales = ["log10", "ln"]
    # We will vary mel_mean and see the result of /debug/extract_reference
    
    results = []
    for scale in scales:
        print(f"\n--- Testing {scale} ---")
        # First, set the scale
        requests.post(f"{URL}/debug/instrument", json={"mel_scale": scale})
        
        # We want to find a mel_mean that results in a final mel_mean near 0.0
        # Let's extract with mel_mean=0 and mel_std=1 to see the raw mean
        requests.post(f"{URL}/debug/instrument", json={"mel_mean": 0.0, "mel_std": 1.0})
        res = requests.post(f"{URL}/debug/extract_reference", json={"pcm_b64": pcm_b64})
        raw_stats = res.json()
        raw_mean = raw_stats['mel_mean']
        print(f"Raw Mel Mean (at mean=0): {raw_mean:.4f}")
        
        # The optimal mel_mean to reach a normalized mean of 0 is raw_mean itself.
        results.append({"scale": scale, "optimal_mean": raw_mean})
        
    print("\n[RESULT]")
    for r in results:
        print(f"Scale: {r['scale']} -> Suggested mel_mean: {r['optimal_mean']:.4f}")

if __name__ == "__main__":
    hunt()
