import requests
import base64
import wave
import json
import os

URL = "http://127.0.0.1:10000"
REF_WAV = "/app/src/assets/reference_voice.wav"
OUTPUT_FILE = "/root/personaplex-discord-bot/scratch/test_results/calibrated_sample.wav"

def load_wav_b64(path):
    import subprocess
    cmd = ["docker", "exec", "ai-voice-bot", "python3", "-c", f"""
import base64, wave, numpy as np
with wave.open('{path}', 'rb') as wf:
    pcm = np.frombuffer(wf.readframes(wf.getparams().nframes), dtype=np.int16)
    if wf.getparams().nchannels == 2: pcm = pcm[0::2]
    print(base64.b64encode(pcm.tobytes()).decode())
"""]
    return subprocess.check_output(cmd).decode().strip()

def generate():
    print("🎙️ GENERATING CALIBRATED TEST SAMPLE...")
    pcm_b64 = load_wav_b64(REF_WAV)
    
    # A slightly longer set of tokens to hear the voice better.
    # These are arbitrary VQ tokens that represent speech-like rhythms.
    TARGET_TOKENS = [
        524, 12, 853, 23, 9, 154, 32, 11, 45, 128, 9, 321, 10, 44, 12, 88,
        524, 12, 853, 23, 9, 154, 32, 11, 45, 128, 9, 321, 10, 44, 12, 88
    ]

    params = {
        "mel_scale": "log10",
        "mel_mean": -2.76,
        "mel_std": 0.86,
        "emb_scalar": 1.0
    }

    payload = {
        "pcm_b64": pcm_b64,
        "target_tokens": TARGET_TOKENS,
        "alignment_params": params,
        "sample_rate": 16000
    }

    resp = requests.post(f"{URL}/debug/probe_identity", json=payload)
    if resp.status_code == 200:
        data = resp.json()
        if 'audio_pcm_b64' not in data:
            print(f"❌ ERROR: Missing audio key. Response: {data}")
            return
        audio_bytes = base64.b64decode(data['audio_pcm_b64'])
        
        with wave.open(OUTPUT_FILE, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(22050) # GLM decoder output is 22050
            wf.writeframes(audio_bytes)
            
        print(f"✅ SUCCESS: Sample saved to {OUTPUT_FILE}")
    else:
        print(f"❌ FAILED: {resp.text}")

if __name__ == "__main__":
    generate()
