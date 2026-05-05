import requests
import base64
import os
import wave
import io

def save_wav(pcm_bytes, path, sample_rate):
    with wave.open(path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2) # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_bytes)

def calibrate():
    ref_path = "/app/src/assets/reference_voice.wav"
    if not os.path.exists(ref_path):
        ref_path = "src/assets/reference_voice.wav"
        
    if not os.path.exists(ref_path):
        print(f"Reference not found")
        return

    with open(ref_path, "rb") as f:
        pcm_b64 = base64.b64encode(f.read()).decode()

    # 1. Extract reference stats
    print("[1] Extracting reference stats...")
    res = requests.post("http://localhost:10000/debug/extract_reference", json={"pcm_b64": pcm_b64})
    stats = res.json()
    print(f"Reference Stats: {stats}")

    # 2. Update engine with these stats
    print("[2] Updating engine alignment config...")
    alignment = {
        "mel_mean": stats["mel_mean"],
        "mel_std": stats["mel_std"],
        "emb_scalar": 1.0
    }
    requests.post("http://localhost:10000/debug/instrument", json=alignment)

    # 3. Sweep emb_scalar
    print("[3] Sweeping emb_scalar for identity transfer...")
    scalars = [1.0, 5.0, 10.0, 20.0, 35.0]
    
    for s in scalars:
        print(f"Testing emb_scalar={s}...")
        requests.post("http://localhost:10000/debug/instrument", json={"emb_scalar": s})
        
        # Probe identity
        probe_res = requests.post("http://localhost:10000/debug/probe_identity", json={
            "pcm_b64": pcm_b64,
            "target_tokens": None,
        })
        
        if probe_res.status_code == 200:
            data = probe_res.json()
            audio_b64 = data["audio_pcm_b64"]
            sr = data.get("sample_rate", 22050)
            pcm_bytes = base64.b64decode(audio_b64)
            out_path = f"/app/scratch/test_results/sweep_scalar_{s}.wav"
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            save_wav(pcm_bytes, out_path, sr)
            print(f"Saved {out_path} at {sr}Hz")
        else:
            print(f"Failed: {probe_res.text}")

if __name__ == "__main__":
    calibrate()
