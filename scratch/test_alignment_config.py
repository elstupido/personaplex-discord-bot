import requests
import base64
import json
import struct
import math

URL = "http://127.0.0.1:10000"

def create_dummy_audio(duration=0.5, sr=48000):
    """Create a simple sine wave PCM without numpy."""
    num_samples = int(duration * sr)
    audio_data = []
    for i in range(num_samples):
        # 440Hz Sine wave
        sample = math.sin(2 * math.pi * 440 * i / sr)
        # Convert to 16-bit PCM
        audio_data.append(int(sample * 32767))
    
    # Pack into bytes (little-endian 16-bit)
    pcm_bytes = struct.pack(f"<{len(audio_data)}h", *audio_data)
    return base64.b64encode(pcm_bytes).decode()

def run_alignment_test():
    print("🚀 GLM-4-Voice Alignment Instrumentation Test (Host-Side)")
    print("==========================================================")
    
    try:
        pcm_b64 = create_dummy_audio()
        
        # 1. Baseline
        print("\n[STEP 1] Fetching Baseline Features...")
        res = requests.post(f"{URL}/debug/extract_reference", json={"pcm_b64": pcm_b64})
        if res.status_code != 200:
            print(f"❌ Error: Server returned {res.status_code}")
            return
            
        data = res.json()
        base_mean = data['mel_mean']
        print(f"✅ Baseline Mel Mean: {base_mean:.4f}")
        print(f"✅ Mel Shape: {data['mel_shape']}")
        
        # 2. Instrument Shift
        print("\n[STEP 2] Tweaking Mel Mean via /debug/instrument...")
        requests.post(f"{URL}/debug/instrument", json={"mel_mean": -30.0})
        
        # 3. Verify Shift
        print("\n[STEP 3] Verifying Parameter Twerk...")
        res = requests.post(f"{URL}/debug/extract_reference", json={"pcm_b64": pcm_b64})
        new_mean = res.json()['mel_mean']
        print(f"✅ New Mel Mean: {new_mean:.4f}")
        
        if abs(new_mean - base_mean) > 1.0:
            print("💎 SUCCESS: Real-time parameter update confirmed.")
        else:
            print("⚠️ WARNING: Parameter update did not result in expected feature shift.")
            
        # 4. Probe Identity (Direct Decoder Test)
        print("\n[STEP 4] Probing Identity Path (Direct-to-Decoder)...")
        # Just use some dummy tokens to see if it synthesizes
        probe_payload = {
            "pcm_b64": pcm_b64,
            "target_tokens": [10, 20, 30, 40, 50],
            "sample_rate": 48000
        }
        res = requests.post(f"{URL}/debug/probe_identity", json=probe_payload)
        if "audio_chunk" in res.json():
            audio_len = len(base64.b64decode(res.json()["audio_chunk"]))
            print(f"✅ Probe Success: Received {audio_len} bytes of synthesized audio.")
        else:
            print(f"❌ Probe Failed: {res.text}")

        # 5. Reset
        print("\n[STEP 5] Resetting Engine to Defaults...")
        requests.post(f"{URL}/debug/instrument", json={"mel_mean": -15.0})
        print("✅ Done.")

    except Exception as e:
        print(f"❌ Test Failed: {e}")

if __name__ == "__main__":
    run_alignment_test()
