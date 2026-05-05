import requests
import json

URL = "http://127.0.0.1:10000"

# CALIBRATED PARAMETERS FOR GLM-4-VOICE (LOG10)
# Found via /debug/extract_reference on a real 16k PCM sample
OPTIMAL_PARAMS = {
    "mel_scale": "log10",
    "mel_mean": -2.76,
    "mel_std": 0.86,
    "emb_scalar": 1.0,  # CosyVoice uses raw embeddings
    "audio_offset": 152353,
    "true_vocab_size": 168960
}

def apply():
    print(f"🚀 APPLYING OPTIMAL CLONING PARAMETERS...")
    resp = requests.post(f"{URL}/debug/instrument", json=OPTIMAL_PARAMS)
    if resp.status_code == 200:
        print("✅ SUCCESS: Engine re-aligned to zero-mean identity manifold.")
        print(json.dumps(OPTIMAL_PARAMS, indent=2))
    else:
        print(f"❌ FAILED: {resp.text}")

if __name__ == "__main__":
    apply()
