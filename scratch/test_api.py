import requests
import base64
import numpy as np

SERVER_URL = "http://127.0.0.1:10000"

def test_connectivity():
    try:
        # 1. Test Health/Config
        resp = requests.get(f"{SERVER_URL}/debug/config")
        print(f"Server Config: {resp.json()}")
        
        # 2. Test Probe Endpoint with minimal data
        dummy_pcm = base64.b64encode(np.zeros(16000, dtype=np.int16).tobytes()).decode()
        payload = {
            "pcm_b64": dummy_pcm,
            "target_tokens": [1, 2, 3],
            "alignment_params": {"mel_mean": -4.0}
        }
        resp = requests.post(f"{SERVER_URL}/debug/probe_identity", json=payload)
        print(f"Probe Status: {resp.status_code}")
        if resp.status_code == 200:
            print("Successfully reached probe endpoint.")
        else:
            print(f"Error: {resp.text}")
            
    except Exception as e:
        print(f"Failed to reach server: {e}")

if __name__ == "__main__":
    test_connectivity()
