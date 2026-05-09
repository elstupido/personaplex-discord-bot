"""
WHY THIS FILE EXISTS:
The 'Acoustic Truth' generator. We are now testing the dedicated 
/v1/audio/speech endpoint which was discovered in the Brain logs.
"""
import requests
import base64
import os
import struct
import wave
import math
import json

def analyze_health(filepath, transcribed_text="", expected_text=""):
    with wave.open(filepath, 'rb') as w:
        params = w.getparams()
        frames = w.readframes(params.nframes)
        samples = struct.unpack(f"<{len(frames)//2}h", frames)
    duration = len(samples) / params.framerate
    max_val = max(abs(s) for s in samples)
    rms = math.sqrt(sum(s*s for s in samples) / len(samples))
    zcr = sum(1 for i in range(1, len(samples)) if samples[i-1] * samples[i] < 0) / len(samples)
    crest_factor = max_val / rms if rms > 0 else 0
    accuracy = 1.0 if expected_text.replace("[neutral]", "").strip().lower() in transcribed_text.lower() else 0.0
    
    metrics = {
        "status": "HEALTHY" if accuracy > 0.5 else "MUMBLE",
        "duration_s": round(duration, 3),
        "max_amplitude": max_val,
        "rms_energy": round(rms, 2),
        "crest_factor": round(crest_factor, 2),
        "zcr": round(zcr, 4),
        "sample_rate": params.framerate,
        "channels": params.nchannels,
        "transcription": {"expected": expected_text, "observed": transcribed_text, "accuracy": accuracy}
    }
    print(f"\n--- 🧠 [AI-DIAGNOSTIC-BLOCK] ---\n{json.dumps(metrics, indent=2)}\n")

def generate_samples(text="hey stupid"):
    url_speech = "http://localhost:8000/v1/audio/speech"
    url_asr = "http://localhost:8000/v1/chat/completions"
    ref_path = "src/assets/reference_voice.wav"
    
    if not os.path.exists(ref_path):
        print(f"  ❌ ERROR: Missing reference voice at {ref_path}")
        return

    # Load and encode the seed voice
    with open(ref_path, "rb") as f:
        # WHY: vLLM requires a full Data URI for base64 audio
        ref_audio_b64 = "data:audio/wav;base64," + base64.b64encode(f.read()).decode("utf-8")
    
    # WHY: Fish Speech requires ref_audio + ref_text to clone a voice.
    payload = {
        "model": "fishaudio/s2-pro",
        "input": text,
        "voice": "default", 
        "response_format": "pcm",
        "ref_audio": ref_audio_b64,
        "ref_text": "Hello, I am the StupidBot reference voice.",
        "latency": "normal"
    }
    
    print(f"📡 [PATTERN: CLONING_HANDSHAKE]")
    # Mask the massive base64 string for the log
    print_payload = json.loads(json.dumps(payload))
    print_payload["ref_audio"] = f"<BASE64_LEN_{len(ref_audio_b64)}>"
    print(f"  📤 REQUEST: {json.dumps(print_payload, indent=2)}")
    
    try:
        resp = requests.post(url_speech, json=payload, timeout=60)
        resp.raise_for_status()
        raw_bytes = resp.content # Binary PCM output
        
        print(f"  📥 RESPONSE: Received {len(raw_bytes)} bytes of raw binary audio.")
        
        if len(raw_bytes) < 100:
            print(f"  💀 [FAILURE] Header-only or empty response.")
            return

    # --- 👂 [TRANSCRIPTION LOOPBACK] ---
    print(f"  👂 Dispatching to 'bot-ears' (SenseVoice) for verification...")
    
    # WHY: bot-ears expects 16kHz mono WAV. 
    import io
    import asyncio
    import websockets
    
    # Resample 41kHz to 16kHz (crude but effective for testing)
    pcm_data = struct.unpack(f"<{len(raw_bytes)//2}h", raw_bytes)
    resampled_pcm = pcm_data[::2] # Very basic decimation for 44->22 approx, better to just let bot-ears handle header
    
    wav_buf = io.BytesIO()
    with wave.open(wav_buf, 'wb') as bw:
        bw.setnchannels(1)
        bw.setsampwidth(2)
        bw.setframerate(41000) # Keep original rate, bot-ears will resample via torchaudio
        bw.writeframes(raw_bytes)
    
    async def get_transcription():
        try:
            async with websockets.connect("ws://localhost:8765/?format=wav") as ws:
                await ws.send(wav_buf.getvalue())
                resp = await ws.recv()
                return json.loads(resp).get("text", "FAILED_TO_TRANSCRIBE")
        except Exception as e:
            return f"ASR_ERROR: {e}"

    transcribed_text = asyncio.run(get_transcription())
    print(f"  📥 ASR OBSERVED: \"{transcribed_text}\"")

        # --- 💾 [SAVE] ---
        filepath = "src/assets/positives/hey_stupid_final.wav"
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with wave.open(filepath, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(41000)
            wav_file.writeframes(raw_bytes)
        
        print(f"  ✅ Saved: {filepath}")
        analyze_health(filepath, transcribed_text, expected_text=text)

    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"  🔍 RAW ERROR: {e.response.text}")

if __name__ == "__main__":
    generate_samples()
