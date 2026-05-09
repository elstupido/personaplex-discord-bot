"""
WHY THIS FILE EXISTS:
Standalone ASR WebSocket Server (Bot-Ears) powered by FunASR & SenseVoice.

WHY:
Decoupling ASR from vLLM-Omni allows 100% of the VRAM in the brain container to be 
dedicated to the TTS engine. This server accepts raw audio streams, downsamples 
them to 16kHz natively on the GPU, and returns the transcribed text.
"""
import asyncio
import websockets
import json
import urllib.parse
import numpy as np
import io
import soundfile as sf
import torch
import torchaudio
import re
import os
import http
from funasr import AutoModel

# WHY: We must unset LD_PRELOAD here so that child processes (like pip/uname)
# don't try to load torch-python libraries and crash. Torch has already 
# loaded these into the current process memory at startup, so we are safe.
if "LD_PRELOAD" in os.environ:
    print(f"🧬 [bot-ears] Internalizing surgery: unsetting LD_PRELOAD for child processes...")
    del os.environ["LD_PRELOAD"]

# WHY: Target the specific snapshot in the volume-mounted cache to bypass slow downloads.
local_model_path = "/app/model_cache/hub/models--FunAudioLLM--SenseVoiceSmall/snapshots/3eb3b4eeffc2f2dde6051b853983753db33e35c3"

print("👂 [bot-ears] Loading SenseVoiceSmall model from local cache on CUDA...")
model = AutoModel(
    model=local_model_path, 
    trust_remote_code=False, 
    device="cuda:0",
    disable_update=True
)
print("👂 [bot-ears] Model loaded successfully.")

# Cache resamplers to avoid recreating them constantly
resamplers = {}

def get_resampler(orig_freq, new_freq):
    key = (orig_freq, new_freq)
    if key not in resamplers:
        resamplers[key] = torchaudio.transforms.Resample(orig_freq=orig_freq, new_freq=new_freq).to("cuda:0")
    return resamplers[key]

def clean_sensevoice_text(text: str) -> str:
    """Removes the <|zh|><|NEUTRAL|><|Speech|> style prefix tags returned by SenseVoice."""
    # SenseVoice returns tags like <|en|><|NEUTRAL|><|Speech|> Hello there.
    cleaned = re.sub(r'<\|.*?\|>', '', text).strip()
    return cleaned

async def health_check(connection, request):
    if request.path == "/health":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None

async def handle_client(websocket):
    # WHY: In websockets 13.0+, the path is part of the request object on the websocket.
    path = websocket.request.path
    # Parse query parameters (e.g. ?format=pcm&rate=48000)
    query = urllib.parse.urlparse(path).query
    params = urllib.parse.parse_qs(query)
    
    fmt = params.get('format', ['wav'])[0].lower()
    rate = int(params.get('rate', ['48000'])[0])
    
    print(f"👂 [bot-ears] Client connected. Format: {fmt}, Rate: {rate}Hz")
    
    try:
        async for message in websocket:
            if isinstance(message, bytes):
                # 1. Parse Audio Data
                if fmt == 'wav':
                    try:
                        wav_io = io.BytesIO(message)
                        audio_data, sample_rate = sf.read(wav_io)
                        # soundfile returns (samples, channels) - we want (channels, samples)
                        if len(audio_data.shape) > 1:
                            audio_data = audio_data.T
                        waveform = torch.from_numpy(audio_data).float()
                        if len(waveform.shape) == 1:
                            waveform = waveform.unsqueeze(0)
                        waveform = waveform.to("cuda:0")
                    except Exception as e:
                        print(f"⚠️ [bot-ears] Failed to parse WAV: {e}")
                        await websocket.send(json.dumps({"error": "Invalid WAV data"}))
                        continue
                elif fmt == 'pcm':
                    # Raw PCM int16
                    pcm_int16 = np.frombuffer(message, dtype=np.int16)
                    # Convert to float32 [-1.0, 1.0]
                    waveform = torch.from_numpy(pcm_int16).float().unsqueeze(0) / 32768.0
                    waveform = waveform.to("cuda:0")
                    sample_rate = rate
                else:
                    await websocket.send(json.dumps({"error": f"Unsupported format: {fmt}"}))
                    continue

                # 2. Resample to 16kHz for SenseVoice
                if sample_rate != 16000:
                    resampler = get_resampler(sample_rate, 16000)
                    waveform = resampler(waveform)
                
                # Convert back to CPU numpy array for FunASR
                audio_array = waveform.squeeze(0).cpu().numpy()
                
                # 3. Transcribe
                try:
                    res = model.generate(input=audio_array, language="auto", use_itn=True)
                    text = res[0].get("text", "")
                    
                    text = clean_sensevoice_text(text)
                    print(f"👂 [bot-ears] Transcribed: '{text}'")
                    
                    response = {
                        "text": text
                    }
                    await websocket.send(json.dumps(response))
                except Exception as e:
                    print(f"💥 [bot-ears] Inference failed: {e}")
                    await websocket.send(json.dumps({"error": str(e)}))
                    
    except websockets.exceptions.ConnectionClosed:
        print("👂 [bot-ears] Client disconnected.")
    except Exception as e:
        print(f"💥 [bot-ears] Connection error: {e}")

async def main():
    print("👂 [bot-ears] WebSocket server starting on ws://0.0.0.0:8765")
    async with websockets.serve(handle_client, "0.0.0.0", 8765, process_request=health_check):
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())
