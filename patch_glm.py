import os
import subprocess

PATCH_CODE = r'''
import base64
import numpy as np
import torch
import torchaudio
import io
import uuid
import re
from speech_tokenizer.modeling_whisper import WhisperVQEncoder
from transformers import WhisperFeatureExtractor
from flow_inference import AudioDecoder
from audio_process import AudioStreamProcessor
from speech_tokenizer.utils import extract_speech_token

# Load additional models for Audio support
print("[SmartServer] Initializing Audio Tokenizer and Decoder...")
DEVICE = "cuda"
TOKENIZER_PATH = "THUDM/glm-4-voice-tokenizer"
DECODER_PATH = "./glm-4-voice-decoder"

whisper_model = WhisperVQEncoder.from_pretrained(TOKENIZER_PATH).eval().to(DEVICE)
feature_extractor = WhisperFeatureExtractor.from_pretrained(TOKENIZER_PATH)
audio_decoder = AudioDecoder(
    config_path=os.path.join(DECODER_PATH, "config.yaml"),
    flow_ckpt_path=os.path.join(DECODER_PATH, "flow.pt"),
    hift_ckpt_path=os.path.join(DECODER_PATH, "hift.pt"),
    device=DEVICE
)

@app.post("/generate_audio_stream")
async def generate_audio_stream(request: Request):
    params = await request.json()
    raw_audio_b64 = params.get("audio")
    
    # 1. Decode incoming audio
    audio_bytes = base64.b64decode(raw_audio_b64)
    # Discord is 48k, Whisper wants 16k
    audio_tensor, orig_sr = torchaudio.load(io.BytesIO(audio_bytes))
    if orig_sr != 16000:
        audio_tensor = torchaudio.transforms.Resample(orig_sr, 16000)(audio_tensor)
    
    # Save to temp for extract_speech_token (it expects a path)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        torchaudio.save(tmp.name, audio_tensor, 16000)
        audio_tokens = extract_speech_token(whisper_model, feature_extractor, [tmp.name])[0]
    
    if len(audio_tokens) == 0:
        return {"error": "No speech detected"}

    # 2. Format Prompt
    audio_str = "".join([f"<|audio_{x}|>" for x in audio_tokens])
    prompt = f"<|user|>\n<|begin_of_audio|>{audio_str}<|end_of_audio|><|assistant|>streaming_transcription\n"
    
    # 3. Generate and Stream Back
    async def audio_generator():
        gen_params = {
            "prompt": prompt,
            "temperature": params.get("temperature", 0.2),
            "top_p": params.get("top_p", 0.8),
            "max_new_tokens": params.get("max_new_tokens", 2000)
        }
        
        audio_offset = worker.glm_tokenizer.convert_tokens_to_ids('<|audio_0|>')
        this_uuid = str(uuid.uuid4())
        audio_tokens_buffer = []
        block_size = 50
        
        # Internal generation loop
        for chunk_bytes in worker.generate_stream_gate(gen_params):
            data = json.loads(chunk_bytes.decode())
            token_id = data["token_id"]
            
            if token_id >= audio_offset:
                audio_tokens_buffer.append(token_id - audio_offset)
            
            if len(audio_tokens_buffer) >= block_size:
                tts_token = torch.tensor(audio_tokens_buffer, device=DEVICE).unsqueeze(0)
                tts_speech, _ = audio_decoder.token2wav(tts_token, uuid=this_uuid, finalize=False)
                
                # Convert to PCM16 base64
                pcm = (tts_speech.clamp(-1, 1).cpu().numpy() * 32767).astype(np.int16)
                yield (json.dumps({"audio_chunk": base64.b64encode(pcm.tobytes()).decode()}) + "\n").encode()
                audio_tokens_buffer = []
        
        # Finalize
        if audio_tokens_buffer:
            tts_token = torch.tensor(audio_tokens_buffer, device=DEVICE).unsqueeze(0)
            tts_speech, _ = audio_decoder.token2wav(tts_token, uuid=this_uuid, finalize=True)
            pcm = (tts_speech.clamp(-1, 1).cpu().numpy() * 32767).astype(np.int16)
            yield (json.dumps({"audio_chunk": base64.b64encode(pcm.tobytes()).decode()}) + "\n").encode()

    return StreamingResponse(audio_generator())
'''

def patch():
    # 1. Get the original file content
    print("[patch] Reading model_server.py from container...")
    try:
        content = subprocess.check_output(["docker", "exec", "ai-voice-bot", "cat", "/app/glm/model_server.py"]).decode()
    except:
        # If the container name is different (e.g. from docker-compose)
        content = subprocess.check_output(["docker", "exec", "personaplex-discord-bot-bot-1", "cat", "/app/glm/model_server.py"]).decode()

    # 2. Inject imports and endpoint
    if "generate_audio_stream" in content:
        print("[patch] Already patched!")
        return

    # Find the line before 'if __name__ == "__main__":'
    insertion_point = content.find('if __name__ == "__main__":')
    new_content = content[:insertion_point] + PATCH_CODE + content[insertion_point:]

    # 3. Write it back
    print("[patch] Writing patched version back to container...")
    with open("temp_server.py", "w") as f:
        f.write(new_content)
    
    try:
        subprocess.run(["docker", "cp", "temp_server.py", "ai-voice-bot:/app/glm/model_server.py"], check=True)
    except:
        subprocess.run(["docker", "cp", "temp_server.py", "personaplex-discord-bot-bot-1:/app/glm/model_server.py"], check=True)
    
    os.remove("temp_server.py")
    print("[patch] Success! Please restart the container with 'docker compose up bot'")

if __name__ == "__main__":
    patch()
