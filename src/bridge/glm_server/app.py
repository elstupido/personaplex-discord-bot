import os
import uuid
import json
import base64
import torch
import numpy as np
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from threading import Thread, Lock

from .config import Colors, DECODER_SAMPLE_RATE
from .engine import GLMVoiceEngine
from .streamer import TokenQueueStreamer, VocabGuardProcessor

from utils.logger import setup_logger

logger = setup_logger("bridge.glm_server.app")

app = FastAPI()
server_app = None

class StreamContext:
    """Encapsulates the state and processing logic for a single streaming request."""
    def __init__(self, engine: GLMVoiceEngine, prompt_len, ref_token, ref_feat):
        self.engine = engine
        self.prompt_len = prompt_len
        self.ref_token = ref_token
        self.ref_feat = ref_feat
        self._dev = engine.device
        
        # State
        self.audio_buffer = []
        self.generated_audio_tokens = []
        self.tts_mels = []
        self.prev_mel = None
        self.processed_count = 0
        self.size_idx = 0
        self.block_sizes = [50, 100, 150]
        self.session_uuid = str(uuid.uuid4())
        
        # Resolution logic
        user_tokens = self.engine.glm_tokenizer.convert_tokens_to_ids("<|user|>")
        self.end_token_id = user_tokens if isinstance(user_tokens, int) else 151337

    def handle_token(self, token_id):
        """Processes a single token from the streamer and returns a JSON chunk if ready."""
        self.processed_count += 1
        
        # 1. Ignore prompt echo
        if self.processed_count <= self.prompt_len:
            return None
            
        # 2. Check for termination
        if token_id == self.end_token_id:
            return "STOP"
            
        # 3. Route Token
        if token_id < self.engine.audio_offset:
            # Text Chunk
            text = self.engine.glm_tokenizer.decode([token_id])
            return json.dumps({"text_chunk": text}) if text else None
            
        elif self.engine.audio_offset <= token_id < self.engine.true_vocab_size:
            # Audio Token
            raw_tok = token_id - self.engine.audio_offset
            self.audio_buffer.append(raw_tok)
            self.generated_audio_tokens.append(raw_tok)
            
            # Check if we should decode a block
            if len(self.audio_buffer) >= self.block_sizes[self.size_idx]:
                return self.decode_block(finalize=False)
                
        return None

    def decode_block(self, finalize=False):
        """Prepares conditioning and decodes the current audio buffer."""
        if not self.audio_buffer:
            return None
            
        # 1. Update block size for progressive latency
        if not finalize and self.size_idx < len(self.block_sizes) - 1:
            self.size_idx += 1
            
        # 2. Build Conditioning (Reference Anchoring)
        prompt_token, prompt_feat = self._build_conditioning()
        
        # 3. Vocode to PCM
        audio_payload, tts_mel = self._vocode(
            self.audio_buffer, prompt_token, prompt_feat, finalize
        )
        
        # 4. Update History for Continuity
        if tts_mel is not None:
            self.tts_mels.append(tts_mel.transpose(1, 2))
            self.prev_mel = tts_mel
            
        self.audio_buffer = []
        return audio_payload

    def _build_conditioning(self):
        """Constructs the prompt features/tokens, anchored ONLY to the original reference."""
        # We stop appending generated history here because the decoder appears to be 
        # re-synthesizing the prompt content, causing the "doubling" effect.
        
        prompt_feat = self.ref_feat if self.ref_feat is not None \
                      else torch.zeros(1, 0, 80, device=self._dev)
        
        prompt_token = self.ref_token if self.ref_token is not None \
                       else torch.zeros(1, 0, dtype=torch.int32, device=self._dev)
            
        return prompt_token, prompt_feat

    def _vocode(self, buffer, prompt_token, prompt_feat, finalize):
        """Low-level call to the flow decoder."""
        # Generating a fresh UUID for every block to prevent the decoder 
        block_uuid = str(uuid.uuid4())
        tts_token = torch.tensor(buffer, device=self._dev, dtype=torch.int32).unsqueeze(0)
        try:
            with torch.inference_mode():
                tts_speech, tts_mel = self.engine.audio_decoder.token2wav(
                    tts_token, uuid=block_uuid,
                    prompt_token=prompt_token,
                    prompt_feat=prompt_feat,
                    finalize=finalize
                )
            pcm = (tts_speech.clamp(-1, 1).cpu().numpy() * 32767).astype(np.int16)
            pcm_b64 = base64.b64encode(pcm.tobytes()).decode()
            return json.dumps({"audio_chunk": pcm_b64}), tts_mel
        except Exception as e:
            logger.error(f"Decoding Error: {e}")
            return None, None


class GLMServerApp:
    def __init__(self, engine: GLMVoiceEngine):
        self.engine = engine

    def stream_response(self, messages):
        """Main generator for the streaming API response."""
        with self.engine.lock:
            # 1. Model Input Preparation
            inputs_dict = self.engine.prepare_input(messages)
            input_ids = inputs_dict["input_ids"]
            prompt_len = input_ids.shape[1]
            
            # 2. Pipeline Infrastructure
            streamer = TokenQueueStreamer(prompt_len=prompt_len)
            vocab_guard = VocabGuardProcessor(self.engine.true_vocab_size)
            
            # 3. Kick off generation in background
            self._start_generation(inputs_dict, input_ids, streamer, vocab_guard)
            
            # 4. Context Initialization
            ctx = StreamContext(
                self.engine, 
                prompt_len, 
                self.engine.ref_prompt_token, 
                self.engine.ref_prompt_feat
            )
            
            logger.info(f"Streaming response (Prompt: {prompt_len} tokens)...")
            
            # 5. Token Processing Loop
            for token_id in streamer:
                result = ctx.handle_token(token_id)
                
                if result == "STOP":
                    break
                elif result:
                    yield (result + "\n").encode()
            
            # 6. Final Flush
            final_audio = ctx.decode_block(finalize=True)
            if final_audio:
                yield (final_audio + "\n").encode()
                
            # 7. Response Summary (for history caching)
            generated_only = streamer.all_tokens[streamer.prompt_len:]
            yield (json.dumps({"audio_tokens": generated_only}) + "\n").encode()
            
            logger.info("Generation Complete.")

    def _start_generation(self, inputs_dict, input_ids, streamer, vocab_guard):
        """Launches the LLM generation in a dedicated thread with safety locks."""
        generate_kwargs = {
            **inputs_dict,
            "attention_mask": torch.ones_like(input_ids),
            "max_new_tokens": 512,
            "temperature": 0.8,
            "top_p": 1.0,
            "streamer": streamer,
            "pad_token_id": self.engine.glm_tokenizer.eos_token_id,
            "logits_processor": [vocab_guard]
        }
        
        def safe_generate():
            with self.engine.gen_lock: # Prevent concurrent GPU access
                with torch.inference_mode():
                    self.engine.glm_model.generate(**generate_kwargs)
                    
        Thread(target=safe_generate).start()


# --- FastAPI Routes ---

@app.post("/tokenize_audio")
async def tokenize_route(request: Request):
    payload = await request.json()
    tokens = []
    for segment in payload.get("segments", []):
        seg_tokens = server_app.engine.process_segment(segment)
        if seg_tokens: tokens.extend(seg_tokens)
    return {"tokens": tokens}

@app.post("/clone_reference")
async def clone_reference_route(request: Request):
    payload = await request.json()
    name = payload.get("name", "default")
    sample_rate = payload.get("sample_rate", 16000)
    server_app.engine.set_voice_reference(
        pcm_b64=payload["pcm_b64"],
        name=name,
        sample_rate=sample_rate,
    )
    return {"status": "ok", "name": name}

@app.post("/load_voice_profile")
async def load_voice_profile_route(request: Request):
    payload = await request.json()
    ok = server_app.engine.load_voice_profile(payload["name"])
    return {"status": "ok" if ok else "not_found"}

@app.post("/clear_reference")
async def clear_reference_route():
    server_app.engine.clear_voice_reference()
    return {"status": "ok"}

@app.post("/generate_complex_stream")
async def generate_route(request: Request):
    payload = await request.json()
    return StreamingResponse(server_app.stream_response(payload))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10000)
    parser.add_argument("--model-path", type=str, default="THUDM/glm-4-voice-9b")
    parser.add_argument("--tokenizer-path", type=str, default="THUDM/glm-4-voice-tokenizer")
    parser.add_argument("--flow-path", type=str, default="/app/glm/glm-4-voice-decoder")
    args = parser.parse_args()

    engine = GLMVoiceEngine(args.model_path, args.tokenizer_path, args.flow_path)
    server_app = GLMServerApp(engine)
    uvicorn.run(app, host=args.host, port=args.port)
