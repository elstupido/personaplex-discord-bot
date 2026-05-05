"""
The PersonaPlex Inference Gateway (The API Frontier).

WHY THIS FILE EXISTS:
This is the 'Surface' of our AI cluster. It uses FastAPI to expose the 
complex, high-performance neural subsystems (Brain, Vault, Larynx) as 
a standard, streaming HTTP interface. 

WHY FASTAPI?
We need asynchronous handling for high-concurrency voice sessions. FastAPI 
allows us to stream audio chunks back to the client the MOMENT they are 
synthesized, reducing the perceived 'Latency to First Word' to sub-second levels.

WHY STREAMING?
Waiting for a 10-second response to be fully generated before playing it 
would feel like a walkie-talkie from the 1980s. Streaming allows the 
AI and the human to stay in 'Temporal Sync'.
"""
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

from .config import Colors, DECODER_SAMPLE_RATE, GLM_MODEL_PATH, GLM_TOKENIZER_PATH, GLM_DECODER_PATH
from .engine import GLMVoiceEngine
from .streamer import TokenQueueStreamer, VocabGuardProcessor
from .diagnostics import AcousticAnalyzer

from core.logger import setup_logger

logger = setup_logger("server.app")

# CONCEPT: THE INFERENCE GATEWAY
# Why FastAPI? Because speech is a time-sensitive signal. 
# We use an asynchronous event loop to handle multiple Discord voice 
# channels simultaneously without one speaker's synthesis blocking another's.
#
# WHY STREAMING?
# Waiting for a 10-second response to be fully generated before playing it 
# would feel like a walkie-talkie from the 1980s. Streaming allows the 
# AI and the human to stay in 'Temporal Sync'.
app = FastAPI()
class PersonaplexServer:
    """
    The Orchestrator of Neural Life.
    
    WHY: This class ties the FastAPI app to the underlying Voice Engine.
    It acts as a singleton that holds the state of the Brain.
    """
    def __init__(self):
        self.engine = GLMVoiceEngine(
            model_path=GLM_MODEL_PATH,
            tokenizer_path=GLM_TOKENIZER_PATH,
            flow_path=GLM_DECODER_PATH
        )

server_app = PersonaplexServer()

class StreamContext:
    """
    The Temporal Anchor for a Single Conversation Turn.
    
    WHY THIS CLASS EXISTS:
    A single API request represents a 'Burst' of interaction. This class 
    manages the 'Identity Entanglement' of that burst—holding the generated 
    tokens, the accumulating audio buffers, and the history needed to 
    ensure the AI doesn't forget what it just said 2 seconds ago.
    
    WHY TURN-FINALIZATION?
    Audio synthesis happens in 'Blocks'. If we just stopped at the last 
    token, the audio might end with a 'click' or a cut-off phoneme. This 
    context manages the 'Finalize' flag, which tells the vocoder to 
    smoothly fade out the acoustic wave once the semantic intent is complete.
    """
    def __init__(self, engine: GLMVoiceEngine, prompt_len, ref_token, ref_feat, ref_embedding, vocoder='glm'):
        self.engine = engine
        self.prompt_len = prompt_len
        self.ref_token = ref_token
        self.ref_feat = ref_feat
        self.ref_embedding = ref_embedding
        self.vocoder_backend = vocoder
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
        self.full_audio_accum = [] # Collect output for fidelity analysis
        
        # Resolution logic
        user_tokens = self.engine.glm_tokenizer.convert_tokens_to_ids("<|user|>")
        self.end_token_id = user_tokens if isinstance(user_tokens, int) else 151337

    def handle_token(self, token_id):
        """Processes a single token from the streamer and returns a JSON chunk if ready."""
        self.processed_count += 1
        if self.processed_count <= self.prompt_len:
            return None
        if token_id == self.end_token_id:
            return "STOP"
        if token_id < self.engine.alignment_config.audio_offset:
            text = self.engine.glm_tokenizer.decode([token_id])
            return json.dumps({"text_chunk": text}) if text else None
        elif self.engine.alignment_config.audio_offset <= token_id < self.engine.alignment_config.true_vocab_size:
            raw_tok = token_id - self.engine.alignment_config.audio_offset
            self.audio_buffer.append(raw_tok)
            self.generated_audio_tokens.append(raw_tok)
            if len(self.audio_buffer) >= self.block_sizes[self.size_idx]:
                return self.decode_block(finalize=False)
        return None

    def decode_block(self, finalize=False):
        """Prepares conditioning and decodes the current audio buffer."""
        if not self.audio_buffer: return None
        if not finalize and self.size_idx < len(self.block_sizes) - 1:
            self.size_idx += 1
        prompt_token, prompt_feat = self._build_conditioning()
        audio_payload, tts_mel = self._vocode(
            self.audio_buffer, prompt_token, prompt_feat, finalize
        )
        if tts_mel is not None:
            self.tts_mels.append(tts_mel.transpose(1, 2))
            self.prev_mel = tts_mel
        self.audio_buffer = []
        return audio_payload

    def _build_conditioning(self):
        """Constructs the prompt features/tokens, anchored to the reference + history."""
        prompt_feat = self.ref_feat if self.ref_feat is not None \
                      else torch.zeros(1, 0, 80, device=self._dev)
        prompt_token = self.ref_token if self.ref_token is not None \
                       else torch.zeros(1, 0, dtype=torch.int32, device=self._dev)

        if self.tts_mels:
            hist_feat = torch.cat(self.tts_mels, dim=1)
            prompt_feat = torch.cat([prompt_feat, hist_feat], dim=1)
            hist_toks = self.generated_audio_tokens[:-len(self.audio_buffer)]
            if hist_toks:
                hist_token_tensor = torch.tensor(hist_toks, device=self._dev, dtype=torch.int32).unsqueeze(0)
                prompt_token = torch.cat([prompt_token, hist_token_tensor], dim=1)
        return prompt_token, prompt_feat

    def _vocode(self, buffer, prompt_token, prompt_feat, finalize):
        """
        The Vocoding Hot-Path.
        
        WHY:
        This is where the discrete tokens generated by the LLM are transformed into 
        audible waveforms. We use a 'VocoderManager' to abstract this process. 
        
        WHY ABSTRACT THE VOCODER?
        The LLM generates 'what' is said (tokens), but the Vocoder decides 'how' 
        it sounds (acoustic quality). By separating them, we can swap out the 
        default Flow+HiFT vocoder for more advanced ones (like Fish Audio) 
        without touching the complex generation logic above. It also allows 
        us to run different 'acoustic laws' for different characters if we wanted.
        
        WHY THE B64 ENCODING?
        Raw PCM bytes are not JSON-serializable. To stream audio back to the 
        Discord bridge over a standard HTTP response, we must 'Crystallize' 
        the binary data into a Base64 string.
        """
        block_uuid = str(uuid.uuid4())
        # Convert the buffer (list of ints) into a GPU-resident Long Tensor.
        # WHY LONG? Because token IDs are indices into the embedding table.
        tts_token = torch.tensor(buffer, device=self._dev, dtype=torch.long).unsqueeze(0)
        
        # Use the reference identity's embedding if available, otherwise silence.
        emb = self.ref_embedding if self.ref_embedding is not None \
              else torch.zeros(1, 192, device=self._dev)
        
        try:
            with torch.inference_mode():
                # We delegate the actual 'heavy lifting' of acoustic reconstruction 
                # to the VocoderManager. We pass it the 'Physical Laws' (conditioning) 
                # so it knows whose voice it is synthesizing.
                tts_speech, tts_mel = self.engine.vocoder_manager.reconstruct(
                    tts_token, 
                    uuid=block_uuid,
                    prompt_token=prompt_token, 
                    prompt_feat=prompt_feat,
                    embedding=emb, 
                    finalize=finalize,
                    backend=self.vocoder_backend
                )
            
            # Convert float32 [-1.0, 1.0] to int16 [-32768, 32767].
            # WHY INT16? This is the industry standard for high-quality audio 
            # transport (WAV/PCM) and what Discord's Opus encoder expects.
            pcm = (tts_speech.clamp(-1, 1).cpu().numpy() * 32767).astype(np.int16)
            
            # We keep a high-resolution float32 copy for our 'Fidelity Analysts'.
            # WHY? Because int16 loses bit-depth information needed for 
            # precise spectral frequency analysis.
            self.full_audio_accum.append(tts_speech.cpu()) 
            
            pcm_b64 = base64.b64encode(pcm.tobytes()).decode()
            return json.dumps({"audio_chunk": pcm_b64}), tts_mel
        except Exception as e:
            logger.error(f"Decoding Error: {e}")
            return None, None

class GLMServerApp:
    def __init__(self, engine: GLMVoiceEngine):
        self.engine = engine

    def stream_response(self, messages, vocoder='glm'):
        """Main generator for the streaming API response."""
        with self.engine.lock:
            inputs_dict = self.engine.prepare_input(messages)
            input_ids = inputs_dict["input_ids"]
            prompt_len = input_ids.shape[1]
            streamer = TokenQueueStreamer(prompt_len=prompt_len)
            vocab_guard = VocabGuardProcessor(self.engine.alignment_config.true_vocab_size)
            self._start_generation(inputs_dict, input_ids, streamer, vocab_guard)
            ctx = StreamContext(
                self.engine, prompt_len, 
                self.engine.ref_prompt_token, 
                self.engine.ref_prompt_feat,
                self.engine.ref_prompt_embedding,
                vocoder=vocoder
            )
            logger.info(f"Streaming response (Prompt: {prompt_len} tokens)...")
            for token_id in streamer:
                result = ctx.handle_token(token_id)
                if result == "STOP": break
                elif result: yield (result + "\n").encode()
            final_audio = ctx.decode_block(finalize=True)
            if final_audio: yield (final_audio + "\n").encode()
            
            # --- POST-TURN FIDELITY ANALYSIS ---
            # If a voice reference is active, we analyze the output in a background 
            # thread to measure 'Hard Metrics' of the clone fidelity.
            if self.engine.alignment_config.enable_diagnostics and \
               self.engine.ref_prompt_embedding is not None and ctx.full_audio_accum:
                self._run_fidelity_analysis(ctx)

            generated_only = streamer.all_tokens[streamer.prompt_len:]
            yield (json.dumps({"audio_tokens": generated_only}) + "\n").encode()
            logger.info("Generation Complete.")

    def _start_generation(self, inputs_dict, input_ids, streamer, vocab_guard):
        generate_kwargs = {
            **inputs_dict, "attention_mask": torch.ones_like(input_ids),
            "max_new_tokens": 512, "temperature": 0.8, "top_p": 1.0,
            "streamer": streamer, "pad_token_id": self.engine.glm_tokenizer.eos_token_id,
            "logits_processor": [vocab_guard]
        }
        def safe_generate():
            with self.engine.gen_lock:
                with torch.inference_mode(): self.engine.glm_model.generate(**generate_kwargs)
        Thread(target=safe_generate).start()

    def _run_fidelity_analysis(self, ctx):
        """Background task to run acoustic diagnostics on the generated turn."""
        def analyze():
            try:
                # 1. Join all audio segments [1, T]
                full_wav = torch.cat(ctx.full_audio_accum, dim=1)
                
                # 2. Resample 22kHz -> 16kHz for CAM++ and analysis
                audio_16k = torchaudio.functional.resample(full_wav, DECODER_SAMPLE_RATE, 16000)
                
                # 3. Trigger Report
                # We pass the stored ref_stats to avoid re-processing the reference audio.
                AcousticAnalyzer.run_full_report(
                    ctx.engine, audio_16k, ctx.ref_embedding, 
                    ref_stats=ctx.engine.ref_stats
                )
                # Special Case: Manual Override for report logging
                # (We already logged the similarity, but we can do more here if needed)
                
            except Exception as e:
                logger.error(f"Fidelity Analysis Error: {e}")

        Thread(target=analyze).start()

# --- FastAPI Routes ---

@app.post("/transcribe")
async def transcribe_route(request: Request):
    """
    The High-Performance ASR Portal.
    
    WHY: Offloads the heavy lifting of speech recognition to the server.
    ✨ [DIAGNOSTIC] Processing remote ASR request...
    """
    payload = await request.json()
    pcm_b64 = payload.get("audio_b64")
    if not pcm_b64:
        return {"error": "Missing audio_b64"}
    
    text = server_app.engine.transcribe(pcm_b64)
    if text:
        logger.debug(f"✨ [ASR] Recognized: '{text}'")
    return {"text": text}

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
    sample_rate = payload.get("sample_rate", 48000)
    prompt_text = payload.get("prompt_text") # Grounding text from transcription
    
    server_app.engine.set_voice_reference(
        pcm_b64=payload["pcm_b64"], 
        name=name, 
        sample_rate=sample_rate,
        prompt_text=prompt_text
    )
    return {"status": "ok", "name": name, "grounded": prompt_text is not None}

@app.post("/load_voice_profile")
async def load_voice_profile_route(request: Request):
    payload = await request.json()
    ok = server_app.engine.load_voice_profile(payload["name"])
    return {"status": "ok" if ok else "not_found"}

@app.get("/list_voices")
async def list_voices_route():
    """
    CONCEPT: The Character Repository
    Scans the local filesystem for all saved 'Digital Souls' (.pt files). 
    This allows a DM to quickly see which NPCs are currently available for use.
    """
    import glob
    files = glob.glob("/app/voice_profiles/*.pt")
    names = [os.path.basename(f).replace(".pt", "") for f in files]
    return {"voices": names}

@app.post("/clear_reference")
async def clear_reference_route():
    server_app.engine.clear_voice_reference()
    return {"status": "ok"}

@app.post("/generate_complex_stream")
async def generate_route(request: Request):
    payload = await request.json()
    
    # WHY: Defensive check. If the client sends a raw list (Legacy Mode), we wrap it.
    if isinstance(payload, list):
        messages = payload
        vocoder = "glm"
    else:
        messages = payload.get("messages", [])
        vocoder = payload.get("vocoder", "glm")
        
    return StreamingResponse(server_app.stream_response(messages, vocoder=vocoder))

# --- Instrumentation & Probing ---

@app.post("/debug/instrument")
async def instrument_route(request: Request):
    payload = await request.json()
    server_app.engine.alignment_config.update(payload)
    logger.info(f"[Instrumentation] Updated config with: {payload}")
    reinit_keys = ['f_min', 'f_max', 'n_mels', 'sample_rate', 'n_fft']
    if any(k in payload for k in reinit_keys):
        server_app.engine._update_mel_basis()
    return {"status": "ok", "current_config": server_app.engine.alignment_config.to_dict()}

@app.post("/debug/extract_reference")
async def extract_reference_route(request: Request):
    payload = await request.json()
    pcm_torch = server_app.engine._decode_pcm(payload["pcm_b64"])
    sr = payload.get("sample_rate", 48000)
    with torch.no_grad():
        # Force raw extraction (ignore current normalization)
        feat = server_app.engine._compute_matcha_mel(pcm_torch, sr)
        cfg = server_app.engine.alignment_config
        raw_feat = feat * (cfg.mel_std + 1e-6) + cfg.mel_mean
        
        emb = server_app.engine._extract_speaker_embedding(pcm_torch, sr)
        tokens = server_app.engine._extract_prompt_tokens(pcm_torch, sr)
    return {
        "tokens": tokens[0].tolist(), "embedding_sum": emb.sum().item(),
        "mel_mean": raw_feat.mean().item(), "mel_std": raw_feat.std().item(), "mel_shape": list(raw_feat.shape)
    }

@app.post("/debug/probe_identity")
async def probe_identity_route(request: Request):
    payload = await request.json()
    pcm_torch = server_app.engine._decode_pcm(payload["pcm_b64"])
    target_tokens = payload.get("target_tokens", [1, 2, 3])
    sr = payload.get("sample_rate", 48000)
    with torch.no_grad():
        feat = server_app.engine._compute_matcha_mel(pcm_torch, sr)
        emb = server_app.engine._extract_speaker_embedding(pcm_torch, sr)
        prompt_token = server_app.engine._extract_prompt_tokens(pcm_torch, sr)
        # If target_tokens is not provided, use prompt_token for self-reconstruction test
        target_tokens = payload.get("target_tokens")
        if target_tokens is None:
            tts_token = prompt_token
        else:
            tts_token = torch.as_tensor(target_tokens, device=server_app.engine.device, dtype=torch.int32).unsqueeze(0)
        tts_speech, _ = server_app.engine.vocoder_manager.reconstruct(
            tts_token, uuid="probe", prompt_token=prompt_token, prompt_feat=feat,
            embedding=emb, finalize=True
        )
        pcm = (tts_speech.clamp(-1, 1).cpu().numpy().flatten() * 32767).astype(np.int16)
        return {
            "audio_pcm_b64": base64.b64encode(pcm.tobytes()).decode(),
            "sample_rate": server_app.engine.alignment_config.sample_rate # or just 22050
        }

@app.post("/warmup")
async def warmup_route():
    """
    Explicit warmup to prime CUDA kernels.
    
    WHY FORCE=TRUE?
    Even if the engine is configured for 'Lazy' loading, calling this endpoint 
    means the operator wants the bot 'Ready to speak'. We force the 
    instantiation of the selected models so there's zero latency on the 
    first word.
    """
    server_app.engine.warmup(force=True)
    return {"status": "warm"}

if __name__ == "__main__":
    # Start the engine process. 
    # Port 10000 is our standard 'Inference Gateway' port.
    uvicorn.run(app, host="0.0.0.0", port=10000)
