"""
GLM-4-Voice bridge.

Connects the Discord voice intake pipeline to the GLM-4-Voice inference server.
Responsibilities:
  - Transcription (Whisper, for diagnostics / echo mode)
  - Audio segment accumulation and turn assembly
  - Streaming response handling from the GLM server
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import pathlib
from typing import Any

import aiohttp
import discord
import numpy as np
import logging

from .base import BridgeBase
from .orchestrator import orchestrator, Colors
from .components import AudioResampler, WhisperTokenizer, GLMAudioDecoder

logger = logging.getLogger("bridge.glm")

# ---------------------------------------------------------------------------
# Feature flags
# ---------------------------------------------------------------------------

# Set True to skip all model HTTP calls and just echo prompts back to voice.
# Flip to False to re-enable real GLM inference.
PROMPT_ECHO_MODE: bool = os.getenv("PROMPT_ECHO_MODE", "true").lower() == "true"

# Set True to play raw intake bytes back into Discord with zero processing.
# Use this to hear exactly what the intake layer captured, before Whisper or GLM.
RAW_ECHO_MODE: bool = os.getenv("RAW_ECHO_MODE", "false").lower() == "true"

# ---------------------------------------------------------------------------
# Transcriber
# ---------------------------------------------------------------------------

class Transcriber:
    """
    Lazy-loaded Whisper wrapper.

    Tries faster-whisper first (float16 CUDA), falls back to openai-whisper.
    Call transcribe() directly — model loads on first use.
    """

    MODEL_NAME: str = os.environ.get("WHISPER_MODEL", "small.en")

    def __init__(self):
        self._model   = None
        self._backend = None   # 'faster' | 'openai' | 'unavailable'

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transcribe(self, stereo_pcm: bytes) -> str:
        """
        [DISABLED] Transcribe raw stereo int16 48 kHz PCM bytes to English text.
        """
        return "[transcription disabled]"
        # self._ensure_loaded()
        # if self._backend == "unavailable" or self._model is None:
        #     return "[transcription unavailable]"
        # try:
        #     audio_16k = self._prepare_audio(stereo_pcm)
        #     return self._run_model(audio_16k)
        # except Exception as e:
        #     return f"[transcription error: {e}]"

    def warmup(self) -> str:
        """Transcribe a silent clip to force model initialisation onto GPU."""
        # Stereo int16 silence: 0.5 s × 48000 Hz × 2 channels × 2 bytes
        silence = np.zeros(int(48000 * 0.5 * 2), dtype=np.int16).tobytes()
        return self.transcribe(silence)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_loaded(self):
        if self._backend is not None:
            return
        if self._try_load_faster():
            return
        if self._try_load_openai():
            return
        self._backend = "unavailable"
        print(
            f"{Colors.YELLOW}[Transcriber] No whisper library found. "
            f"Install with: pip install faster-whisper{Colors.RESET}"
        )

    def _try_load_faster(self) -> bool:
        try:
            from faster_whisper import WhisperModel
            self._model   = WhisperModel(self.MODEL_NAME, device="cuda", compute_type="float16")
            self._backend = "faster"
            print(f"{Colors.GREEN}[Transcriber] faster-whisper '{self.MODEL_NAME}' ready (CUDA float16).{Colors.RESET}")
            return True
        except ImportError:
            return False

    def _try_load_openai(self) -> bool:
        try:
            import whisper as _ow
            self._model   = self._load_openai_with_recovery(_ow)
            self._backend = "openai"
            print(f"{Colors.GREEN}[Transcriber] openai-whisper '{self.MODEL_NAME}' ready (CUDA).{Colors.RESET}")
            return True
        except ImportError:
            return False

    def _load_openai_with_recovery(self, _ow):
        """Load model, clearing corrupt cache on checksum failure."""
        def _load():
            return _ow.load_model(self.MODEL_NAME, device="cuda")
        try:
            return _load()
        except RuntimeError as e:
            if "SHA256" not in str(e) and "checksum" not in str(e):
                raise
            cache = pathlib.Path.home() / ".cache" / "whisper" / f"{self.MODEL_NAME}.pt"
            if cache.exists():
                os.remove(cache)
                print(f"{Colors.YELLOW}[Transcriber] Corrupt cache cleared — re-downloading {self.MODEL_NAME}...{Colors.RESET}")
            return _load()

    @staticmethod
    def _prepare_audio(stereo_pcm: bytes) -> np.ndarray:
        """Stereo int16 48 kHz → left-channel float32 16 kHz (Whisper format)."""
        import torch
        import torchaudio.functional as F_audio
        left   = np.frombuffer(stereo_pcm, dtype=np.int16)[0::2].astype(np.float32) / 32768.0
        tensor = torch.from_numpy(left).unsqueeze(0)
        return F_audio.resample(tensor, 48000, 16000).squeeze(0).numpy()

    def _run_model(self, audio_16k: np.ndarray) -> str:
        if self._backend == "faster":
            segs, _ = self._model.transcribe(
                audio_16k,
                beam_size=5,
                language="en",
                condition_on_previous_text=False,
            )
            return " ".join(s.text for s in segs).strip() or "[silence]"
        else:  # openai-whisper
            result = self._model.transcribe(
                audio_16k,
                fp16=True,
                language="en",
                condition_on_previous_text=False,
            )
            return result.get("text", "").strip() or "[silence]"


# Standard transcriber replaced by WhisperTokenizer, but kept as fallback via components.py
_transcriber = Transcriber()


# ---------------------------------------------------------------------------
# GLMBridge
# ---------------------------------------------------------------------------

class GLMBridge(BridgeBase):
    """
    Bridge for THUDM/GLM-4-Voice.

    Process loop:
      1. _gather_audio_segments  — waits for one turn, merges trailing dispatches
      2. orchestrator.assemble_payload — wraps turn in full message history
      3. ECHO: _echo_prompt      — prints + plays back without inference
         LIVE: _post_to_server   — streams response from GLM server
    """

    # Turn boundary logic is strictly managed by StreamingSink.
    # When a segment arrives in the queue, the turn is considered complete.

    def __init__(self, voice_preset: str = None, text_prompt: str = None, audio_source=None):
        super().__init__("glm-4", 22050)
        self.url          = f"http://127.0.0.1:10000/generate_complex_stream"
        self.audio_source = audio_source
        self.session      = None
        self.audio_queue  = asyncio.Queue()
        self.upsample_queue = asyncio.Queue(maxsize=20)
        self.is_running   = False
        self.is_processing = False
        self.is_awake     = False
        self.ding_pcm     = b""
        self.processing_ding_pcm = b""
        self.vc           = None   # Set by VoiceServiceCog after joining
        self.text_channel = None   # Set by VoiceServiceCog after joining
        
        # Audio Pipeline Components
        self.resampler = AudioResampler()
        self.tokenizer = WhisperTokenizer()
        self.decoder = GLMAudioDecoder()

        # System prompt management
        self.base_system_prompt = (
            "User will provide you with a speech instruction. Do it step by step. "
            "First, think about the instruction and provide a brief plan, "
            "then follow the instruction and respond directly to the user."
        )
        if text_prompt:
            self.system_prompt = f"{self.base_system_prompt}\n\nPersona Instructions: {text_prompt}"
        else:
            self.system_prompt = self.base_system_prompt

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=None),
            read_bufsize=1024 * 1024,
        )
        self.is_running = True
        await self._warmup()

    async def close(self):
        self.is_running = False
        if self.session:
            await self.session.close()

    async def start_streaming(self):
        asyncio.create_task(self._process_loop())
        asyncio.create_task(self._pipelined_upsampler())

    async def _pipelined_upsampler(self):
        """Background task to upsample audio chunks without blocking the main stream loop."""
        loop = asyncio.get_event_loop()
        while self.is_running:
            try:
                audio_22k = await self.upsample_queue.get()
                if audio_22k is None:
                    continue
                
                # Resample in thread pool to avoid blocking event loop
                audio_48k = await loop.run_in_executor(None, self.resampler.upsample_bytes, audio_22k)
                
                if self.audio_source:
                    self.audio_source.feed(audio_48k)
                
                self.upsample_queue.task_done()
            except Exception as e:
                logger.error(f"[GLMBridge] Upsampler error: {e}")
                await asyncio.sleep(0.1)

    async def send_audio_packet(self, data: Any):
        if self.is_running:
            # Immediate local drop when a turn is finalized by the AudioEngine
            if self.vc:
                try:
                    if hasattr(self.vc, 'sink') and hasattr(self.vc.sink, 'is_listening'):
                        self.vc.sink.is_listening = False
                    # We no longer call change_voice_state(self_deaf=True) here to avoid Discord API ratelimits
                except Exception as e:
                    logger.warning(f"[GLMBridge] Failed to set is_listening: {e}")
                    
            # REJECT queueing if there is already a prompt waiting or being processed
            if self.is_processing or not self.audio_queue.empty():
                logger.warning("[GLMBridge] Dropping overlapping audio turn: pipeline is busy.")
                return
                
            await self.audio_queue.put(data)

    async def play_ding(self):
        """Play the wake word confirmation sound. Called by the Orchestrator."""
        if hasattr(self, 'ding_pcm') and self.ding_pcm and getattr(self, 'audio_source', None):
            self.audio_source.feed_raw(self.ding_pcm)

    # ------------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------------

    async def _warmup(self):
        """Pre-load Whisper and prime the GLM server concurrently."""
        print(f"{Colors.CYAN}[GLMBridge] Warmup started — pre-loading models...{Colors.RESET}")
        results = await asyncio.gather(
            self._warmup_whisper(),
            self._warmup_glm_server(),
            self._warmup_pipeline_components(),
            return_exceptions=True,
        )
        for r in results:
            if isinstance(r, Exception):
                logger.warning(f"[GLMBridge] Warmup subtask failed (bot still works): {r}")
                
        # Load Wake Word Ding
        try:
            source = discord.FFmpegPCMAudio("src/assets/wake_word_ding.mp3")
            while True:
                chunk = source.read()
                if not chunk:
                    break
                self.ding_pcm += chunk
            source.cleanup()
            logger.info(f"[GLMBridge] Loaded Wake Word Ding ({len(self.ding_pcm)} bytes)")
        except Exception as e:
            logger.warning(f"[GLMBridge] Failed to load Wake Word Ding: {e}")
            
        # Load Processing Ding
        try:
            source = discord.FFmpegPCMAudio("src/assets/processing_ding.mp3")
            while True:
                chunk = source.read()
                if not chunk:
                    break
                self.processing_ding_pcm += chunk
            source.cleanup()
            logger.info(f"[GLMBridge] Loaded Processing Ding ({len(self.processing_ding_pcm)} bytes)")
        except Exception as e:
            # Fallback to wake_word_ding if processing_ding is missing
            self.processing_ding_pcm = self.ding_pcm
            logger.warning(f"[GLMBridge] Failed to load Processing Ding (using fallback): {e}")
            
        if not any(isinstance(r, Exception) for r in results):
            logger.info("[GLMBridge] Warmup complete — bot is hot and ready.")

    async def _warmup_whisper(self):
        loop   = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, _transcriber.warmup)
        logger.info(f"[GLMBridge] Whisper warm (test: '{result}').")

    async def _warmup_pipeline_components(self):
        """Warmup the new tokenizer and decoder components."""
        loop = asyncio.get_running_loop()
        await asyncio.gather(
            loop.run_in_executor(None, self.tokenizer.warmup),
            loop.run_in_executor(None, self.decoder.warmup)
        )

    async def _warmup_glm_server(self):
        if PROMPT_ECHO_MODE:
            logger.info("[GLMBridge] GLM warmup skipped (PROMPT_ECHO_MODE).")
            return
        try:
            payload = [{"role": "system", "content": "Warmup."}, {"role": "user", "content": "Hi"}]
            async with self.session.post(self.url, json=payload) as resp:
                async for _ in resp.content:
                    pass
            logger.info("[GLMBridge] GLM server warmed.")
        except Exception as e:
            logger.warning(f"[GLMBridge] GLM server warmup failed: {e}")

    # ------------------------------------------------------------------
    # Process loop
    # ------------------------------------------------------------------

    async def _process_loop(self):
        while self.is_running:
            try:
                incoming = await self.audio_queue.get()
                if not incoming:
                    continue

                self.is_processing = True
                
                segments = incoming if isinstance(incoming, list) else [incoming]
                
                # Each segment['audio'] is now raw PCM bytes (from AudioEngine)
                # We base64 encode them for the GLM server.
                for seg in segments:
                    if isinstance(seg.get('audio'), (bytes, bytearray)):
                        seg['audio'] = base64.b64encode(seg['audio']).decode('utf-8')

                # 2. Build payload with local system prompt
                messages = []
                messages.append({"role": "system", "content": self.system_prompt})
                ## TODO: figure out how to do history without the bot looping
                #messages.extend(orchestrator.history[-orchestrator.history_limit:])
                
                user_turn = {
                    "role": "user",
                    "content": "",
                    "audio_tokens": None,
                    "audio_segments": segments
                }
                messages.append(user_turn)

                did_deafen = False
                try:
                    if RAW_ECHO_MODE:
                        await self._raw_echo(segments)
                    elif PROMPT_ECHO_MODE:
                        await self._echo_prompt(segments, messages)
                    else:
                        # Live Inference: Tokenize audio locally before sending
                        loop = asyncio.get_event_loop()

                        # Batch all segments into one large processing pass to reduce GPU/Thread overhead
                        raw_parts = [base64.b64decode(seg["audio"]) for seg in segments]
                        full_pcm_48k = b"".join(raw_parts)
                        
                        audio_16k = await loop.run_in_executor(None, self.resampler.downsample, full_pcm_48k)
                        features = await loop.run_in_executor(None, self.tokenizer.extract_features, audio_16k)
                        all_tokens = await loop.run_in_executor(None, self.tokenizer.get_vq_tokens, features)
                            
                        # GUARD: Skip turn if too few audio tokens were extracted (noise/mic blip)
                        if len(all_tokens) < 15:
                            logger.info(f"[GLMBridge] Skipping micro-turn (noise/too short: {len(all_tokens)} tokens).")
                            continue

                        # We have a valid prompt! Deafen via WebSocket to protect the connection from CryptoErrors during heavy GPU load.
                        if self.vc:
                            try:
                                await self.vc.guild.change_voice_state(channel=self.vc.channel, self_deaf=True)
                                did_deafen = True
                            except Exception as e:
                                logger.warning(f"[GLMBridge] Failed to deafen: {e}")

                        # If this turn was flagged as a voice clone reference by the Orchestrator
                        if incoming.get('is_clone_reference', False):
                            # Send the raw 48kHz stereo int16 PCM directly to the server.
                            # The server will do a single GPU resample to both 22050Hz (Mel)
                            # and 16kHz (Whisper VQ tokens), avoiding a double resample.
                            pcm_b64 = base64.b64encode(full_pcm_48k).decode()

                            # Derive a short, safe slug from the Discord username
                            raw_name = incoming.get('username', 'user')
                            profile_name = raw_name.split('#')[0][:16].lower().replace(' ', '_')

                            async with aiohttp.ClientSession() as clone_session:
                                base_url = self.url.rsplit('/', 1)[0]
                                async with clone_session.post(
                                    f"{base_url}/clone_reference",
                                    json={"pcm_b64": pcm_b64, "name": profile_name, "sample_rate": 48000}
                                ) as response:
                                    await response.read()
                            logger.info(f"[GLMBridge] Voice reference for '{profile_name}' sent to server.")
                            if self.text_channel:
                                asyncio.create_task(self.text_channel.send(
                                    f"🎙️ Voice cloned as **{profile_name}**! I'll respond in your voice."
                                ))
                            continue

                        # If here, it's a normal AWAKE prompt. Server handles voice cloning
                        # implicitly via the cached prompt_feat in the vocoder — no token manipulation needed.
                        # Replace raw audio segments with VQ tokens in the payload
                        if messages[-1]["role"] == "user":
                            messages[-1]["audio_tokens"] = all_tokens
                            messages[-1]["audio_segments"] = []

                        # Disable listening IMMEDIATELY after prompt is captured
                        self.is_awake = False
                        
                        # Play the processing ding after "going asleep" but before server processing
                        if self.processing_ding_pcm and self.audio_source:
                            self.audio_source.feed_raw(self.processing_ding_pcm)

                        await self._post_to_server(messages)

                finally:
                    # Wait for audio playback to finish before undeafening
                    if self.vc and self.audio_source and hasattr(self.audio_source, 'is_playing'):
                        while self.audio_source.is_playing:
                            await asyncio.sleep(0.1)

                    # Always undeafen after processing (even on error)
                    if self.vc:
                        try:
                            if did_deafen:
                                await self.vc.guild.change_voice_state(channel=self.vc.channel, self_deaf=False)
                            if hasattr(self.vc, 'sink') and hasattr(self.vc.sink, 'is_listening'):
                                self.vc.sink.is_listening = True
                        except Exception as e:
                            logger.warning(f"[GLMBridge] Failed to undeafen: {e}")
                            
                    self.is_processing = False

            except Exception as e:
                logger.error(f"[GLMBridge] Process loop error — retrying: {e}")
                await asyncio.sleep(1)

    # ------------------------------------------------------------------
    # Raw echo mode — pipe intake bytes straight back to Discord
    # ------------------------------------------------------------------

    async def _raw_echo(self, segments: list):
        """
        Play back the audio exactly as the tokenizer would see it.

        Pipeline (mirrors engine.py process_segment exactly):
          1. Decode raw 48 kHz stereo int16 bytes
          2. Extract left channel → float32
          3. Resample 48 kHz → 16 kHz  (this is what extract_speech_token receives)
          4. Upsample back 16 kHz → 48 kHz stereo int16 for Discord playback

        What you hear = what the tokenizer hears.
        """
        import torch
        import torchaudio.functional as F_audio

        if not self.vc or not self.vc.is_connected():
            logger.warning("[RAW_ECHO] No voice client — cannot play back.")
            return
        
        if self.audio_source:
            self.audio_source.clear()

        tasks = []
        for seg in segments:
            # seg['audio'] is base64 encoded raw PCM bytes (48kHz stereo)
            raw = base64.b64decode(seg['audio'])
            dur = seg.get('duration_s', len(raw) / 192000)

            logger.info(f"[RAW_ECHO] Playing raw engine output ({dur:.2f}s)")

            if self.audio_source and hasattr(self.audio_source, 'feed_raw'):
                self.audio_source.feed_raw(raw)
            else:
                # Fallback (slower)
                source = discord.PCMAudio(io.BytesIO(raw))
                tasks.append(self._play_fallback(source))
        
        if tasks:
            await asyncio.gather(*tasks)

    async def _play_fallback(self, source):
        while self.vc.is_playing():
            await asyncio.sleep(0.05)
        self.vc.play(source)


    # ------------------------------------------------------------------
    # Echo mode (diagnostic)
    # ------------------------------------------------------------------

    async def _echo_prompt(self, segments: list, payload: list):
        """Execute the full audio tokenizer and detokenizer pipeline without invoking LLM."""
        loop = asyncio.get_event_loop()

        logger.info("[PROMPT ECHO MODE] PIPELINE START")
        
        if self.audio_source:
            self.audio_source.clear()

        for idx, seg in enumerate(segments):
            pcm_data = base64.b64decode(seg["audio"])
            logger.info(f"[Segment {idx}] Raw PCM bytes: {len(pcm_data)}B")
            
            # 1. Downsample (CPU bound, run in executor)
            audio_16k = await loop.run_in_executor(None, self.resampler.downsample, pcm_data)
            logger.info(f" -> Downsampled to 16kHz float32: shape {audio_16k.shape}")

            # 2. Tokenize (Whisper GPU)
            features = await loop.run_in_executor(None, self.tokenizer.extract_features, audio_16k)
            tokens = await loop.run_in_executor(None, self.tokenizer.get_vq_tokens, features)
            logger.info(f" -> Tokens extracted: {len(tokens)} tokens")

            # 3. Transcribe (Whisper GPU)
            transcription = await loop.run_in_executor(None, self.tokenizer.transcribe, audio_16k)
            logger.info(f" -> Transcription: {transcription}")

            # 4. Prepare Tokens for Detokenizer
            token_tensor = await loop.run_in_executor(None, self.decoder.prepare_tokens, tokens)
            
            # 5. Generate Flow Mel Spectrogram
            mel = await loop.run_in_executor(None, self.decoder.generate_flow_mel, token_tensor)
            
            # 6. Vocode to Waveform (22kHz mono)
            audio_22k = await loop.run_in_executor(None, self.decoder.vocode_to_waveform, mel)
            logger.info(f" -> Synthesized audio: {audio_22k.shape}")

            # 7. Upsample to 48kHz stereo (CPU bound)
            audio_48k = await loop.run_in_executor(None, self.resampler.upsample, audio_22k)
            logger.info(f" -> Upsampled final PCM: {len(audio_48k)}B")

            # 8. Play output
            await self._play_audio(audio_48k)
            logger.info(" -> Dispatched to VoiceClient")

        logger.info("[PROMPT ECHO MODE] PIPELINE END")

    async def _play_audio(self, pcm: bytes):
        """Play raw stereo int16 Discord PCM through the voice channel."""
        if self.vc is None:
            return
            
        if self.audio_source and hasattr(self.audio_source, 'feed_raw'):
            self.audio_source.feed_raw(pcm)
        else:
            # Fallback
            while self.vc.is_playing():
                await asyncio.sleep(0.05)
            self.vc.play(discord.PCMAudio(io.BytesIO(pcm)))

    # ------------------------------------------------------------------
    # Live inference
    # ------------------------------------------------------------------

    async def _post_to_server(self, payload: list):
        async with self.session.post(self.url, json=payload) as response:
            if response.status != 200:
                logger.error(f"[GLMBridge] Server returned HTTP {response.status} — skipping turn.")
                return
            await self._handle_server_stream(response)

    async def _handle_server_stream(self, response: aiohttp.ClientResponse):
        logger.info("┌─── STREAMING RESPONSE ───────────────────────────────")
        if self.audio_source:
            self.audio_source.prepare_for_model()
        
        full_text         = ""
        total_audio_bytes = 0
        audio_tokens      = []

        async for line in response.content:
            if not line:
                continue
            try:
                chunk = json.loads(line)
                full_text, total_audio_bytes = await self._dispatch_chunk(
                    chunk, full_text, total_audio_bytes, audio_tokens
                )
            except (json.JSONDecodeError, KeyError):
                pass  # Non-JSON stream delimiters are expected

        # Strip internal model tags from the saved history
        clean_text = full_text.replace("♪", "").replace("streaming_transcription", "").strip()
        if not clean_text and total_audio_bytes:
            clean_text = "[Audio Response]"
        elif not clean_text:
            clean_text = "[No response]"
            
        orchestrator.add_assistant_response(clean_text, audio_tokens=audio_tokens)
        
        # Send text response to Discord
        if self.text_channel and clean_text and clean_text not in ("[Audio Response]", "[No response]"):
            # Run in task to avoid blocking the stream cleanup
            asyncio.create_task(self.text_channel.send(f"**Stupid-Bot:** {clean_text}"))
            
        logger.info(
            f"Turn complete: {total_audio_bytes}B audio, "
            f"{len(full_text)} chars. History: {len(orchestrator.history)} messages."
        )
        # Add newline to console if we printed text
        if full_text:
             print()
        logger.info("└──────────────────────────────────────────────────────")

    async def _dispatch_chunk(self, chunk: dict, full_text: str, total_audio_bytes: int, audio_tokens: list):
        """Route a single NDJSON chunk to the appropriate handler. Returns updated text/byte totals."""
        if "audio_chunk" in chunk:
            audio_22k = base64.b64decode(chunk["audio_chunk"])
            
            # Pipeline the upsampling to avoid blocking the network read loop
            await self.upsample_queue.put(audio_22k)
            
            # Estimate upsampled size for logging (22k mono -> 48k stereo)
            total_audio_bytes += int(len(audio_22k) * (48000 / 22050) * 2)
        elif "audio_tokens" in chunk:
            audio_tokens.extend(chunk["audio_tokens"])
        elif "text_chunk" in chunk:
            text = chunk["text_chunk"]
            # Filter internal model tags
            if any(tag in text for tag in ("streaming_transcription", "♪")):
                return full_text, total_audio_bytes
            full_text += text
            print(text, end="", flush=True)
        elif "error" in chunk:
            print(f"\n{Colors.RED}[GLMBridge] Model error: {chunk['error']}{Colors.RESET}")
        return full_text, total_audio_bytes
