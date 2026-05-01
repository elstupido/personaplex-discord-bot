import threading
import time
import collections
import numpy as np
from typing import Callable, Optional
from utils.logger import setup_logger

logger = setup_logger("TriggerEngine")

class TriggerEngine:
    """
    Model-agnostic streaming wake word and trigger engine.
    Listens to raw audio chunks, maintains a rolling window, and transcribes
    in a background thread to detect trigger words instantly.
    """
    def __init__(self, wake_word: str, clone_word: str, on_trigger: Callable[[str], None]):
        self.wake_word = wake_word.lower()
        self.clone_word = clone_word.lower()
        self.on_trigger = on_trigger

        self.is_running = False
        
        # Audio buffer (rolling window)
        self.buffer_lock = threading.Lock()
        # 4 seconds of audio max at 48kHz stereo int16 (4 seconds * 48000 * 2 * 2 = 768000 bytes)
        self.max_buffer_bytes = 4 * 48000 * 2 * 2
        self.audio_buffer = bytearray()
        
        self._transcriber = None
        self._thread = None
        self._last_transcribe_time = 0

    def start(self):
        self.is_running = True
        self._thread = threading.Thread(target=self._transcribe_loop, daemon=True, name="TriggerEngine")
        self._thread.start()
        logger.info(f"[TriggerEngine] Started. Listening for: '{self.wake_word}', '{self.clone_word}'")

    def warmup(self):
        """
        Pre-load the transcription model and run a dummy inference to warm CUDA kernels.
        Call this during bot startup so the first real trigger fires instantly.
        """
        logger.info("[TriggerEngine] Warming up transcriber...")
        self._ensure_transcriber()
        if self._transcriber is None:
            return
        try:
            silence = np.zeros(16000, dtype=np.float32)  # 1s of silence at 16kHz
            if getattr(self, '_backend', None) == "faster":
                list(self._transcriber.transcribe(silence, language="en", beam_size=1, without_timestamps=True)[0])
            elif getattr(self, '_backend', None) == "openai":
                self._transcriber.transcribe(silence, language="en", fp16=True, temperature=0.0)
            logger.info("[TriggerEngine] Transcriber warm and ready.")
        except Exception as e:
            logger.warning(f"[TriggerEngine] Warmup inference failed (non-fatal): {e}")

    def stop(self):
        self.is_running = False
        if self._thread:
            self._thread.join(timeout=2)

    def feed(self, pcm_data: bytes):
        """Called frequently with small chunks of raw 48kHz stereo int16 PCM."""
        if not self.is_running:
            return

        with self.buffer_lock:
            self.audio_buffer.extend(pcm_data)
            # Truncate to max sliding window size (keep the most recent audio)
            if len(self.audio_buffer) > self.max_buffer_bytes:
                overflow = len(self.audio_buffer) - self.max_buffer_bytes
                # Ensure we truncate on a frame boundary (4 bytes per stereo int16 frame)
                overflow = overflow - (overflow % 4)
                self.audio_buffer = self.audio_buffer[overflow:]

    def clear(self):
        """Clears the current buffer (called after a trigger)."""
        with self.buffer_lock:
            self.audio_buffer.clear()

    def _ensure_transcriber(self):
        if self._transcriber is not None:
            return
        
        try:
            from faster_whisper import WhisperModel
            import logging
            logging.getLogger("faster_whisper").setLevel(logging.WARNING)
            logger.info("[TriggerEngine] Loading faster-whisper (tiny.en) for streaming wake word detection...")
            self._transcriber = WhisperModel("tiny.en", device="cuda", compute_type="float16")
            self._backend = "faster"
        except ImportError:
            try:
                import whisper
                logger.info("[TriggerEngine] Loading openai-whisper (tiny.en) for streaming wake word detection...")
                self._transcriber = whisper.load_model("tiny.en", device="cuda")
                self._backend = "openai"
            except Exception as e:
                logger.error(f"[TriggerEngine] Failed to load transcriber: {e}")
                self.is_running = False
        except Exception as e:
            logger.error(f"[TriggerEngine] Failed to load transcriber: {e}")
            self.is_running = False

    def _transcribe_loop(self):
        """Background thread that periodically transcribes the rolling window."""
        self._ensure_transcriber()
        if not self._transcriber:
            return

        # How often to run transcription on the rolling buffer
        interval = 0.5 

        while self.is_running:
            time.sleep(0.1) # Check frequently
            now = time.time()
            if now - self._last_transcribe_time < interval:
                continue

            with self.buffer_lock:
                current_bytes = bytes(self.audio_buffer)

            # We need at least 0.5s of audio to bother transcribing
            if len(current_bytes) < 48000 * 2 * 2 * 0.5:
                continue

            self._last_transcribe_time = now
            
            try:
                # Downsample 48kHz stereo int16 -> 16kHz mono float32 for Whisper
                pcm_float = np.frombuffer(current_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                left_channel = pcm_float[0::2]
                
                # Simple numpy decimation (1/3) since 48k -> 16k is an integer ratio
                audio_16k = left_channel[::3].copy()

                # Transcribe
                transcript = ""
                if getattr(self, '_backend', None) == "faster":
                    segments, _ = self._transcriber.transcribe(
                        audio_16k, 
                        language="en", 
                        beam_size=1, 
                        vad_filter=True,
                        without_timestamps=True
                    )
                    transcript = " ".join([segment.text for segment in segments]).lower()
                elif getattr(self, '_backend', None) == "openai":
                    result = self._transcriber.transcribe(
                        audio_16k,
                        language="en",
                        fp16=True,
                        task="transcribe",
                        temperature=0.0
                    )
                    transcript = result.get("text", "").lower()
                
                cleaned_trans = ''.join(c for c in transcript if c.isalnum() or c.isspace()).strip()
                
                if cleaned_trans:
                    # logger.debug(f"[TriggerEngine] Hear: '{cleaned_trans}'")
                    
                    if self.clone_word in cleaned_trans:
                        logger.info(f"[TriggerEngine] 🔥 Clone word detected: '{self.clone_word}'")
                        self.clear()
                        self.on_trigger("CLONE")
                    elif self.wake_word in cleaned_trans:
                        logger.info(f"[TriggerEngine] 🔥 Wake word detected: '{self.wake_word}'")
                        self.clear()
                        self.on_trigger("WAKE")

            except Exception as e:
                logger.error(f"[TriggerEngine] Transcription error: {e}")
