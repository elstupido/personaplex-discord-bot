import os
import requests
import base64
import collections
import queue
import threading
import time
import numpy as np
import torch
from abc import ABC, abstractmethod
from typing import Optional, List, Tuple

from core.logger import setup_logger

logger = setup_logger("voice.trigger")

# --- BACKEND ABSTRACTION ---

class TranscriptionBackend(ABC):
    """
    Abstract base class for transcription engines.
    
    WHY: This allows us to hot-swap between different acoustic models (Whisper, 
    SenseVoice, etc.) without the TriggerEngine having to know about their 
    internal neural architectures or dependency hell.
    """
    @abstractmethod
    def transcribe(self, audio_data: np.ndarray) -> str:
        """Transcribe PCM audio data (expected 16kHz float32)."""
        pass

    @abstractmethod
    def warmup(self):
        """Warm up the model to avoid first-inference latency."""
        pass

class WhisperBackend(TranscriptionBackend):
    """
    The Traditional Reliable: Faster-Whisper.
    
    WHY: It's fast, efficient, and well-understood. We use the 'tiny.en' 
    model by default for hotword detection because we only care about 
    identifying a few specific tokens, not high-fidelity prose.
    """
    def __init__(self, model_size="tiny.en", device="cuda", compute_type="float16"):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.model = None

    def _ensure_model(self):
        if self.model is None:
            from faster_whisper import WhisperModel
            logger.info(f"[WhisperBackend] Loading {self.model_size} on {self.device} ({self.compute_type})...")
            self.model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)
            logger.info("[WhisperBackend] Model hot.")

    def transcribe(self, audio_data: np.ndarray) -> str:
        self._ensure_model()
        # Faster-Whisper expects float32 in [-1, 1] at 16kHz
        segments, _ = self.model.transcribe(audio_data, beam_size=1, language="en")
        return " ".join([s.text for s in segments]).strip()

    def warmup(self):
        self._ensure_model()
        # Feed silence to prime the engine
        silence = np.zeros(16000, dtype=np.float32)
        self.transcribe(silence)

class RemoteSenseVoiceBackend(TranscriptionBackend):
    """
    The 'Big Fish': Remote SenseVoiceSmall (Fish Audio S2).
    
    WHY: SenseVoice is state-of-the-art for high-concurrency ASR. By offloading 
    to the GLM server, we avoid VRAM duplication in the bot process.
    """
    def __init__(self, server_url=None):
        self.server_url = server_url or os.getenv("GLM_SERVER_URL", "http://127.0.0.1:10000")

    def transcribe(self, audio_data: np.ndarray) -> str:
        """
        Offloads ASR to the 'Inference Gateway'.
        
        WHY POST? For a 2-second buffer, the payload is small enough that 
        HTTP overhead is negligible compared to the 5090's inference speed.
        """
        
        # Convert float32 [-1, 1] to int16 for transport
        pcm_int16 = (audio_data * 32767).astype(np.int16)
        pcm_b64 = base64.b64encode(pcm_int16.tobytes()).decode()
        
        try:
            resp = requests.post(
                f"{self.server_url}/transcribe", 
                json={"audio_b64": pcm_b64},
                timeout=2.0
            )
            if resp.status_code == 200:
                return resp.json().get("text", "")
        except Exception as e:
            logger.error(f"[SenseVoice] Remote ASR failed: {e}")
        return ""

    def warmup(self):
        # We assume the server handles its own warmup.
        pass

# --- THE TRIGGER ENGINE ---

class TriggerEngine:
    """
    The 'Watchdog' of Vocal Intent.
    
    WHY THIS CLASS EXISTS:
    Humans don't speak in discrete packets. We speak in continuous streams. 
    The TriggerEngine maintains a 'Rolling Audio Buffer' (Sliding Window) to 
    constantly monitor the audio for specific 'Hotwords' (Wake/Clone).
    
    WHY THE 16KHZ DOWNSAMPLING?
    Discord audio is 48kHz. Most ASR models (Whisper, SenseVoice) are 
    trained on 16kHz. By downsampling early, we reduce the computational 
    load by 3x without losing the frequency data needed for speech recognition.
    """
    def __init__(self, wake_word="alexa", clone_word="clone", on_trigger=None):
        self.wake_word = wake_word.lower()
        self.clone_word = clone_word.lower()
        self.on_trigger = on_trigger # callback(type, text)
        
        # Audio Pipeline
        self.sample_rate = 16000
        self.window_size_sec = 3.0  # 3 seconds of context
        self.stride_sec = 0.5       # Transcribe every 0.5 seconds
        
        # High-Performance Ring Buffer (No 'cray cray' list copying)
        # WHY: Deques are great for O(1) appends, but converting them to 
        # NumPy for model inference requires a full memory traversal and 
        # reallocation. A pre-allocated NumPy array with a write pointer 
        # allows us to treat memory as a contiguous stream.
        self.buffer_capacity = int(5 * self.sample_rate)
        self.buffer = np.zeros(self.buffer_capacity, dtype=np.float32)
        self.write_ptr = 0
        self.buffer_full = False
        self.buffer_lock = threading.Lock()
        
        # Backend State
        self._backend: Optional[TranscriptionBackend] = None
        self._backend_type = "whisper" # Default
        
        # Threading
        self._stop_event = threading.Event()
        self._loop_thread = None

    def set_backend(self, backend_type: str):
        """Switch the ASR engine (whisper/sensevoice)."""
        if backend_type == self._backend_type and self._backend:
            return
        
        logger.info(f"✨ [TriggerEngine] Leveling up to: **{backend_type.upper()}**")
        
        backends = {
            "whisper": WhisperBackend,
            "sensevoice": RemoteSenseVoiceBackend
        }
        
        if backend_type not in backends:
            raise ValueError(f"Unknown backend: {backend_type}. Try being less creative. ✨")
            
        self._backend = backends[backend_type]()
        self._backend_type = backend_type
        threading.Thread(target=self._backend.warmup, daemon=True).start()

    def feed(self, pcm_bytes: bytes):
        """
        Accepts raw 48kHz 16-bit PCM bytes from the Discord AudioEngine.
        
        WHY THE STRIDING?
        We downsample 48k -> 16k by skipping samples (stride 3). This is 
        crude but extremely fast. For hotword detection, this spectral 
        aliasing is negligible.
        """
        # Convert bytes to int16
        data = np.frombuffer(pcm_bytes, dtype=np.int16)
        # Stereo to Mono if needed (Discord usually sends stereo)
        if len(data) % 2 == 0:
            data = data[0::2]
        
        # Downsample 48k -> 16k (every 3rd sample)
        downsampled = data[::3].astype(np.float32) / 32767.0
        
        with self.buffer_lock:
            # Vectorized push into the ring
            n = len(downsampled)
            if n > self.buffer_capacity:
                downsampled = downsampled[-self.buffer_capacity:]
                n = self.buffer_capacity

            end = self.write_ptr + n
            if end <= self.buffer_capacity:
                self.buffer[self.write_ptr:end] = downsampled
            else:
                # Wrap around logic
                first_part = self.buffer_capacity - self.write_ptr
                self.buffer[self.write_ptr:] = downsampled[:first_part]
                self.buffer[:n - first_part] = downsampled[first_part:]
                self.buffer_full = True
            
            self.write_ptr = (self.write_ptr + n) % self.buffer_capacity
            if not self.buffer_full and end >= self.buffer_capacity:
                self.buffer_full = True

    def start(self):
        if self._loop_thread and self._loop_thread.is_alive():
            return
            
        if not self._backend:
            self.set_backend("whisper")
            
        self._stop_event.clear()
        self._loop_thread = threading.Thread(target=self._transcribe_loop, daemon=True)
        self._loop_thread.start()
        logger.info("[TriggerEngine] Watchdog active.")

    def stop(self):
        self._stop_event.set()
        if self._loop_thread:
            self._loop_thread.join(timeout=1.0)
        logger.info("[TriggerEngine] Watchdog dormant.")

    def _transcribe_loop(self):
        """
        The Sliding Window Inference Loop.
        
        WHY THE RMS CHECK?
        Transcription is expensive (even on a 5090). We only run the model if 
        the RMS (volume) of the buffer exceeds a threshold. This filters out 
        background noise and silence.
        """
        last_run = time.time()
        
        while not self._stop_event.is_set():
            now = time.time()
            if now - last_run < self.stride_sec:
                time.sleep(0.05)
                continue
                
            # Extract window
            with self.buffer_lock:
                samples_needed = int(self.window_size_sec * self.sample_rate)
                
                # SHINY: Instead of if-else hell, we use np.take with mode='wrap' 
                # to extract a contiguous window from our ring buffer indices.
                # WHY: It's mathematically elegant. We calculate the indices 
                # we want, and NumPy handles the memory wrapping for us.
                if not self.buffer_full and self.write_ptr < samples_needed:
                    continue
                
                indices = np.arange(self.write_ptr - samples_needed, self.write_ptr)
                window_np = np.take(self.buffer, indices, mode='wrap')

            # RMS Noise Gate
            rms = np.sqrt(np.mean(window_np**2))
            if rms < 0.005: # Too quiet
                last_run = now
                continue

            # Inference
            try:
                transcript = self._backend.transcribe(window_np).lower()
                if transcript:
                    self._process_transcript(transcript)
            except Exception as e:
                logger.error(f"[TriggerEngine] Loop error: {e}")
                
            last_run = now

    def _process_transcript(self, text: str):
        """
        The Shiny Matcher.
        
        WHY: We use a list comprehension to identify active triggers. 
        It's cleaner than a chain of if-statements and allows for 
        multi-hotword detection if we ever decide to go really wild.
        """
        if not text: return
        
        logger.debug(f"🎙️ [Engine] Hear: '{text}'")
        
        triggers = {"wake": self.wake_word, "clone": self.clone_word}
        matched = [k for k, v in triggers.items() if v in text]
        
        for t_type in matched:
            matched_word = triggers[t_type]
            logger.info(f"✨ [TRIGGER] '{matched_word.upper()}' detected! (Full: '{text}')")
            if self.on_trigger: self.on_trigger(t_type, text)
            
            # Reset buffer to prevent 'Echo Triggers'
            with self.buffer_lock:
                self.buffer.fill(0)
                self.write_ptr = 0
                self.buffer_full = False
            break # Only process one trigger per window
