"""
WHY THIS FILE EXISTS:
The Voice Activation Engine. Orchestrates audio ingestion, resampling, 
and circular buffering for the neural KWS backend.

WHY:
Discord audio is 48kHz stereo; we need 16kHz mono for the neural brain.
This engine handles that transformation in real-time.
"""
import threading
import time
import numpy as np
from voice.kws import SherpaKWSBackend
from core.logger import setup_logger

logger = setup_logger("voice.trigger")

class TriggerEngine:
    """
    The Main Event Loop for Voice Activation.
    Handles circular buffering, resampling, and backend dispatch.
    """
    def __init__(self, wake_word="hey stupid", clone_word="clone my voice", on_trigger=None):
        self.on_trigger = on_trigger
        self.wake_word, self.clone_word = wake_word, clone_word
        self.backend = SherpaKWSBackend()
        
        # Audio Buffer: 2 seconds @ 16kHz
        self.sample_rate = 16000
        self.capacity = 32000
        self.buffer = np.zeros(self.capacity, dtype=np.float32)
        self.write_ptr = 0
        self._buffer_lock = threading.Lock()
        
        self.is_running = False
        self._thread = None

    def start(self):
        if self.is_running: return
        self.is_running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info(f"👂 [TriggerEngine] Listening for '{self.wake_word}'...")

    def feed(self, pcm_bytes: bytes):
        """Ingests 48kHz stereo bytes, downsamples to 16kHz, and stores in buffer."""
        data = np.frombuffer(pcm_bytes, dtype=np.int16)
        if len(data) == 0: return

        # WHY: Discord sends 48kHz STEREO (interleaved L, R).
        # We MUST downmix to mono before resampling, otherwise linear interpolation
        # will blend L and R channels together and create acoustic garbage.
        stereo_data = data.reshape(-1, 2)
        mono_data = stereo_data.mean(axis=1)  # Average L and R

        # WHY: Fast Downsampling (48k -> 16k)
        # Linear interpolation to prevent aliasing.
        x_orig = np.arange(len(mono_data))
        x_target = np.linspace(0, len(mono_data) - 1, len(mono_data) // 3)
        downsampled = np.interp(x_target, x_orig, mono_data).astype(np.float32) / 32768.0

        with self._buffer_lock:
            n = len(downsampled)
            # Circular write
            if self.write_ptr + n <= self.capacity:
                self.buffer[self.write_ptr:self.write_ptr+n] = downsampled
            else:
                first = self.capacity - self.write_ptr
                self.buffer[self.write_ptr:] = downsampled[:first]
                self.buffer[:n-first] = downsampled[first:]
            self.write_ptr = (self.write_ptr + n) % self.capacity

    def _run_loop(self):
        """Continuous inference loop extracting 512-sample chunks."""
        read_ptr = 0
        chunk_size = 512
        
        while self.is_running:
            available = (self.write_ptr - read_ptr) % self.capacity
            if available < chunk_size:
                time.sleep(0.01)
                continue

            # Extract chunk
            if read_ptr + chunk_size <= self.capacity:
                chunk = self.buffer[read_ptr:read_ptr+chunk_size]
            else:
                first = self.capacity - read_ptr
                chunk = np.concatenate([self.buffer[read_ptr:], self.buffer[:chunk_size-first]])
            
            read_ptr = (read_ptr + chunk_size) % self.capacity
            
            # Dispatch to Neural Brain
            trigger = self.backend.detect(chunk)
            if trigger and self.on_trigger:
                self.on_trigger("wake", self.wake_word)

    def warmup(self):
        """Pre-initializes the neural models."""
        self.backend.detect(np.zeros(512, dtype=np.float32))
