"""
WHY THIS FILE EXISTS:
The Voice Output Conduit (The Audio Source).

WHY:
Discord requires audio to be delivered in precise 20ms frames at 48kHz stereo. 
The PersonaPlexAudioSource acts as a 'Time-Aware Buffer' that:
1. Receives variable-length chunks from the AI Bridge.
2. Slices them into exactly 3840-byte frames.
3. Manages a 'Pre-Roll' buffer to prevent network jitter from causing stutters.
4. Signals SpeakingState.voice/none by pausing/resuming the VoiceClient.
"""

import discord
import collections
import threading
import time
from core.logger import setup_logger

logger = setup_logger("voice.source")

DISCORD_FRAME_BYTES = 3840 # 20ms stereo int16 at 48kHz

class PersonaPlexAudioSource(discord.AudioSource):
    """
    Receives mono PCM, resamples/formats it, and feeds Discord's audio loop.
    """
    def __init__(self):
        self.frame_buffer = collections.deque(maxlen=5000) # 100 seconds
        self.lock = threading.Lock()
        self.pre_roll_count = 10 
        self.is_playing = False
        
        self._last_feed_time = 0
        self._first_frame_read = False
        self._vc = None
        self._loop = None
        self._discord_speaking = False

    def attach_vc(self, vc, loop):
        """Wire up the voice client for speaking state control."""
        self._vc = vc
        self._loop = loop

    def _discord_pause(self):
        """Pause the VoiceClient to stop UDP sends."""
        if self._vc is None or self._discord_speaking is False:
            return
        self._discord_speaking = False
        try:
            if self._vc.is_playing():
                self._vc.pause()
                logger.debug("[AudioSource] Paused — SpeakingState.none")
        except Exception as e:
            logger.debug(f"[AudioSource] pause failed: {e}")

    def _discord_resume(self):
        """Resume the VoiceClient to start UDP sends."""
        if self._vc is None or self._discord_speaking is True:
            return
        self._discord_speaking = True
        try:
            if self._vc.is_paused():
                self._vc.resume()
                logger.debug("[AudioSource] Resumed — SpeakingState.voice")
        except Exception as e:
            logger.debug(f"[AudioSource] resume failed: {e}")

    def clear(self):
        """Flush the playback buffer."""
        with self.lock:
            self.frame_buffer.clear()
            self.is_playing = False
            self._first_frame_read = False
        self._discord_pause()

    def prepare_for_model(self):
        """Force pre-roll for the next model response."""
        with self.lock:
            self.is_playing = False
            self._first_frame_read = False

    def feed(self, pcm_48k_stereo: bytes):
        """Feed 48kHz Stereo PCM to be buffered/played."""
        with self.lock:
            if not self.frame_buffer:
                self._last_feed_time = time.time()
                self._first_frame_read = False
            
            self._enqueue_frames(pcm_48k_stereo)
            
            if not self.is_playing and len(self.frame_buffer) >= self.pre_roll_count:
                self.is_playing = True
                self._discord_resume()

    def feed_raw(self, pcm_48k_stereo: bytes):
        """Feed 48kHz stereo directly (instant playback)."""
        with self.lock:
            if not self.frame_buffer:
                self._last_feed_time = time.time()
                self._first_frame_read = False
            
            self._enqueue_frames(pcm_48k_stereo)
            
            if not self.is_playing:
                self.is_playing = True
                self._discord_resume()

    def _enqueue_frames(self, pcm: bytes):
        """Slice and buffer PCM frames."""
        mv = memoryview(pcm)
        for i in range(0, len(pcm), DISCORD_FRAME_BYTES):
            frame = mv[i:i + DISCORD_FRAME_BYTES]
            if len(frame) == DISCORD_FRAME_BYTES:
                self.frame_buffer.append(frame.tobytes())

    def read(self) -> bytes:
        """Called every 20ms by the AudioPlayer."""
        try:
            frame = self.frame_buffer.popleft()
            if not self._first_frame_read:
                self._first_frame_read = True
                logger.debug(f"[AudioSource] [PLAY_START] Buffer size: {len(self.frame_buffer)}")
            return frame
        except IndexError:
            with self.lock:
                self.is_playing = False
            self._discord_pause()
            return b'\x00' * DISCORD_FRAME_BYTES

    def is_opus(self):
        return False

    def cleanup(self):
        with self.lock:
            self.frame_buffer.clear()
            self.is_playing = False
