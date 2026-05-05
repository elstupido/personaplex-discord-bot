"""
The Soundscape Architect.

WHY THIS FILE EXISTS:
A bot without sound effects is just a silent void. This module manages the 
'Acoustic Cues' (dings, glitches, and chimes) that provide the user with 
subconscious feedback about the bot's internal state.
"""

import os
import numpy as np
from .constants import logger

async def load_asset_pcm(path: str, volume: float = 1.0) -> bytes:
    """
    Loads an MP3/WAV and prepares it for Discord's 48kHz Stereo Reality.
    
    WHY TORCHAUDIO?
    Because standard libraries are slow. We use torchaudio to handle the 
    resampling and volume scaling in one clean, GPU-ready sweep (even if 
    we usually stay on CPU for assets).
    """
    import torchaudio
    try:
        audio, sr = torchaudio.load(path)
        
        # WHY VOLUME SCALING?
        # Because 'Response Ready' sounds shouldn't cause hearing loss.
        if volume != 1.0:
            audio = audio * volume
            
        # WHY ENSURE STEREO?
        # Discord's PCMAudio expects interleaved stereo. If we feed it 
        # mono, the listener's left ear gets lonely.
        if audio.shape[0] == 1:
            audio = audio.repeat(2, 1)
            
        # WHY 48000Hz?
        # That is the hard-coded law of Discord voice. Anything else 
        # sounds like Alvin and the Chipmunks.
        if sr != 48000:
            import torchaudio.functional as F
            audio = F.resample(audio, sr, 48000)
            
        # Convert to int16 bytes (Discord's preferred format)
        return (audio.T.numpy() * 32767).astype(np.int16).tobytes()
        
    except Exception as e:
        logger.error(f"[GLMBridge.Assets] Failed to load {path}: {e}")
        return b""
