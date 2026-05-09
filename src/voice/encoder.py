"""
WHY THIS FILE EXISTS:
Voice encoder for voice cloning: saves Discord PCM as a WAV file.

WHY DO WE JUST SAVE A WAV?
The neural brain (Qwen-Omni/Fish-Speech) handles the complex acoustic embeddings.
By saving a high-fidelity WAV, we maintain the maximum information for the
cloning expert to process when requested.
"""
import asyncio
import logging
import os

import numpy as np
import torch
import torchaudio

logger = logging.getLogger("voice.encoder")

DISCORD_SR = 48000

VOICES_DIR = "src/assets/voices"


def _encode_blocking(pcm_bytes: bytes, output_path: str) -> None:
    """
    Save the user's speech as a WAV file in the Moshi voices directory.
    The server's load_voice_prompt() will encode it properly when the session starts.

    We do NOT try to generate .pt embeddings ourselves — the built-in presets
    contain LM hidden-state embeddings (shape [T, 1, 1, 4096]) which require
    running the full LM encoder, not just mimi.encode().

    Input:  raw 48kHz stereo int16 PCM bytes (from StreamingSink rolling buffer)
    Output: 48kHz mono WAV file at output_path
    """
    logger.info(f"Saving voice sample ({len(pcm_bytes)//1000}KB) as WAV...")

    # PCM bytes → mono float32 at 48kHz
    arr = np.frombuffer(pcm_bytes, dtype=np.int16).reshape(-1, 2)
    mono = arr.mean(axis=1).astype(np.float32) / 32768.0
    wav = torch.from_numpy(mono).unsqueeze(0)  # (1, T)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torchaudio.save(output_path, wav, DISCORD_SR)
    logger.info(f"Voice WAV saved: {output_path}")


async def encode_voice_to_pt(pcm_bytes: bytes, output_path: str) -> None:
    """
    Async wrapper — runs blocking WAV save in a thread executor.
    output_path should end in .wav, not .pt.
    """
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _encode_blocking, pcm_bytes, output_path)
