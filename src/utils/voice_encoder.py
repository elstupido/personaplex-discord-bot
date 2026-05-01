"""
Voice encoder for /cloneme: saves Discord PCM as a WAV file in the voices directory.
The Moshi server's load_voice_prompt() handles encoding internally when given a .wav file.
We do NOT try to replicate the .pt embedding format — let the server do it correctly.
"""
import asyncio
import logging
import os

import numpy as np
import torch
import torchaudio

logger = logging.getLogger("utils.voice_encoder")

DISCORD_SR = 48000
HF_REPO = "nvidia/personaplex-7b-v1"

_VOICE_PROMPTS_DIR_CACHE: str | None = None


def _get_voice_prompts_dir() -> str:
    """
    Resolve the voices directory in the PersonaPlex model cache.
    Uses huggingface_hub — avoids importing moshi.server which calls main() at module level.
    """
    global _VOICE_PROMPTS_DIR_CACHE
    if _VOICE_PROMPTS_DIR_CACHE is None:
        from huggingface_hub import snapshot_download
        hf_home = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        snapshot_path = snapshot_download(
            repo_id=HF_REPO,
            cache_dir=os.path.join(hf_home, "hub"),
            local_files_only=True,
        )
        _VOICE_PROMPTS_DIR_CACHE = os.path.join(snapshot_path, "voices")
        logger.info(f"Voice prompts dir resolved: {_VOICE_PROMPTS_DIR_CACHE}")
    return _VOICE_PROMPTS_DIR_CACHE


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
