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

VOICES_DIR = "voice_profiles"


def _encode_blocking(pcm_bytes: bytes, output_path: str) -> None:
    """
    Save the user's speech as a WAV file.
    Input:  raw 48kHz stereo int16 PCM bytes
    Output: 48kHz mono WAV file at output_path
    """
    try:
        logger.info(f"Saving voice sample ({len(pcm_bytes)//1000}KB) as WAV...")

        # 1. Robust PCM Conversion
        # WHY: Handle cases where the buffer might have been truncated on an odd byte
        if len(pcm_bytes) % 2 != 0:
            pcm_bytes = pcm_bytes[:-1]
        
        raw_samples = np.frombuffer(pcm_bytes, dtype=np.int16)
        
        # Ensure we have an even number of samples for stereo-to-mono conversion
        if len(raw_samples) % 2 != 0:
            raw_samples = raw_samples[:-1]
            
        arr = raw_samples.reshape(-1, 2)
        
        # 2. Convert to float32 Mono
        mono = arr.mean(axis=1).astype(np.float32) / 32768.0

        # 3. Trim Silence
        # WHY: Prevent autoregressive TTS from cloning a "slow motion" cadence by 
        # removing leading/trailing dead air.
        threshold = 0.015 # 1.5% amplitude
        non_silent_indices = np.where(np.abs(mono) > threshold)[0]
        
        if len(non_silent_indices) > 0:
            pad = int(0.1 * DISCORD_SR) # 100ms padding
            start_idx = max(0, non_silent_indices[0] - pad)
            end_idx = min(len(mono), non_silent_indices[-1] + pad)
            mono = mono[start_idx:end_idx]
            logger.info(f"✨ [Encoder] Trimmed audio to {len(mono)/DISCORD_SR:.2f}s (from {len(arr)/DISCORD_SR:.2f}s)")
        else:
            logger.warning("⚠️ [Encoder] Voice sample was entirely silent!")

        # 4. Save to Disk
        # WHY: Fish-Speech S2-Pro is natively trained on 44.1kHz audio. 
        # Feeding it 48kHz audio can confuse the duration predictor, causing "slow motion" artifacts.
        wav = torch.from_numpy(mono).unsqueeze(0)  # (1, T)
        
        target_sr = 44100
        if DISCORD_SR != target_sr:
            resampler = torchaudio.transforms.Resample(DISCORD_SR, target_sr)
            wav = resampler(wav)
            
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torchaudio.save(output_path, wav, target_sr)
        logger.info(f"Voice WAV saved: {output_path} (Resampled to {target_sr}Hz)")

    except Exception as e:
        logger.error(f"💥 [Encoder] Failed to save voice profile: {e}", exc_info=True)
        # Re-raise to ensure the future catches it
        raise e


async def encode_voice_to_pt(pcm_bytes: bytes, output_path: str) -> None:
    """
    Async wrapper — runs blocking WAV save in a thread executor.
    output_path should end in .wav, not .pt.
    """
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _encode_blocking, pcm_bytes, output_path)
