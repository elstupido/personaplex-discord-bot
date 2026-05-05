"""
The Forensic Eye.

WHY THIS FILE EXISTS:
Sometimes the AI sounds weird. This module provides diagnostic 'Echo' 
modes that allow us to bypass the LLM and hear exactly what the tokenizer 
or the raw intake layer is capturing.
"""

import asyncio
import base64
from .constants import logger

async def run_raw_echo(bridge, segments):
    """Play back the intake audio with zero processing."""
    for seg in segments:
        raw = base64.b64decode(seg['audio'])
        if bridge.audio_source:
            bridge.audio_source.feed_raw(raw)
        logger.info(f"[Diagnostics] Echoing raw intake ({len(raw)}B)")

async def run_prompt_echo(bridge, segments):
    """Loop tokens straight back to the vocoder (Bypasses LLM)."""
    loop = asyncio.get_event_loop()
    logger.info("[Diagnostics] Prompt Echo Mode Active")
    
    for seg in segments:
        pcm_data = base64.b64decode(seg["audio"])
        # 1. Tokenize -> 2. Vocode -> 3. Upsample
        audio_16k = await loop.run_in_executor(None, bridge.resampler.downsample, pcm_data)
        features = await loop.run_in_executor(None, bridge.tokenizer.extract_features, audio_16k)
        tokens = await loop.run_in_executor(None, bridge.tokenizer.get_vq_tokens, features)
        token_tensor = await loop.run_in_executor(None, bridge.decoder.prepare_tokens, tokens)
        mel = await loop.run_in_executor(None, bridge.decoder.generate_flow_mel, token_tensor)
        audio_22k = await loop.run_in_executor(None, bridge.decoder.vocode_to_waveform, mel)
        audio_48k = await loop.run_in_executor(None, bridge.resampler.upsample, audio_22k)
        
        if bridge.audio_source:
            bridge.audio_source.feed_raw(audio_48k)
