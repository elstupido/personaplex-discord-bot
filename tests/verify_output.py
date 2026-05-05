"""
WHY THIS FILE EXISTS:
Explicit Step Verification (The Manifesto Test). 🏛️📜🧪

WHY:
We reject 'Magic' internal logic. Upsampling and Downsampling are not 
background tasks; they are first-class citizens in the ETL pipeline. 
This test ensures that 'downsampler' and 'upsampler' experts behave 
correctly as standalone steps.
"""

import asyncio
import os
import sys
import torch

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from ai.stupid_base import StupidRegistry, StupidData, AcousticContext, logger
from ai.stupid_config import StupidConfig
import importlib

async def verify_steps():
    logger.info("🚀 Starting Explicit Step Verification...")
    
    # 0. Manual Registration (Mimic Factory)
    importlib.import_module("ai.transform.resampler")
    
    # 1. Test Downsampler (48k Stereo -> 16k Mono)
    logger.info("🧪 Testing 'downsampler' Expert...")
    expert_cls = StupidRegistry.get_expert("downsampler")
    downsampler = expert_cls("test_down")
    
    # WHY: Even standalone steps must be warmed up to ensure the GIL isn't 
    # snatched during the first packet processing.
    if hasattr(downsampler, "warmup"):
        await asyncio.to_thread(downsampler.warmup)
    
    # 48000 samples * 2 channels * 2 bytes = 192000 bytes for 1s
    stereo_48k = b'\x00' * 192000
    ctx_in = AcousticContext(sample_rate=48000)
    data_in = StupidData(content=stereo_48k, context=ctx_in, type="pcm")
    
    async for output in downsampler.process(data_in):
        # 1s at 16kHz should be 16000 float32 samples = 64000 bytes
        # OR if it returns numpy/tensor, we check shape.
        # Downsampler returns np.ndarray in data.content
        logger.info(f"   Downsampled to {output.context.sample_rate}Hz. Content shape: {output.content.shape}")
        if output.context.sample_rate != 16000:
            logger.error("💥 Downsampler failed to update sample rate context.")
            return False

    # 2. Test Upsampler (22k Mono -> 48k Stereo)
    logger.info("🧪 Testing 'upsampler' Expert...")
    expert_cls = StupidRegistry.get_expert("upsampler")
    upsampler = expert_cls("test_up")
    
    if hasattr(upsampler, "warmup"):
        await asyncio.to_thread(upsampler.warmup)
    
    # 22050 int16 samples (1s) = 44100 bytes
    mono_22k = b'\x00' * 44100
    ctx_in = AcousticContext(sample_rate=22050)
    data_in = StupidData(content=mono_22k, context=ctx_in, type="pcm")
    
    async for output in upsampler.process(data_in):
        # 1s at 48kHz Stereo int16 should be 192000 bytes
        logger.info(f"   Upsampled to {output.context.sample_rate}Hz. Content size: {len(output.content)} bytes")
        if len(output.content) != 192000:
            logger.error(f"💥 Upsampler produced {len(output.content)} bytes; expected 192000.")
            return False
        if output.context.sample_rate != 48000:
            logger.error("💥 Upsampler failed to update sample rate context.")
            return False

    logger.info("\n🏆 Explicit Step Verification Complete. Manifesto Honored. 📜✨")
    return True

if __name__ == "__main__":
    success = asyncio.run(verify_steps())
    sys.exit(0 if success else 1)
