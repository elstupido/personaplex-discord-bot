"""
WHY THIS FILE EXISTS:
The 'ResamplerExpert' (The Temporal Bridge). 🌊

WHY:
Audio exists in multiple frequency domains. This expert ensures that 
as data flows from Discord (48kHz) to the AI (16kHz) and back, 
the temporal integrity is preserved using GPU-accelerated interpolation.
"""

import asyncio
from typing import Generator, AsyncGenerator
from ..stupid_base import StupidStep, StupidData, StupidRegistry, logger
from ..components import AudioResampler

@StupidRegistry.register("downsampler")
class DownsamplerExpert(StupidStep):
    """
    Compressing Reality (48kHz -> 16kHz). 📻📉
    """
    def __init__(self, name: str):
        super().__init__(name)
        self.resampler = AudioResampler(device="cuda")
        logger.info(f"✨ [Downsampler] '{name}' initialized. Intake gates open.")

    def warmup(self):
        """Pre-heat the GPU kernels."""
        self.resampler.warmup()

    async def process(self, data: StupidData) -> AsyncGenerator[StupidData, None]:
        if data.type != "pcm":
            yield data
            return

        # Discord (48k) -> AI (16k)
        processed_pcm = await asyncio.to_thread(self.resampler.downsample, data.content)
        data.context.sample_rate = 16000
        data.content = processed_pcm
        yield data

@StupidRegistry.register("upsampler")
class UpsamplerExpert(StupidStep):
    """
    Expanding Reality (22kHz -> 48kHz Stereo). 📻📈
    """
    def __init__(self, name: str):
        super().__init__(name)
        self.resampler = AudioResampler(device="cuda")
        logger.info(f"✨ [Upsampler] '{name}' initialized. Playback gates open.")

    def warmup(self):
        """Pre-heat the GPU kernels."""
        self.resampler.warmup()

    async def process(self, data: StupidData) -> AsyncGenerator[StupidData, None]:
        if data.type != "pcm":
            yield data
            return

        # AI (current_sr) -> Discord (48k Stereo)
        current_sr = data.context.sample_rate
        processed_pcm = await asyncio.to_thread(
            self.resampler.upsample, 
            data.content, 
            orig_sr=current_sr, 
            target_sr=48000
        )
        
        data.context.sample_rate = 48000
        data.content = processed_pcm
        yield data
