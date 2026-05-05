"""
The Acoustic Transformer.

WHY THIS FILE EXISTS:
This module manages the temporal bridge between 22.05kHz (AI synthesis) 
and 48kHz (Discord playback). It handles the pipelined upsampling queue, 
ensuring that the bot can start talking while the next audio chunk is 
still being upsampled in the background.
"""

import asyncio
import base64
from contextlib import asynccontextmanager
from .constants import logger

def get_full_pcm(segments: list) -> bytes:
    """
    Concatenate raw PCM segments.
    
    WHY? Because a 'Turn' is often comprised of multiple audio spurts. 
    We join them here to create a single, continuous signal for analysis.
    """
    return b"".join(base64.b64decode(seg["audio"]) for seg in segments)

async def pipelined_upsampler_task(bridge):
    """
    Background worker for zero-latency upsampling.
    
    WHY? Resampling is a CPU-bound operation. If we did it in the main 
    network loop, the stream would stutter. By using an executor and a 
    queue, we keep the audio flowing like butter.
    """
    loop = asyncio.get_event_loop()
    while bridge.is_running:
        try:
            audio_22k = await bridge.upsample_queue.get()
            if audio_22k is None:
                continue
            
            # WHY EXECUTOR? To avoid blocking the event loop.
            audio_48k = await loop.run_in_executor(
                None, bridge.resampler.upsample_bytes, audio_22k
            )
            
            if bridge.audio_source:
                bridge.audio_source.feed(audio_48k)
            
            bridge.upsample_queue.task_done()
        except Exception as e:
            logger.error(f"[GLMBridge.Audio] Upsampler crash: {e}")
            await asyncio.sleep(0.1)

@asynccontextmanager
async def voice_state_context(bridge):
    """
    State-managed silence.
    
    WHY? When the bot speaks, it should deafen itself to avoid hearing 
    its own echo and triggering an infinite loop of AI-to-AI madness.
    """
    did_deafen = False
    if bridge.vc:
        try:
            await bridge.vc.guild.change_voice_state(
                channel=bridge.vc.channel, self_deaf=True
            )
            did_deafen = True
        except Exception as e:
            logger.warning(f"[GLMBridge.Audio] Deafen failed: {e}")
    
    try:
        yield
    finally:
        # WHY SLEEP? We must wait for the actual audio to exit the speakers 
        # before we open the ears again.
        if bridge.vc and bridge.audio_source and hasattr(bridge.audio_source, 'is_playing'):
            while bridge.audio_source.is_playing:
                await asyncio.sleep(0.1)

        if bridge.vc:
            try:
                if did_deafen:
                    await bridge.vc.guild.change_voice_state(
                        channel=bridge.vc.channel, self_deaf=False
                    )
                if hasattr(bridge.vc, 'sink') and hasattr(bridge.vc.sink, 'is_listening'):
                    bridge.vc.sink.is_listening = True
            except Exception as e:
                logger.warning(f"[GLMBridge.Audio] Undeafen failed: {e}")
