"""
The Persona Sculptor.

WHY THIS FILE EXISTS:
This module handles the identity management of the bot. It manages the 
'Crystallization' of new voice clones and the swapping of active 
character profiles.
"""

import base64
import asyncio
import aiohttp
from .constants import logger
from .audio_ops import get_full_pcm

async def process_voice_clone(bridge, segments, incoming):
    """Offload voice reference gathering to the server."""
    loop = asyncio.get_event_loop()
    full_pcm = get_full_pcm(segments)
    pcm_b64 = base64.b64encode(full_pcm).decode('utf-8')

    # WHY ANALYSIS? To ensure the user didn't just record static.
    prompt_text = "[Analysis Offloaded]"
    if bridge.diagnostics_enabled and not bridge.lazy_load:
        audio_16k = await loop.run_in_executor(None, bridge.resampler.downsample, full_pcm)
        prompt_text = await loop.run_in_executor(None, bridge.tokenizer.transcribe, audio_16k)

    # WHY SLUGS? To ensure identity names are safe for URLs/Filenames.
    raw_name = incoming.get('username', 'user')
    profile_name = raw_name.split('#')[0][:16].lower().replace(' ', '_')

    # Send to server vault
    base_url = bridge.url.rsplit('/', 1)[0]
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{base_url}/clone_reference",
            json={"pcm_b64": pcm_b64, "name": profile_name, "sample_rate": 48000, "prompt_text": prompt_text}
        ) as response:
            await response.read()

    logger.info(f"[GLMBridge.Identity] Identity '{profile_name}' crystallized.")
    if bridge.text_channel:
        await bridge.text_channel.send(f"🎙️ **Identity Crystallized:** {profile_name}")

async def list_voices(bridge):
    """Query the server for saved profiles."""
    base_url = bridge.url.rsplit('/', 1)[0]
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{base_url}/list_voices") as response:
            data = await response.json()
            return data.get("voices", [])

async def switch_voice(bridge, name):
    """Swap the active identity."""
    base_url = bridge.url.rsplit('/', 1)[0]
    async with aiohttp.ClientSession() as session:
        async with session.post(f"{base_url}/load_voice_profile", json={"name": name}) as response:
            data = await response.json()
            if data.get("status") == "ok":
                bridge.active_voice = name
                return True
            return False
