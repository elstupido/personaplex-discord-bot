"""
The Thermal Regulator.

WHY THIS FILE EXISTS:
Starting a neural model cold is like starting a car in a blizzard—it's 
slow and clunky. This module ensures that all components (local and 
remote) are primed and ready for action before the user starts talking.
"""

import asyncio
import os
from .constants import logger, Colors, LAZY_LOAD_MODELS
from .assets import load_asset_pcm

async def perform_full_warmup(bridge):
    """Prime everything concurrently."""
    logger.info(f"{Colors.CYAN}[GLMBridge.Warmup] Igniting neural kernels...{Colors.RESET}")
    
    tasks = [warmup_glm_server(bridge)]
    if not LAZY_LOAD_MODELS:
        tasks.append(warmup_local_pipeline(bridge))
    else:
        logger.info("[GLMBridge.Warmup] Lazy Mode Active: Defending VRAM. Local pipeline warmup skipped.")
    
    # WHY GATHER? To save startup time by priming everything in parallel.
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for r in results:
        if isinstance(r, Exception):
            logger.warning(f"[GLMBridge.Warmup] Subtask failed: {r}")

    # Load system sounds
    await _load_system_assets(bridge)
    logger.info("[GLMBridge.Warmup] Bot is hot and ready.")

async def warmup_local_pipeline(bridge):
    """Warmup the local tokenizer and decoder."""
    loop = asyncio.get_running_loop()
    await asyncio.gather(
        loop.run_in_executor(None, bridge.tokenizer.warmup),
        loop.run_in_executor(None, bridge.decoder.warmup)
    )

async def warmup_glm_server(bridge):
    """Trigger the server's explicit 'force=True' initialization."""
    try:
        async with bridge.session.post(bridge.warmup_url) as resp:
            data = await resp.json()
            if data.get("status") == "warm":
                logger.info("[GLMBridge.Warmup] Server reported HOT.")
    except Exception as e:
        logger.warning(f"[GLMBridge.Warmup] Server warmup failed: {e}")

async def _load_system_assets(bridge):
    """Load UI audio cues."""
    asset_dir = "src/assets"
    
    # WHY Phase 1/2/3? To provide a consistent audio UI language.
    bridge.ding_pcm = await load_asset_pcm(os.path.join(asset_dir, "wake_word_ding.mp3"), 1.0)
    bridge.turn_finalized_pcm = await load_asset_pcm(os.path.join(asset_dir, "turn_finalized.mp3"), 1.0)
    # Windows sound at 50% volume as requested.
    bridge.response_ready_pcm = await load_asset_pcm(os.path.join(asset_dir, "response_ready.mp3"), 0.5)
