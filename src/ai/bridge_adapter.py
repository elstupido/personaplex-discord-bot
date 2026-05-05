"""
WHY THIS FILE EXISTS:
The 'StupidBridge' (The Structural Adapter). 🌉

WHY:
During Phase 2 & 3, we have a 'Dual-Reality' problem. The Discord Cog 
expects a 'Bridge' object with a 'send_audio_packet' method, but our 
new experts live in a 'Recursive ETL' pipeline. This adapter wraps 
the pipeline into a bridge-shaped box.
"""

import asyncio
from .stupid_base import StupidJob, StupidData, AcousticContext, logger
from .stupid_runner import StupidRunner
from .stupid_config import StupidConfig
from .providers.glm.assets import load_asset_pcm
import os

class StupidBridgeAdapter:
    """
    Wraps a StupidRunner and a default Job into a legacy Bridge interface. 🔄
    """
    def __init__(self, model_type: str):
        self.model_type = model_type
        self.runner = StupidRunner()
        self.is_running = False
        
        # 1. Fetch the Master Blueprint 📜
        # WHY: Instead of guessing what steps to run, we ask the Sigil Registry.
        self.blueprint = StupidConfig.get_active_blueprint()
        logger.info(f"✨ [StupidBridge] Adapter active for '{model_type}'. Using Blueprint: {self.blueprint.name}")
        
        # Acoustic UI Assets
        self.ding_pcm = b""
        self.audio_source = None # Set by Cog

    async def connect(self):
        self.is_running = True
        logger.info(f"✨ [StupidBridge] Warming up experts for '{self.model_type}'...")
        
        # Load assets and warmup concurrently
        await asyncio.gather(
            self.runner.warmup(self.blueprint.steps),
            self._load_assets()
        )
        logger.info(f"✅ [StupidBridge] '{self.model_type}' connected and warmed up.")

    async def _load_assets(self):
        """Load UI audio cues."""
        asset_dir = "src/assets"
        try:
            self.ding_pcm = await load_asset_pcm(os.path.join(asset_dir, "wake_word_ding.mp3"), 1.0)
            logger.debug("✅ [StupidBridge] UI Assets loaded.")
        except Exception as e:
            logger.error(f"⚠️ [StupidBridge] Asset load failed: {e}")

    async def play_ding(self):
        """Play the wake word confirmation sound."""
        if self.ding_pcm and self.audio_source:
            self.audio_source.feed_raw(self.ding_pcm)
        else:
            logger.warning("⚠️ [StupidBridge] Ding requested but assets/source missing.")

    async def start_streaming(self):
        # The new pipeline doesn't 'stream' in the background like the old one.
        # It processes turns on-demand. No action needed here. 😴
        pass

    async def close(self):
        self.is_running = False
        logger.info(f"🛑 [StupidBridge] '{self.model_type}' adapter closed.")

    async def send_audio_packet(self, payload: dict):
        """
        The Bridge-to-Runner Gateway. ⚡
        
        WHY: 
        The Orchestrator hands us a 'payload' (with raw audio and metadata). 
        We wrap it in a StupidJob and feed it to the Runner.
        """
        if not self.is_running: return

        # 1. Extract raw audio and context 🔈
        audio = payload.get('audio', b'')
        user_id = payload.get('user_id')
        
        # 2. Build the Atomic Particle ⚛️
        ctx = AcousticContext(user_id=user_id)
        data = StupidData(content=audio, context=ctx, type="pcm")
        
        # 3. Define the Blueprint Job 📜
        # WHY: No more magic strings! We use the steps defined in the blueprint.
        job = StupidJob(
            steps=self.blueprint.steps,
            data=data
        )
        
        logger.debug(f"🚀 [StupidBridge] Dispatching job for user {user_id}...")
        
        # 4. Run the Job ⚡
        # Note: In a real bot, we'd handle the results (audio/text) here 
        # to feed the discord audio source.
        results = await self.runner.execute_job(job)
        
        logger.debug(f"🌊 [StupidBridge] Job complete. Produced {len(results)} output particles.")
        # TODO: Feed results to AudioSource (Phase 3)
