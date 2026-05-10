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
import numpy as np

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
        self.active_voice = "default"
        self.voices_dir = "voice_profiles"

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

    async def save_voice_profile(self, name: str, audio_pcm: bytes):
        """Save a new voice profile to disk with a transcript."""
        from voice.encoder import encode_voice_to_pt
        from .transform.vllm_omni_asr import ASRExpert
        
        output_path = os.path.join(self.voices_dir, f"{name}.wav")
        txt_path = os.path.join(self.voices_dir, f"{name}.txt")
        
        logger.info(f"💾 [StupidBridge] Transcribing reference for: {name}...")
        
        # 1. Transcribe the reference audio for better Fish-Speech alignment
        # WHY: Fish S2 Pro needs to know EXACTLY what is said in the ref audio.
        try:
            # Simple downmix/resample for ASR expert (16kHz mono)
            import numpy as np
            arr = np.frombuffer(audio_pcm, dtype=np.int16).reshape(-1, 2)
            mono_48k = arr.mean(axis=1).astype(np.int16)
            
            # Basic decimation 48k -> 16k (Every 3rd sample)
            mono_16k = mono_48k[::3].tobytes()
            
            from .stupid_base import AcousticContext
            asr = ASRExpert("clone-transcriber")
            ctx = AcousticContext(user_id="cloning_reference")
            data = StupidData(content=mono_16k, context=ctx, type="pcm")
            async for result in asr.process(data):
                transcript = result.content
                if transcript:
                    with open(txt_path, "w") as f:
                        f.write(transcript)
                    logger.info(f"📝 [StupidBridge] Reference transcript saved: \"{transcript}\"")
                break
        except Exception as e:
            logger.error(f"⚠️ [StupidBridge] Transcription failed during clone: {e}")

        # 2. Save the WAV
        logger.info(f"💾 [StupidBridge] Saving voice profile: {name} -> {output_path}")
        await encode_voice_to_pt(audio_pcm, output_path)
        return True

    async def load_voice_profile(self, name: str) -> bool:
        """Switch the active voice profile."""
        path = os.path.join(self.voices_dir, f"{name}.wav")
        if os.path.exists(path):
            self.active_voice = name
            logger.info(f"✨ [StupidBridge] Active voice switched to: {name}")
            return True
        return False

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
        ctx.metadata['is_partial'] = payload.get('is_partial', False)
        data = StupidData(content=audio, context=ctx, type="pcm")
        
        # 3. Define the Blueprint Job 📜
        # WHY: No more magic strings! We use the steps defined in the blueprint.
        job = StupidJob(
            steps=self.blueprint.steps,
            data=data
        )

        # 4. Handle Voice Cloning 🎙️
        if payload.get('is_clone_reference'):
            await self.save_voice_profile(self.active_voice, audio)
            return # Don't process as a normal turn
        
        logger.info(f"🚀 [StupidBridge] Dispatching {len(audio)} bytes for user {user_id} via '{self.blueprint.name}'")
        
        # 5. Run the Job ⚡
        # WHY: We iterate through the stream of results (PCM particles) 
        # and feed them directly into the Discord source for playback.
        
        # Inject the active voice into the context so the TTS expert can find it
        data.context.metadata['active_voice'] = self.active_voice
        
        async for result in self.runner.execute_job_stream(job):
            if result.type == "pcm" and self.audio_source:
                # Ensure it's bytes (experts might return numpy/tensors)
                content = result.content
                if not isinstance(content, bytes):
                    if isinstance(content, (list, np.ndarray)):
                        content = np.array(content, dtype=np.int16).tobytes()
                
                self.audio_source.feed(content)
        
        logger.info(f"🌊 [StupidBridge] Job complete for user {user_id}.")

    async def speak(self, text: str, voice_name: str = None):
        """
        The 'Mouth-Only' Gateway. 👄
        
        WHY: 
        For /say commands, we bypass the ears (ASR) and the brain (LLM) 
        and go straight to the vocal cords (TTS). We use the exact same 
        expert chain that Reasoning uses to ensure vocal consistency.
        """
        if not self.is_running: 
            logger.warning("⚠️ [StupidBridge] Speak requested but adapter is not running.")
            return

        # 1. Build the Text Particle ⚛️
        # WHY: We use a 'manual' user_id to distinguish this from AI-generated turn logic.
        ctx = AcousticContext(user_id="manual_tts")
        # Inject the requested voice (if any), fallback to the globally active voice, or finally 'default'
        ctx.metadata['active_voice'] = voice_name or getattr(self, 'active_voice', 'default')
        data = StupidData(content=text, context=ctx, type="text")
        
        # 2. Define the 'Mouth' Job 📜
        # WHY: vllm-omni-tts (The Fish Mouth) -> upsampler (48kHz for Discord).
        steps = ["vllm-omni-tts", "upsampler"]
        job = StupidJob(steps=steps, data=data)
        
        logger.info(f"👄 [StupidBridge] Manual TTS dispatch: \"{text[:50]}...\"")
        
        # 3. Execute and Stream to Discord 🔈
        try:
            async for result in self.runner.execute_job_stream(job):
                if result.type == "pcm" and self.audio_source:
                    content = result.content
                    if not isinstance(content, bytes):
                        import numpy as np
                        if isinstance(content, (list, np.ndarray)):
                            content = np.array(content, dtype=np.int16).tobytes()
                    
                    self.audio_source.feed(content)
            logger.info("✅ [StupidBridge] Manual TTS playback complete.")
        except Exception as e:
            logger.error(f"💥 [StupidBridge] Manual TTS failed: {e}", exc_info=True)
