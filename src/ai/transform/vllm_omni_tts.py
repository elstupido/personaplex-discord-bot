# WHY THIS FILE EXISTS:
# This is the 'Mouth' of the Acoustic Bridge. 
# It leverages the Locked-In Fish S2 Pro stages (1 & 2) for peak fidelity.

import os
import aiohttp
import base64
from ..stupid_base import StupidStep, StupidData, StupidRegistry, logger
from typing import AsyncGenerator, Optional

@StupidRegistry.register("vllm-omni-tts")
class TTSExpert(StupidStep):
    def __init__(self, name: str):
        super().__init__(name)
        # WHY: We use the dedicated speech endpoint for high-fidelity cloning.
        self.api_url = os.getenv("VLLM_SPEECH_URL", os.getenv("VLLM_SERVER_URL", "http://localhost:8000/v1"))
        if not self.api_url.endswith("/audio/speech"):
            self.api_url = f"{self.api_url.rstrip('/')}/audio/speech"
        self.ref_voice_path = "src/assets/reference_voice.wav"
        self.voices_dir = "voice_profiles"
        self.ref_audio_b64 = None
        self.ref_text = "Hello, I am the StupidBot reference voice."
        self.current_loaded_voice = None
        self.session: Optional[aiohttp.ClientSession] = None

    async def _ensure_session(self):
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None))

    async def warmup(self):
        """Pre-heat the Mouth."""
        await self._ensure_session()
        self._load_ref_voice()
        logger.info(f"✨ [TTSExpert] '{self.name}' warmed.")
            
    def _load_ref_voice(self, voice_name: str = None):
        """
        Loads the 'Acoustic Soul' of the bot.
        
        WHY: Fish-TTS requires a reference sample to anchor the vocal identity. 
        """
        # If we already have this voice loaded, skip
        if self.current_loaded_voice == voice_name and self.ref_audio_b64:
            return

        target_path = self.ref_voice_path
        if voice_name and voice_name != "default":
            custom_path = os.path.join(self.voices_dir, f"{voice_name}.wav")
            if os.path.exists(custom_path):
                target_path = custom_path
                logger.info(f"🎭 [TTS] Using custom voice profile: {voice_name}")
            else:
                logger.warning(f"⚠️ [TTS] Voice profile '{voice_name}' not found. Falling back to default.")

        try:
            if os.path.exists(target_path):
                with open(target_path, "rb") as f:
                    self.ref_audio_b64 = "data:audio/wav;base64," + base64.b64encode(f.read()).decode("utf-8")
                self.current_loaded_voice = voice_name
                logger.info(f"🧬 [TTS] Reference voice loaded: {target_path}")
                
                # Load transcript if available
                txt_path = target_path.replace(".wav", ".txt")
                if os.path.exists(txt_path):
                    with open(txt_path, "r") as f:
                        self.ref_text = f.read().strip()
                    logger.info(f"📝 [TTS] Reference transcript loaded: \"{self.ref_text}\"")
                else:
                    self.ref_text = "Hello, I am the StupidBot reference voice."
            else:
                logger.warning(f"⚠️ [TTS] Reference voice missing at {target_path}")
        except Exception as e:
            logger.error(f"💥 [TTS] Failed to load reference voice: {e}")

    async def process(self, data: StupidData) -> AsyncGenerator[StupidData, None]:
        await self._ensure_session()
        
        # Determine which voice to use from context metadata
        active_voice = data.context.metadata.get('active_voice', 'default')
        self._load_ref_voice(active_voice)
        
        response_text = data.content
        if data.type != "text" or response_text is None:
            yield data
            return

        logger.info(f"🗣️ [TTS] Synthesizing: \"{response_text}\" via {self.api_url}...")

        # WHY: Use /v1/audio/speech for direct access to the Fish-Speech engine.
        payload = {
            "model": "fishaudio/s2-pro",
            "input": response_text,
            "voice": "reference", # Changed from "default" to "reference" to force use of ref_audio
            "response_format": "pcm",
            "ref_audio": self.ref_audio_b64,
            "ref_text": self.ref_text,
            "latency": "normal"
        }

        try:
            async with self.session.post(self.api_url, json=payload) as resp:
                if resp.status == 200:
                    # WHY: We stream the raw binary PCM chunks as they arrive from the server.
                    # This allows the bot to start playing audio before the full generation is done.
                    buffer = b""
                    async for chunk, _ in resp.content.iter_chunks():
                        if chunk:
                            buffer += chunk
                            # WHY: aiohttp chunks might break exactly halfway through a 2-byte (16-bit) sample.
                            # We must buffer the remainder to ensure np.frombuffer downstream doesn't crash.
                            if len(buffer) >= 2:
                                remainder = len(buffer) % 2
                                if remainder > 0:
                                    valid_chunk = buffer[:-remainder]
                                    buffer = buffer[-remainder:]
                                else:
                                    valid_chunk = buffer
                                    buffer = b""

                                chunk_data = StupidData(
                                    content=valid_chunk,
                                    context=data.context,
                                    type="pcm"
                                )
                                chunk_data.context.sample_rate = 24000
                                yield chunk_data
                else:
                    error_text = await resp.text()
                    logger.error(f"❌ [TTS] Failed: {resp.status} - {error_text}")
        except Exception as e:
            logger.error(f"💥 [TTS] Request failed: {e}", exc_info=True)
            # In an ETL pipeline, we might want to yield a specific error particle 
            # instead of just the original data to signal failure.
            data.type = "signal"
            data.content = "ERR_TTS_FAILED"
        
        yield data
