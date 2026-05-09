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
        self.ref_audio_b64 = None
        self.session: Optional[aiohttp.ClientSession] = None

    async def _ensure_session(self):
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None))

    async def warmup(self):
        """Pre-heat the Mouth."""
        await self._ensure_session()
        self._load_ref_voice()
        logger.info(f"✨ [TTSExpert] '{self.name}' warmed.")
            
    def _load_ref_voice(self):
        """
        Loads the 'Acoustic Soul' of the bot.
        
        WHY: Fish-TTS requires a reference sample to anchor the vocal identity. 
        We load it once and cache it in memory to avoid disk I/O on every turn.
        """
        if self.ref_audio_b64 is None:
            if os.path.exists(self.ref_voice_path):
                with open(self.ref_voice_path, "rb") as f:
                    self.ref_audio_b64 = "data:audio/wav;base64," + base64.b64encode(f.read()).decode("utf-8")
                logger.info(f"🧬 [TTS] Reference voice loaded: {self.ref_voice_path}")
            else:
                logger.warning(f"⚠️ [TTS] Reference voice missing at {self.ref_voice_path}")

    async def process(self, data: StupidData) -> AsyncGenerator[StupidData, None]:
        await self._ensure_session()
        self._load_ref_voice()
        
        response_text = data.content
        if data.type != "text" or response_text is None:
            yield data
            return

        logger.info(f"🗣️ [TTS] Synthesizing: \"{response_text}\" via {self.api_url}...")

        # WHY: Use /v1/audio/speech for direct access to the Fish-Speech engine.
        payload = {
            "model": "fishaudio/s2-pro",
            "input": response_text,
            "voice": "default", 
            "response_format": "pcm",
            "ref_audio": self.ref_audio_b64,
            "ref_text": os.getenv("REF_TEXT", "Hello, I am the StupidBot reference voice."),
            "latency": "normal"
        }

        try:
            async with self.session.post(self.api_url, json=payload) as resp:
                if resp.status == 200:
                    # /v1/audio/speech returns raw binary PCM
                    data.content = await resp.read()
                    data.type = "pcm"
                    data.context.sample_rate = 24000
                    logger.info(f"✅ [TTS] Generated {len(data.content)} bytes of high-fidelity audio (24kHz).")
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
