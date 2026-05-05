"""
WHY THIS FILE EXISTS:
The 'GLMExpert' (The Unified Brain). 🧠

WHY:
GLM-4-Voice is a unified multi-modal model. It doesn't just 'think' in 
text; it 'hears' and 'speaks' in tokens. This expert wraps the 
streaming gateway into the Recursive ETL pipeline, yielding a 
mixed-media stream of text and audio particles.
"""

import os
import json
import base64
import aiohttp
from typing import Generator
from ..stupid_base import StupidStep, StupidData, StupidRegistry, logger

@StupidRegistry.register("glm-4")
class GLMExpert(StupidStep):
    """
    The Unified Multi-modal Expert. 🤖
    """
    def __init__(self, name: str):
        super().__init__(name)
        self.server_url = os.getenv("GLM_SERVER_URL", "http://127.0.0.1:10000")
        self.url = f"{self.server_url}/generate_complex_stream"
        self.session = None # Initialized lazily
        logger.info(f"✨ [GLMExpert] '{name}' initialized. Neural gates are open at {self.server_url}")

    async def warmup(self):
        """Pre-heat the neural connection."""
        # WHY: Establishing a session involves network I/O. 
        # We do it now so the first turn is instant.
        await self._ensure_session()
        logger.info("[GLMExpert] Neural session warmed.")

    async def _ensure_session(self):
        """
        Maintain a persistent session. 🦾
        
        WHY: Re-opening TCP connections for every turn is a latency sin 
        we refuse to commit.
        """
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None))

    async def process(self, data: StupidData) -> Generator[StupidData, None, None]:
        """
        The Neural Transformation. ⚡
        
        WHY:
        This expert takes in 'tokens' (semantic tokens from ASR) or 'pcm' 
        (raw audio) and yields 'text' and 'pcm' chunks as they arrive 
        from the server.
        """
        await self._ensure_session()
        
        # 1. Prepare Payload 📦
        # For now, we mimic the legacy GLMBridge message structure.
        # WHY: To ensure compatibility with the existing GLM-Server.
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": ""}
        ]
        
        if data.type == "tokens":
            # We have semantic tokens! 🧠
            messages[-1]["audio_tokens"] = data.content
        elif data.type == "pcm":
            # We have raw audio. B64 it for the server. 🔈
            messages[-1]["audio_b64"] = base64.b64encode(data.content).decode('utf-8')
        
        payload = {
            "messages": messages,
            "vocoder": "glm" # Default to high-fidelity vocoder
        }

        logger.debug(f"🚀 [GLMExpert] Dispatching turn to {self.server_url}...")
        
        try:
            async with self.session.post(self.url, json=payload) as response:
                if response.status != 200:
                    logger.error(f"🛑 [GLMExpert] Server returned error {response.status}")
                    return

                # 2. Iterate over the Neural River 🌊
                async for line in response.content:
                    if not line: continue
                    try:
                        chunk = json.loads(line)
                        
                        # Yield Audio Chunks 🔈
                        if "audio_chunk" in chunk:
                            audio_22k = base64.b64decode(chunk["audio_chunk"])
                            yield StupidData(
                                content=audio_22k,
                                context=data.context,
                                type="pcm"
                            )
                        
                        # Yield Text Chunks ✍️
                        elif "text_chunk" in chunk:
                            yield StupidData(
                                content=chunk["text_chunk"],
                                context=data.context,
                                type="text"
                            )
                            
                    except json.JSONDecodeError:
                        continue
                        
        except Exception as e:
            logger.error(f"💥 [GLMExpert] Neural connection lost: {e}")
            
        logger.debug("✅ [GLMExpert] Turn processing complete.")

    async def cleanup(self):
        """Close the neural gates. 🚪"""
        if self.session:
            await self.session.close()
            logger.info("[GLMExpert] Neural session closed.")
