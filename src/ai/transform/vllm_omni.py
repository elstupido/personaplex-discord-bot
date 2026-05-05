import asyncio
import json
import aiohttp
import os
import time
from typing import AsyncGenerator
from ai.stupid_base import StupidStep, StupidData, AcousticContext, logger
from ai.stupid_base import StupidRegistry

@StupidRegistry.register("vllm-omni")
class VLLMOmniExpert(StupidStep):
    """
    WHY THIS EXISTS:
    The 'Acoustic Bridge' to the vLLM-Brain container. 
    It acts as the client for the disaggregated stage-graph.
    
    WHY DISAGGERGATED?
    Because neural kernels for multimodal models (Thinker/Talker/Vocoder) 
    are too heavy to run in the same GIL as the Discord bot. This adapter 
    offloads the 'Brain' work to a dedicated CUDA container.
    """

    def __init__(self, name: str = "vllm-omni"):
        super().__init__(name)
        self.server_url = os.getenv("VLLM_SERVER_URL", "http://vllm-brain:8000/v1")
        self.session: Optional[aiohttp.ClientSession] = None
        self.last_brain_latencies = [] # For the "Acoustic Trust" index

    async def _get_session(self) -> aiohttp.ClientSession:
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                headers={"Content-Type": "application/json"}
            )
        return self.session

    async def process(self, data: StupidData) -> AsyncGenerator[StupidData, None]:
        """
        Pipes the incoming data (usually PCM or Tokens) into the Brain.
        
        WHY STREAMING?
        Standard HTTP is too slow for 50Hz physics. We use a streaming 
        response to overlap 'Thinking' with 'Talking'.
        """
        if data.type != "pcm":
            logger.warning(f"⚠️ [vLLM] Expected 'pcm' data, got '{data.type}'. Ignoring.")
            return

        session = await self._get_session()
        
        # Prepare the payload for vLLM-Omni (fish-audio style)
        # Note: vLLM-Omni expects a specific multimodal request format
        payload = {
            "model": "fishaudio/s2-pro",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "data": data.content.tolist() if hasattr(data.content, "tolist") else data.content}
                    ]
                }
            ],
            "stream": True,
            "max_tokens": 512
        }

        start_time = time.time()
        
        try:
            async with session.post(f"{self.server_url}/chat/completions", json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"💥 [vLLM Brain] Error {response.status}: {error_text}")
                    return

                # Read the stream of tokens/audio chunks
                async for chunk in response.content.iter_any():
                    if not chunk:
                        continue
                    
                    # Convert brain output back into a StupidData packet
                    # vLLM-Omni streaming often returns SSE (Server-Sent Events)
                    # We need to parse the 'delta' which contains the audio tokens/PCM
                    
                    # TODO: Implement robust SSE parsing for vLLM-Omni output format
                    # For now, we assume the chunk is the raw audio delta
                    
                    yield StupidData(
                        content=chunk,
                        context=data.context,
                        type="tokens" # Brain usually returns tokens or vocoder output
                    )

            end_time = time.time()
            latency = (end_time - start_time) * 1000
            self.last_brain_latencies.append(latency)
            if len(self.last_brain_latencies) > 10:
                self.last_brain_latencies.pop(0)
            
            logger.debug(f"🧠 [vLLM Brain] Turn completed in {latency:.2f}ms")

        except Exception as e:
            logger.error(f"💥 [vLLM Brain] Connection failed: {e}")
            # Inject a 'Expert Meltdown' signal into the river
            yield StupidData(content="ERR_BRAIN_OFFLINE", context=data.context, type="signal")

    async def close(self):
        if self.session:
            await self.session.close()
            logger.info("🔌 [vLLM Brain] Acoustic Bridge closed.")
