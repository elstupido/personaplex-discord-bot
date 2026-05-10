# WHY THIS FILE EXISTS:
# This is the 'Body-Brain' reasoning core, now disaggregated to a remote vLLM container.
# WHY: We run the reasoning model in a dedicated GPU container (vllm-actual-brain) 
# instead of System RAM. This frees up the Discord bot's CPU and ensures fast, 
# non-blocking text generation.

import os
import aiohttp
import json
from ..stupid_base import StupidStep, StupidData, StupidRegistry, logger
from typing import AsyncGenerator, Optional

@StupidRegistry.register("vllm_reasoning")
class VLLMReasoningExpert(StupidStep):
    """
    The 'Thinker' that delegates to the vLLM API. 🧠📡
    """
    def __init__(self, name: str):
        super().__init__(name)
        # WHY: Target the dedicated reasoning container, not the TTS container.
        self.api_url = os.getenv("VLLM_REASONING_URL", "http://localhost:8001/v1/chat/completions")
        self.model_id = os.getenv("REASONING_MODEL", "Qwen/Qwen2.5-1.5B-Instruct-AWQ")
        self.session: Optional[aiohttp.ClientSession] = None
        logger.info(f"✨ [VLLMReasoning] '{name}' initialized. Targeting: {self.api_url}")

    async def _ensure_session(self):
        """Lazy initialization of the aiohttp session and auto-detect model."""
        if self.session is None or self.session.closed:
            # WHY: The first request triggers vLLM to compile CUDA graphs, which takes ~20s. 
            self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60))
            
            # WHY: If you change the model in docker-compose, we shouldn't have to hardcode it here.
            # We hit the /v1/models endpoint to ask vLLM what model it's currently serving.
            try:
                models_url = self.api_url.replace("/chat/completions", "/models")
                async with self.session.get(models_url) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        self.model_id = data["data"][0]["id"]
                        logger.info(f"✨ [VLLMReasoning] Auto-detected served model: {self.model_id}")
            except Exception as e:
                logger.warning(f"⚠️ [VLLMReasoning] Could not auto-detect model, falling back to {self.model_id}: {e}")

    async def process(self, data: StupidData) -> AsyncGenerator[StupidData, None]:
        await self._ensure_session()
        
        user_input = data.content
        if data.type != "text" or user_input is None:
            yield data
            return

        # WHY: Empty prompts confuse models.
        if not user_input.strip():
            logger.warning("⚠️ [Think] Empty prompt received. Bypassing reasoning.")
            data.content = "I'm sorry, I didn't hear anything. Could you repeat that?"
            yield data
            return

        # System Prompt Injection (from env)
        system_prompt = os.getenv("TEXT_PROMPT", "You are a helpful assistant. Keep your answers brief.")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]
        
        payload = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": 256,
            "temperature": 0.8,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            "stream": False # WHY: We want the full text before sending to TTS for better prosody.
        }

        try:
            logger.info(f"💡 [VLLMReasoning] Querying Thinker: \"{user_input}\"...")
            async with self.session.post(self.api_url, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    response_text = result["choices"][0]["message"]["content"].strip()
                    logger.info(f"💡 [VLLMReasoning] Response: \"{response_text}\"")
                    data.content = response_text
                    data.type = "text"
                else:
                    err_text = await response.text()
                    logger.error(f"💥 [VLLMReasoning] API Error ({response.status}): {err_text}")
                    data.content = "Ah, my brain hurts. Could you say that again?"
                    data.type = "text"
        except Exception as e:
            logger.error(f"💥 [VLLMReasoning] Connection failed: {e}")
            data.content = "I seem to have lost my train of thought."
            data.type = "text"

        yield data

    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()
