import asyncio
import json
import aiohttp
import os
import time
import re
import base64
from typing import AsyncGenerator, Optional, Any, List, Dict
from ai.stupid_base import StupidStep, StupidData, AcousticContext, logger
from ai.stupid_base import StupidRegistry

@StupidRegistry.register("vllm-omni")
class VLLMOmniExpert(StupidStep):
    """
    WHY THIS EXISTS:
    The 'Acoustic Bridge' for the dual-stage neural pipeline.
    It orchestrates the flow from Gemma (Comprehension) to Fish-Speech (Synthesis).
    
    WHY THE CASCADE?
    Waiting for a full LLM response is for people with slow internet. 
    By streaming from the Brain and synthesizing sentence-by-sentence, 
    we overlap cognition with elocution, maintaining that sub-100ms vibe.
    
    WHY ROOM AWARENESS?
    In a Discord channel, multiple people talk. By aggregating per-user 
    audio streams into a single multimodal request, the Brain can reason 
    about the conversation's social dynamics (Diarization).
    """

    def __init__(self, name: str = "vllm-omni"):
        super().__init__(name)
        self.brain_url = os.getenv("BRAIN_URL", "http://localhost:8000/v1")
        
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Stateful Buffers (per user_id)
        # WHY: We aggregate audio from multiple speakers to provide 
        # 'Room Awareness'. The Brain can then reason about who said what.
        self.active_buffers: Dict[str, bytearray] = {}
        self.last_activity: float = 0
        
        # Regex for sentence boundaries (The 'Punctuation Trigger')
        self.sentence_regex = re.compile(r'[^.!?]+[.!?]')

    async def _get_session(self) -> aiohttp.ClientSession:
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=60),
                headers={"Content-Type": "application/json"}
            )
        return self.session

    def _create_wav_header(self, pcm_data: bytes, sample_rate: int = 16000) -> bytes:
        import struct
        num_channels = 1
        sample_width = 2
        header = bytearray(b'RIFF')
        header.extend(struct.pack('<I', 36 + len(pcm_data)))
        header.extend(b'WAVEfmt ')
        header.extend(struct.pack('<I', 16))
        header.extend(struct.pack('<HHIIHH', 1, num_channels, sample_rate, 
                                 sample_rate * num_channels * sample_width, 
                                 num_channels * sample_width, 8 * sample_width))
        header.extend(b'data')
        header.extend(struct.pack('<I', len(pcm_data)))
        return bytes(header) + pcm_data

    async def process(self, data: StupidData) -> AsyncGenerator[StupidData, None]:
        # 1. Identity & State Management
        user_id = data.context.user_id if data.context else "default"
        is_partial = data.context.metadata.get("is_partial", False) if data.context else False

        if user_id not in self.active_buffers:
            self.active_buffers[user_id] = bytearray()

        # 2. Accumulate Acoustic Deltas
        # WHY: The downsampler returns numpy arrays (float32), not bytes.
        # We must handle both formats since different pipeline configurations
        # might feed us raw bytes OR processed numpy/torch tensors.
        content = data.content
        if isinstance(content, (bytes, bytearray)):
            self.active_buffers[user_id].extend(content)
            self.last_activity = time.time()
        elif hasattr(content, '__array__'):
            # WHY: numpy/torch tensors from the downsampler arrive as float32 [-1, 1].
            # We convert to int16 PCM bytes for the WAV header wrapper.
            import numpy as np
            arr = np.asarray(content, dtype=np.float32)
            pcm_int16 = (arr * 32767).clip(-32768, 32767).astype(np.int16)
            self.active_buffers[user_id].extend(pcm_int16.tobytes())
            self.last_activity = time.time()
        else:
            logger.warning(f"⚠️ [VLLMOmni] Unexpected content type: {type(content)}")
        
        if is_partial:
            # Continue accumulating; we only fire on a full turn trigger
            return

        # --- PHASE 1: THE BRAIN (gpt-omni/mini-omni) ---
        session = await self._get_session()
        # Aggregate all non-empty buffers for the 'Room'
        # WHY: This allows the model to handle multi-speaker diarization natively.
        content_payload = []
        for uid, buffer in list(self.active_buffers.items()):
            if not buffer: continue
            
            # Label the speaker for the Brain's logic
            content_payload.append({"type": "text", "text": f"[Speaker: {uid}]"})
            
            audio_b64 = base64.b64encode(self._create_wav_header(bytes(buffer))).decode('utf-8')
            content_payload.append({
                "type": "input_audio",
                "input_audio": {"data": audio_b64, "format": "wav"}
            })
            # Clear the buffer after packaging
            self.active_buffers[uid] = bytearray()

        if not content_payload:
            return

        # Append the final instructional prompt
        content_payload.append({"type": "text", "text": "Analyze the speaker(s) and respond appropriately."})

        payload = {
            "model": os.getenv("GEMMA_MODEL", "fishaudio/s2-pro"),
            "messages": [
                {
                    "role": "user",
                    "content": content_payload
                }
            ],
            "stream": True,
            "max_tokens": 512,
            "modalities": ["text", "audio"],
            "audio": {"voice": "default", "format": "pcm"}
        }

        logger.info(f"🧠 [Unified Brain] Dispatched multimodal turn with {len(content_payload)//2} speaker(s) to {self.brain_url}")
        
        try:
            async with session.post(f"{self.brain_url}/chat/completions", json=payload) as response:
                if response.status != 200:
                    logger.error(f"💥 [Brain] Error {response.status}: {await response.text()}")
                    return

                async for line in response.content:
                    if not line: continue
                    line_str = line.decode("utf-8").strip()
                    if not line_str.startswith("data:"): continue
                    data_str = line_str[len("data:"):].strip()
                    if data_str == "[DONE]": break
                    
                    try:
                        chunk = json.loads(data_str)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        
                        # Handle Text Output (Log only)
                        text_content = delta.get("content", "")
                        if text_content:
                            # We don't yield text to the audio sink, just log the thought process
                            pass

                        # Handle Audio Output (Primary)
                        # WHY: Native Stage-Graph overlap ensures this streams as soon as the first 
                        # acoustic token is generated by the 'Talker' stage.
                        audio_data_b64 = delta.get("audio_data", "")
                        if audio_data_b64:
                            audio_bytes = base64.b64decode(audio_data_b64)
                            # WHY: We MUST update the sample rate to 24kHz (Fish-Speech standard) 
                            # to prevent pitch-shifting in the upsampler.
                            data.context.sample_rate = 24000
                            yield StupidData(content=audio_bytes, context=data.context, type="pcm")

                    except json.JSONDecodeError:
                        continue

        except Exception as e:
            logger.error(f"💥 [Brain] Connection failed: {e}")
            yield StupidData(content="ERR_BRAIN_OFFLINE", context=data.context, type="signal")


    async def close(self):
        if self.session:
            await self.session.close()
            logger.info("🔌 [vLLM Brain] Acoustic Bridge closed.")
