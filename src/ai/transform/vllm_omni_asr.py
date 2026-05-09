"""
WHY THIS FILE EXISTS:
The 'vLLM-Omni' ASR Expert (The Remote Ears). 👂🧠

WHY:
In the disaggregated 'Sacred Fish' architecture, the transcription is handled 
by Stage 0 of the vLLM-brain (SenseVoice Small). This expert packages raw 
PCM audio into a multimodal request and dispatches it to the inference 
server.

DISAGREGGATED FLOW:
1. This expert (ASR) -> Returns Text.
2. CPUReasoningExpert -> Returns Text.
3. TTSExpert -> Returns Audio.
"""

import websockets
import asyncio
import base64
import json
import os
import io
import wave
import numpy as np
from typing import AsyncGenerator

from ..stupid_base import StupidStep, StupidData, StupidRegistry, logger

@StupidRegistry.register("vllm-omni-asr")
class ASRExpert(StupidStep):
    """
    Expert for transcribing audio via vLLM-Omni Stage 0 (SenseVoice).
    """
    def __init__(self, name: str):
        super().__init__(name)
        # WHY: Target the dedicated bot-ears transcriber service over WebSockets.
        self.ws_url = os.getenv("BOT_EARS_URL", "ws://localhost:8765/?format=wav&rate=16000")
        self.ws = None
        logger.info(f"✨ [ASR] Expert '{name}' pointing to {self.ws_url}")

    async def connect(self):
        if self.ws is None or self.ws.closed:
            self.ws = await websockets.connect(self.ws_url)

    async def process(self, data: StupidData) -> AsyncGenerator[StupidData, None]:
        await self.connect()

        if data.type != "pcm" or data.content is None:
            yield data
            return

        # 1. Normalize for SenseVoice
        # WHY: SenseVoice Stage 0 expects 16kHz float32 audio.
        audio_content = data.content
        if isinstance(audio_content, bytes):
            pcm_int16 = np.frombuffer(audio_content, dtype=np.int16)
        else:
            pcm_int16 = (np.asarray(audio_content) * 32767).clip(-32768, 32767).astype(np.int16)
        
        # Diagnostic: Check if we are actually sending sound
        peak = np.abs(pcm_int16).max() if len(pcm_int16) > 0 else 0
        logger.info(f"👂 [ASR] Signal Peak: {peak} (Threshold for life: >1000)")

        # 2. Package for bot-ears 👂
        # WHY: Use the 'wave' module to create a standard, compliant WAV header
        # torchaudio on the server will automatically detect the sample rate from the header.
        wav_buf = io.BytesIO()
        with wave.open(wav_buf, 'wb') as bw:
            bw.setnchannels(1)
            bw.setsampwidth(2)
            bw.setframerate(16000) # Assumes incoming data is already 16kHz downmixed from trigger engine
            bw.writeframes(pcm_int16.tobytes())
        
        wav_data = wav_buf.getvalue()

        logger.info(f"👂 [ASR] Dispatching to 'bot-ears' WebSocket...")
        
        try:
            await self.ws.send(wav_data)
            result_str = await self.ws.recv()
            result = json.loads(result_str)
            
            if "error" in result:
                logger.error(f"❌ [ASR] Bot-Ears Error: {result['error']}")
                data.content = ""
            else:
                text = result.get("text", "").strip()
                if text:
                    logger.info(f"👂 [ASR] Transcribed: \"{text}\"")
                    data.content = text
                else:
                    logger.warning(f"⚠️ [ASR] Empty result from SenseVoice.")
                    data.content = ""
            
            data.type = "text"
            
        except websockets.exceptions.ConnectionClosed:
            logger.error(f"💥 [ASR] WebSocket closed unexpectedly. Reconnecting next turn...")
            self.ws = None
            data.content = ""
            data.type = "text"
        except Exception as e:
            logger.error(f"💥 [ASR] Remote call failed: {e}")
            data.type = "text"
            data.content = ""
        
        yield data

    async def close(self):
        if self.ws:
            await self.ws.close()
