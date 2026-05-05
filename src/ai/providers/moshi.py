"""
WHY THIS FILE EXISTS:
Bridge implementation for the Moshi (Kyutai) / PersonaPlex-7B model.
Handles real-time audio streaming via a custom binary WebSocket protocol, 
including Opus encoding/decoding on the fly.

WHY USE BINARY WEBSOCKETS?
Because JSON overhead for high-frequency audio frames would explode 
latency and bandwidth usage. We use a single-byte 'kind' header followed 
by raw Opus payloads to keep the stream as tight as possible.
"""
import asyncio
from core.logger import setup_logger
import os
import torch
import numpy as np
import aiohttp
import time as _time
import sphn
from ..stupid_base import BridgeBase
from ...voice.resampler import UniversalResampler

logger = setup_logger("ai.providers.moshi")

DISCORD_SR = 48000
MOSHI_SR = 48000  # Corrected: PersonaPlex inputs/outputs at 48kHz native

class MoshiBridge(BridgeBase):
    """
    Implementation of the bridge for Moshi / PersonaPlex.
    Handles 24kHz audio frames and the custom binary WebSocket protocol.
    """
    def __init__(self, voice_preset: str, text_prompt: str, audio_source):
        super().__init__("moshi", MOSHI_SR)
        self.voice_preset = voice_preset
        self.text_prompt = text_prompt
        self.audio_source = audio_source
        
        self.audio_queue = asyncio.Queue()
        self.last_audio_time = _time.monotonic()
        self.running = False
        
        # Audio adapters
        if DISCORD_SR != MOSHI_SR:
            self.downsampler = UniversalResampler(DISCORD_SR, MOSHI_SR, name="MoshiInput")
            self.upsampler = UniversalResampler(MOSHI_SR, DISCORD_SR, name="MoshiOutput")
        else:
            self.downsampler = None
            self.upsampler = None
        
        self.ws = None
        self.session = None

    async def connect(self):
        """Connect to the Moshi server with system prompts."""
        server_url = os.getenv("MOSHI_SERVER_URL", "http://localhost:8998/chat")
        self.session = aiohttp.ClientSession()
        
        params = {
            "voice_prompt": self.voice_preset,
            "text_prompt": self.text_prompt,
        }
        
        logger.info(f"Connecting to Moshi server: {server_url} with preset {self.voice_preset}")
        self.ws = await self.session.ws_connect(server_url, params=params)
        
        # Wait for handshake
        logger.info("Waiting for Moshi handshake (processing system prompts)...")
        msg = await self.ws.receive(timeout=30)
        if msg.type == aiohttp.WSMsgType.BINARY and msg.data == b'\x00':
            logger.info("Moshi handshake received — conversation is live!")
            self.running = True
        else:
            logger.error(f"Unexpected handshake response: {msg}")
            raise RuntimeError("Moshi handshake failed")

    async def start_streaming(self):
        """Start the async send/recv loops."""
        if not self.running:
            return
        asyncio.create_task(self._send_loop())
        asyncio.create_task(self._recv_loop())

    async def _send_loop(self):
        """Sends Discord audio to Moshi server."""
        # Moshi uses Opus for the stream, but we send it at 24kHz usually?
        # Wait, the current PersonaPlex bridge sends Opus-encoded bytes.
        # But PersonaPlex's OpusWriter is set to 24000 or 48000?
        # Looking at original code: opus_writer = sphn.OpusStreamWriter(24000)
        
        opus_writer = sphn.OpusStreamWriter(MOSHI_SR)
        _frames_sent = 0
        _last_report = _time.monotonic()
        
        while self.running:
            try:
                incoming = await self.audio_queue.get()
                
                # 'incoming' is now expected to be a dict with 'audio' as raw PCM bytes
                pcm_bytes = incoming.get('audio')
                if not pcm_bytes:
                    continue
                
                pcm_array = np.frombuffer(pcm_bytes, dtype=np.int16)
                
                # Stereo 48kHz -> Mono 48kHz
                stereo = pcm_array.reshape(-1, 2)
                mono_pcm = stereo.mean(axis=1).astype(np.float32) / 32768.0
                
                if self.downsampler:
                    mono_tensor = torch.from_numpy(mono_pcm)
                    mono_pcm = self.downsampler(mono_tensor).flatten().numpy()
                
                # Encode and send
                opus_writer.append_pcm(mono_pcm)
                opus_bytes = opus_writer.read_bytes()
                if len(opus_bytes) > 0:
                    await self.ws.send_bytes(b'\x01' + opus_bytes)
                    _frames_sent += 1
                
                self.audio_queue.task_done()
                
                if _time.monotonic() - _last_report >= 5.0:
                    logger.debug(f"[SEND] Moshi: {_frames_sent} frames sent in 5s")
                    _frames_sent = 0
                    _last_report = _time.monotonic()
                    
            except Exception as e:
                logger.exception("Error in Moshi send loop")
                break

    async def _recv_loop(self):
        """Receives audio responses from Moshi server."""
        opus_reader = sphn.OpusStreamReader(DISCORD_SR) # Discord always wants 48k
        text_buffer = []
        pcm_buffer = np.array([], dtype=np.float32)
        FRAME_SIZE = DISCORD_SR // 50 # 20ms
        
        while self.running:
            try:
                msg = await self.ws.receive()
                if msg.type == aiohttp.WSMsgType.BINARY:
                    kind = msg.data[0]
                    payload = msg.data[1:]
                    
                    if kind == 0x01: # Audio
                        self.last_audio_time = _time.monotonic()
                        opus_reader.append_bytes(payload)
                        new_pcm = opus_reader.read_pcm()
                        
                        if new_pcm.shape[-1] > 0:
                            pcm_buffer = np.concatenate([pcm_buffer, new_pcm.flatten()])
                        
                        while len(pcm_buffer) >= FRAME_SIZE:
                            frame = pcm_buffer[:FRAME_SIZE]
                            pcm_buffer = pcm_buffer[FRAME_SIZE:]
                            # Convert to Discord's int16 stereo
                            int_audio = np.clip(frame * 32767, -32768, 32767).astype(np.int16)
                            stereo = np.column_stack([int_audio, int_audio])
                            self.audio_source.feed(stereo.tobytes())
                            
                    elif kind == 0x02: # Text
                        text = payload.decode('utf-8', errors='replace')
                        text_buffer.append(text)
                        if ' ' in text or len(text_buffer) > 20:
                            logger.info(f"[PersonaPlex] {''.join(text_buffer)}")
                            text_buffer.clear()
                            
                elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED):
                    break
            except Exception:
                logger.exception("Error in Moshi recv loop")
                break

    async def close(self):
        self.running = False
        if self.ws:
            await self.ws.close()
        if self.session:
            await self.session.close()
