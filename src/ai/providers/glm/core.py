"""
The Logistics Hub & Time-Lord of PersonaPlex.

WHY THIS FILE EXISTS:
This is the main entry point for the GLM Bridge. It has been refactored 
to be a 'Thin Orchestrator'—it holds the state (queues, sessions, config) 
but delegates all the heavy lifting to specialized functional modules.
"""

import asyncio
import aiohttp
import os
import base64
from .constants import logger, PROMPT_ECHO_MODE, RAW_ECHO_MODE, DIAGNOSTICS_ENABLED, LAZY_LOAD_MODELS
from .audio_ops import pipelined_upsampler_task, voice_state_context
from .inference_ops import post_to_server
from . import identity_ops
from .warmup import perform_full_warmup
from .diagnostics import run_raw_echo, run_prompt_echo
from ...stupid_base import BridgeBase
from ...components import AudioResampler, WhisperTokenizer, GLMAudioDecoder

class GLMBridge(BridgeBase):
    """
    The Logistics Hub & Time-Lord of PersonaPlex.
    
    WHY THIS CLASS EXISTS:
    This is the main entry point for the GLM Bridge. It has been refactored 
    to be a 'Thin Orchestrator'—it holds the state (queues, sessions, config) 
    but delegates all the heavy lifting to specialized functional modules.
    """
    def __init__(self, voice_preset=None, text_prompt=None, audio_source=None, vocoder='glm', server_url=None):
        super().__init__("glm-4", 22050)
        self.server_url = server_url or os.getenv("GLM_SERVER_URL", "http://127.0.0.1:10000")
        self.url = f"{self.server_url}/generate_complex_stream"
        self.warmup_url = f"{self.server_url}/warmup"
        self.audio_source = audio_source
        self.session = None
        self.audio_queue = asyncio.Queue()
        self.upsample_queue = asyncio.Queue(maxsize=20)
        self.is_running = False
        self.is_processing = False
        
        # Identity and Persona State
        # WHY? We need to track who the AI is pretending to be.
        self.voice_preset = voice_preset
        self.active_voice = voice_preset
        self.vocoder = vocoder
        self.diagnostics_enabled = DIAGNOSTICS_ENABLED
        self.lazy_load = LAZY_LOAD_MODELS
        
        # Audio Asset Buffers (Initialized hot during warmup)
        self.ding_pcm = b""
        self.turn_finalized_pcm = b""
        self.response_ready_pcm = b""
        
        # Injected Discord context (Set by VoiceServiceCog)
        self.vc = None
        self.text_channel = None
        
        # Pipeline Components
        self.resampler = AudioResampler()
        self.tokenizer = WhisperTokenizer()
        self.decoder = GLMAudioDecoder()
        
        # Prompt Logic
        self.system_prompt = self._build_system_prompt(text_prompt)

    def _build_system_prompt(self, text_prompt):
        base = ("User will provide you with a speech instruction. Do it step by step. "
                "First, think about the instruction and provide a brief plan, "
                "then follow the instruction and respond directly to the user.")
        return f"{base}\n\nPersona Instructions: {text_prompt}" if text_prompt else base

    # --- Lifecycle ---
    async def connect(self):
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None))
        self.is_running = True
        await perform_full_warmup(self)

    async def close(self):
        self.is_running = False
        if self.session: await self.session.close()

    async def start_streaming(self):
        asyncio.create_task(self._process_loop())
        asyncio.create_task(pipelined_upsampler_task(self))

    # --- Interaction ---
    async def send_audio_packet(self, data):
        if not self.is_running or self.is_processing or not self.audio_queue.empty():
            return
        await self.audio_queue.put(data)

    async def _process_loop(self):
        while self.is_running:
            try:
                incoming = await self.audio_queue.get()
                if incoming: await self._handle_turn(incoming)
            except Exception as e:
                logger.error(f"[GLMBridge.Core] Loop Error: {e}")
                await asyncio.sleep(1)

    async def _handle_turn(self, incoming):
        self.is_processing = True
        try:
            segments = incoming if isinstance(incoming, list) else [incoming]
            
            # Defensive Base64 encoding
            for seg in segments:
                if isinstance(seg.get('audio'), (bytes, bytearray)):
                    seg['audio'] = base64.b64encode(seg['audio']).decode('utf-8')

            # Build messages
            messages = [{"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": "", "audio_tokens": None, "audio_segments": segments}]

            # Diagnostics Mode
            if RAW_ECHO_MODE: return await run_raw_echo(self, segments)
            if PROMPT_ECHO_MODE: return await run_prompt_echo(self, segments)

            # Live Processing
            async with voice_state_context(self):
                if incoming.get('is_clone_reference', False):
                    await identity_ops.process_voice_clone(self, segments, incoming)
                else:
                    # Prepare message with full B64 for server-side tokenization
                    from .audio_ops import get_full_pcm
                    messages[-1]["audio_b64"] = base64.b64encode(get_full_pcm(segments)).decode('utf-8')
                    messages[-1]["audio_segments"] = []
                    
                    if self.turn_finalized_pcm and self.audio_source:
                        self.audio_source.feed_raw(self.turn_finalized_pcm)
                    
                    await post_to_server(self, messages)
        finally:
            self.is_processing = False

    # --- Identity Delegation ---
    # WHY DELEGATE? To keep this file small. The 'Identity' logic lives in identity_ops.py.
    async def list_voices(self): return await identity_ops.list_voices(self)
    async def switch_voice(self, name): return await identity_ops.switch_voice(self, name)

    async def play_ding(self):
        """Play the wake word confirmation sound."""
        if self.ding_pcm and self.audio_source:
            self.audio_source.feed_raw(self.ding_pcm)

    async def close(self):
        """Cleanly close the session."""
        if self.session:
            await self.session.close()
