"""
WHY THIS FILE EXISTS:
The 'WhisperTokenizerExpert' (The Semantic Phoneme Decoder). 🎙️

WHY:
Raw audio is too dense. This expert compresses the 50Hz PCM river into 
a sequence of semantic tokens that the LLM expert can actually 'read'. 
It bridges the gap between Acoustic Physics and Linguistic Intent.
"""

import asyncio
from typing import Generator, AsyncGenerator
from ..stupid_base import StupidStep, StupidData, StupidRegistry, logger
from ..components import WhisperTokenizer

@StupidRegistry.register("whisper_tokenizer")
class WhisperTokenizerExpert(StupidStep):
    """
    The Information Compressor. 🧠
    """
    def __init__(self, name: str):
        super().__init__(name)
        # Initialize the machinery. 🤖
        # WHY: Tokenizers are heavy (VQ Encoders). We load once and keep warm.
        self.tokenizer = WhisperTokenizer(device="cuda")
        logger.info(f"✨ [WhisperTokenizerExpert] '{name}' initialized. Semantic gates are open.")

    def warmup(self):
        """Pre-heat the semantic machinery."""
        # WHY: Tokenizers carry heavy VQ Encoders and massive imports.
        # We MUST do this in the warmup phase to avoid stalling the heartbeat.
        self.tokenizer.warmup()

    async def process(self, data: StupidData) -> AsyncGenerator[StupidData, None]:
        """
        Compress PCM into Tokens. 🗜️
        
        WHY: 
        We expect 16kHz float32 PCM (typically from the ResamplerExpert). 
        The output is a list of VQ tokens (type='tokens').
        """
        if data.type != "pcm":
            logger.warning(f"⚠️ [TokenizerExpert] Received non-PCM data: {data.type}")
            yield data
            return

        # Ensure we are at 16kHz (Standard for Whisper VQ)
        if data.context.sample_rate != 16000:
            logger.warning(f"⚠️ [TokenizerExpert] Audio is {data.context.sample_rate}Hz. Expected 16000Hz. Performance may suffer!")

        logger.debug(f"🎙️ [Tokenizer] Extracting features from {len(data.content)} samples...")
        
        # 1. Extract Mel Features
        features = await asyncio.to_thread(self.tokenizer.extract_features, data.content)
        
        # 2. Vector Quantize to Tokens
        tokens = await asyncio.to_thread(self.tokenizer.get_vq_tokens, features)
        
        if not tokens:
            logger.debug("📭 [Tokenizer] No tokens extracted (silence or error).")
            return

        # 3. Yield the Tokens
        logger.debug(f"✅ [Tokenizer] Produced {len(tokens)} semantic tokens. 🧠")
        
        # Create a new StupidData packet for the tokens
        yield StupidData(
            content=tokens,
            context=data.context,
            type="tokens"
        )
