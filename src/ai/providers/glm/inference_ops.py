"""
The Neural Diplomat.

WHY THIS FILE EXISTS:
This module handles the 'Negotiation' with the Inference Gateway. It 
manages the POST requests, parses the streaming NDJSON response, and 
routes chunks of data (text or audio) to their respective destinations.
"""

import base64
import json
import asyncio
from .constants import logger, Colors
from ...orchestrator import orchestrator

async def post_to_server(bridge, messages):
    """Send the structured turn to the server and handle the response stream."""
    payload = {
        "messages": messages,
        "vocoder": bridge.vocoder
    }
    try:
        async with bridge.session.post(bridge.url, json=payload) as response:
            if response.status != 200:
                logger.error(f"[GLMBridge.Inference] HTTP {response.status} - Turn skipped.")
                return
            await handle_server_stream(bridge, response)
    except Exception as e:
        logger.error(f"[GLMBridge.Inference] Connection lost: {e}")

async def handle_server_stream(bridge, response):
    """Parse NDJSON and dispatch chunks."""
    if bridge.audio_source:
        # INTERRUPT: Stop any previous sounds (like the 'Finalized' glitch)
        bridge.audio_source.clear()
        bridge.audio_source.prepare_for_model()
        
        # WHY PHASE 3? To give the user a 'Response Ready' chime.
        if bridge.response_ready_pcm:
            bridge.audio_source.feed_raw(bridge.response_ready_pcm)
    
    full_text         = ""
    total_audio_bytes = 0
    audio_tokens      = []

    async for line in response.content:
        if not line: continue
        try:
            chunk = json.loads(line)
            # WHY DISPATCH? To separate audio vs text vs errors.
            full_text, total_audio_bytes = await dispatch_chunk(
                bridge, chunk, full_text, total_audio_bytes, audio_tokens
            )
        except (json.JSONDecodeError, KeyError):
            pass  # Expected delimiters

    # Finalize the turn in the orchestrator
    _finalize_turn(bridge, full_text, total_audio_bytes, audio_tokens)

async def dispatch_chunk(bridge, chunk, full_text, total_audio_bytes, audio_tokens):
    """Route a single NDJSON chunk."""
    if "audio_chunk" in chunk:
        audio_22k = base64.b64decode(chunk["audio_chunk"])
        # Pipeline the upsampling to avoid blocking the network loop
        await bridge.upsample_queue.put(audio_22k)
        total_audio_bytes += int(len(audio_22k) * (48000 / 22050) * 2)
        
    elif "audio_tokens" in chunk:
        audio_tokens.extend(chunk["audio_tokens"])
        
    elif "text_chunk" in chunk:
        text = chunk["text_chunk"]
        # Filter internal model noise tags
        if any(tag in text for tag in ("streaming_transcription", "♪")):
            return full_text, total_audio_bytes
        full_text += text
        print(text, end="", flush=True) # Live console stream
        
    elif "error" in chunk:
        logger.error(f"\n{Colors.RED}[Server Error] {chunk['error']}{Colors.RESET}")
        
    return full_text, total_audio_bytes

def _finalize_turn(bridge, full_text, total_audio_bytes, audio_tokens):
    """Clean up and record history."""
    clean_text = full_text.replace("♪", "").replace("streaming_transcription", "").strip()
    if not clean_text and total_audio_bytes:
        clean_text = "[Audio Response]"
    elif not clean_text:
        clean_text = "[No response]"
        
    orchestrator.add_assistant_response(clean_text, audio_tokens=audio_tokens)
    
    # WHY ASYNC TASK? So we don't block the network loop while waiting for 
    # Discord's rate-limited message API.
    if bridge.text_channel and clean_text not in ("[Audio Response]", "[No response]"):
        asyncio.create_task(bridge.text_channel.send(f"**PersonaPlex:** {clean_text}"))
        
    logger.info(f"Turn Complete: {total_audio_bytes}B audio, {len(full_text)} chars.")
