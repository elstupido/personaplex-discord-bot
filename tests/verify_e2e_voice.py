"""
WHY THIS FILE EXISTS:
The End-to-End Voice Integrity Test. 🧠🎙️🧪

WHY:
Verifying the Brain is online is good. Verifying the Body can walk is good. 
But verifying the 'Soul' (the E2E interaction) is what actually matters. 
This test simulates a voice turn, sends it to the disaggregated brain, 
and captures the resulting audio to ensure the river is fully connected.
"""

import asyncio
import os
import sys
import time
import numpy as np
from unittest.mock import MagicMock

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from ai.stupid_factory import create_bridge
from ai.stupid_base import logger
from core.logger import setup_logger

logger = setup_logger("verify_e2e")

async def verify_e2e_voice():
    logger.info("============================================================")
    logger.info("STUPIDBOT E2E VOICE INTEGRITY AUDIT")
    logger.info("============================================================")

    # 1. Setup Environment
    os.environ["MODEL_TYPE"] = "vllm-omni"
    # Ensure we point to the container network if running in 'test' service, 
    # or localhost if running locally with port mapping.
    # The 'test' service in docker-compose sees 'vllm-omni' host.
    os.environ["VLLM_SERVER_URL"] = os.getenv("VLLM_SERVER_URL", "http://vllm-omni:8000/v1")

    # 2. Instantiate the Bridge via Factory
    logger.info("Phase 1: Instantiating Acoustic Bridge...")
    bridge = await create_bridge(voice_preset="VARM3", text_prompt="test")
    
    # 3. Mock the AudioSource (The Output Sink)
    mock_source = MagicMock()
    received_audio = []
    
    def mock_feed(data):
        logger.info(f"Received {len(data)} bytes of audio from Brain!")
        received_audio.append(data)
        
    mock_source.feed = mock_feed
    bridge.audio_source = mock_source

    # 4. Connect & Warmup
    logger.info("Phase 2: Connecting & Warming up Experts...")
    try:
        await bridge.connect()
    except Exception as e:
        logger.error(f"Connection Failed: {e}")
        return False

    # 5. Simulate a Multi-Part Voice Turn (Turn Finalization Test)
    logger.info("🎙️ Phase 3: Testing Turn Finalization (Multi-part)...")
    
    # Send 2 partials
    for i in range(2):
        logger.info(f"📤 [E2E] Sending Partial Chunk {i+1}...")
        partial_payload = {
            'audio': np.zeros(16000, dtype=np.int16).tobytes(), # ~330ms
            'user_id': 12345,
            'is_partial': True
        }
        await bridge.send_audio_packet(partial_payload)
        
        # Verify no audio received yet
        if len(received_audio) > 0:
            logger.error("💥 Finalization Failure: Received audio before turn was finalized!")
            return False

    # Send finalization packet
    logger.info("📤 [E2E] Sending Finalization Packet...")
    final_payload = {
        'audio': np.zeros(16000, dtype=np.int16).tobytes(),
        'user_id': 12345,
        'is_partial': False
    }
    
    start_time = time.time()
    await bridge.send_audio_packet(final_payload)
    elapsed = (time.time() - start_time)
    
    # 6. Final Audit
    logger.info("============================================================")
    if len(received_audio) > 0:
        logger.info(f"E2E SUCCESS: Received {len(received_audio)} audio chunks.")
        logger.info(f"Turn Latency: {elapsed:.2f}s")
        logger.info("============================================================")
        return True
    else:
        logger.error("E2E FAILURE: Brain was silent. No audio reached the sink.")
        logger.info("============================================================")
        return False

if __name__ == "__main__":
    import logging
    # Enable debug for more detail
    logging.getLogger("ai.transform.vllm_omni").setLevel(logging.DEBUG)
    
    success = asyncio.run(verify_e2e_voice())
    sys.exit(0 if success else 1)
