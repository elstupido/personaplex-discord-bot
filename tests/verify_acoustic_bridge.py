"""
WHY THIS FILE EXISTS:
The Acoustic Bridge Integrity Test. 🧠🌉🧪

WHY:
In the disaggregated architecture, the connection between the Body and 
the Brain is the most fragile link. If the 'Acoustic Bridge' (HTTP/gRPC) 
stalls the event loop while waiting for a response, the 50Hz heartbeat 
dies. This test ensures the bridge is non-blocking and handles SSE 
streaming correctly.
"""

import asyncio
import time
import sys
import os
import json
import numpy as np
from aiohttp import web

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from ai.stupid_base import StupidData, AcousticContext, logger
from ai.transform.vllm_omni import VLLMOmniExpert
from tests.verify_heartbeat import heartbeat_monitor

async def mock_vllm_server(request):
    """Simulates vLLM-Omni streaming audio responses."""
    resp = web.StreamResponse(
        status=200,
        reason='OK',
        headers={'Content-Type': 'text/event-stream'}
    )
    await resp.prepare(request)
    
    # Simulate a few chunks of 'audio' data in SSE format
    for i in range(5):
        chunk = {
            "choices": [{
                "delta": {
                    "audio": [0.1] * 100 # Mock PCM
                }
            }]
        }
        await resp.write(f"data: {json.dumps(chunk)}\n\n".encode('utf-8'))
        await asyncio.sleep(0.02) # Simulate 50Hz neural generation
        
    await resp.write(b"data: [DONE]\n\n")
    return resp

async def run_bridge_test():
    logger.info("🚀 Starting Acoustic Bridge Integrity Test...")
    
    # 1. Start Mock Server
    app = web.Application()
    app.router.add_post('/v1/chat/completions', mock_vllm_server)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '127.0.0.1', 8001)
    await site.start()
    
    logger.info("📡 Mock Brain online at http://127.0.0.1:8001")
    
    # 2. Setup Expert
    os.environ["VLLM_SERVER_URL"] = "http://127.0.0.1:8001/v1"
    expert = VLLMOmniExpert()
    
    # 3. Start Heartbeat Monitor
    monitor_ticks = 0
    max_stall = 0
    async def monitor():
        nonlocal monitor_ticks, max_stall
        async for t, ms in heartbeat_monitor():
            monitor_ticks = t
            max_stall = ms
            
    monitor_task = asyncio.create_task(monitor())
    
    # 4. Stress the Bridge
    logger.info("🧪 Pumping 20ms audio frames across the bridge...")
    
    dummy_pcm = np.zeros(960, dtype=np.float32) # 20ms @ 48kHz
    start_time = time.time()
    start_ticks = monitor_ticks
    
    packet_count = 0
    async for packet in expert.process(StupidData(content=dummy_pcm, context=AcousticContext(), type="pcm")):
        packet_count += 1
        if packet_count % 2 == 0:
            logger.debug(f"   [Bridge] Received packet {packet_count} from Brain...")
            
    total_time = time.time() - start_time
    end_ticks = monitor_ticks
    
    # 5. Metrics
    expected_ticks = total_time / 0.01
    actual_ticks = end_ticks - start_ticks
    hti = (actual_ticks / expected_ticks) * 100 if expected_ticks > 0 else 100
    
    logger.info(f"[METRIC] packets_received={packet_count} hti={hti:.2f} max_stall_ms={max_stall*1000:.1f}")
    
    # Clean up
    await expert.close()
    await runner.cleanup()
    monitor_task.cancel()
    
    if hti < 90:
        logger.error(f"💥 BRIDGE FAILURE: The network IO stalled the heartbeat! HTI={hti:.1f}%")
        return False
        
    if packet_count == 0:
        logger.error("💥 BRIDGE FAILURE: No packets received from mock brain!")
        return False
        
    logger.info("✅ Acoustic Bridge verified: Loop remained responsive during disaggregated IO.")
    return True

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    success = asyncio.run(run_bridge_test())
    sys.exit(0 if success else 1)
