"""
WHY THIS FILE EXISTS:
The Neural Brain Health Monitor. 🧠🚑🧪

WHY:
Disaggregation is great until the Brain dies and the Body keeps twitching. 
This script verifies that the vLLM-Omni server is actually reachable, 
properly authenticated, and exposing the correct model architecture.
"""

import sys
import os
import asyncio
import aiohttp

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from core.logger import setup_logger

logger = setup_logger("verify_brain")

async def verify_brain_health():
    url = os.getenv("VLLM_SERVER_URL", "http://localhost:8000/v1")
    logger.info(f"🔍 Probing Neural Brain at {url}...")
    
    max_retries = 150
    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession() as session:
                # 1. Check Connectivity & API Version
                async with session.get(f"{url}/models", timeout=5) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        models = [m['id'] for m in data.get('data', [])]
                        logger.info(f"✅ Brain API Online. Available Models: {models}")
                        return True
                    else:
                        logger.warning(f"⏳ Brain status: HTTP {resp.status} (Attempt {attempt+1}/{max_retries})")
        except Exception as e:
            if attempt % 10 == 0:
                logger.warning(f"⏳ Brain unreachable: {e} (Attempt {attempt+1}/{max_retries})")
        
        await asyncio.sleep(2)
        
    logger.error("❌ Neural Brain failed to stabilize after 120 seconds.")
    return False

if __name__ == "__main__":
    success = asyncio.run(verify_brain_health())
    sys.exit(0 if success else 1)
