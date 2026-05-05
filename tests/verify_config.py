"""
WHY THIS FILE EXISTS:
The Sigil Registry Verification (Config Test). 📜🧪

WHY:
Centralizing config only works if the Factory actually respects the 
blueprints. This script proves that we can switch 'Recipes' at runtime 
just by changing the MODEL_TYPE.
"""

import asyncio
import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from ai.stupid_config import StupidConfig
from ai.stupid_factory import create_bridge
from ai.stupid_base import StupidRegistry, logger

async def verify_config():
    logger.info("🚀 Starting Sigil Registry Verification...")
    
    # 1. Test Configuration Defaults
    logger.info(f"Checking defaults: Model={StupidConfig.model_type()}, Vocoder={StupidConfig.vocoder_type()}")
    
    # 2. Test Blueprint Lookup
    blueprint = StupidConfig.get_active_blueprint()
    logger.info(f"✅ Active Blueprint: {blueprint.name} (Steps: {blueprint.steps})")
    
    # 3. Test Dynamic Factory Registration
    # We'll set the model to 'diagnostics' to test a specific recipe
    os.environ["MODEL_TYPE"] = "diagnostics"
    
    # Reset Registry for clean test (optional but good for clarity)
    # Actually, we'll just see if 'diagnostics' expert is registered by the factory
    
    logger.info("🧪 Testing Factory Dynamic Registration for 'diagnostics'...")
    bridge = await create_bridge(voice_preset="VARM3", text_prompt="test")
    await bridge.connect()
    
    # Verify the expert was registered
    try:
        expert = StupidRegistry.get_expert("diagnostics")
        logger.info(f"✅ Expert 'diagnostics' found in registry after factory call.")
    except KeyError:
        logger.error("💥 Expert 'diagnostics' NOT found in registry. Factory failed to load it!")
        return False
        
    # 4. Test Blueprint Application
    # The bridge (StupidBridgeAdapter) should have the diagnostics blueprint
    if bridge.blueprint.steps == ["diagnostics"]:
        logger.info("✅ BridgeAdapter correctly adopted the 'diagnostics' blueprint.")
    else:
        logger.error(f"💥 BridgeAdapter has wrong steps: {bridge.blueprint.steps}")
        return False

    # 5. Test VRAM Threshold
    limit = StupidConfig.get_vram_limit()
    logger.info(f"✅ VRAM Threshold confirmed at {limit*100}%")

    logger.info("\n🏆 All Config Signatures Validated. The Master Blueprint is operational. 📜✨")
    return True

if __name__ == "__main__":
    success = asyncio.run(verify_config())
    sys.exit(0 if success else 1)
