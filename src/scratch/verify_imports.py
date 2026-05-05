import os
import sys

# Add src to sys.path
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

try:
    print("Checking core...")
    import core.bot
    import core.logger
    
    print("Checking ai...")
    import ai.factory
    import ai.orchestrator
    from ai.providers.glm.core import GLMBridge
    
    print("Checking voice...")
    import voice.cog
    import voice.engine
    
    print("Checking server...")
    # These might fail due to missing dependencies in this env (like transformers)
    # but let's see if the imports themselves resolve.
    import server.main
    import server.glm_server.app
    
    print("SUCCESS: All internal imports resolved.")
except Exception as e:
    print(f"FAILURE: {e}")
    import traceback
    traceback.print_exc()
