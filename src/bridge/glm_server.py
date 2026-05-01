"""
GLM-4-Voice Server Entry Point
Refactored into a modular package structure for better maintainability.
"""
import os
import sys

# Add the parent directories to sys.path so we can import the new package and utils
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)
sys.path.insert(0, src_dir)

from glm_server.config import Colors
from glm_server.engine import GLMVoiceEngine
from glm_server.app import app, GLMServerApp
import uvicorn

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10000)
    parser.add_argument("--model-path", type=str, default="THUDM/glm-4-voice-9b")
    parser.add_argument("--tokenizer-path", type=str, default="THUDM/glm-4-voice-tokenizer")
    parser.add_argument("--flow-path", type=str, default="/app/glm/glm-4-voice-decoder")
    args = parser.parse_args()

    # Automatic path correction for WSL/Docker environments
    if not os.path.exists(args.flow_path):
        alt_path = "/app/glm/glm-4-voice-decoder"
        if os.path.exists(alt_path):
            args.flow_path = alt_path

    print(f"{Colors.BLUE}{Colors.BOLD}")
    print("==================================================")
    print(" GLM-4-VOICE SERVER v2.0 (MODULAR) ")
    print("==================================================")
    print(f"{Colors.RESET}")

    engine = GLMVoiceEngine(args.model_path, args.tokenizer_path, args.flow_path)
    # Inject the engine into the app's global state (or used by the route)
    import glm_server.app as app_mod
    app_mod.server_app = GLMServerApp(engine)
    
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
