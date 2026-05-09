import os
import sys
import runpy

# WHY THIS FILE EXISTS:
# The Cleanroom Wrapper. 🛡️🧠
#
# WHY:
# We need LD_PRELOAD to stabilize the Torch ABI in the main process. 
# However, if we leave it in the environment, every child process (like 'sh') 
# will try to load it and crash with 'undefined symbol: PyInstanceMethod_Type'.
#
# This wrapper unsets the variable IMMEDIATELY after Python starts, 
# ensuring the main process is patched but its children are safe.

if "LD_PRELOAD" in os.environ:
    print(f"🧬 [Brain-Wrapper] Internalizing surgery: unsetting LD_PRELOAD for child processes...")
    del os.environ["LD_PRELOAD"]

# WHY: We use runpy to execute the vllm-omni entrypoint as if it were the main script.
# This preserves the CLI arguments passed via docker-compose.
if __name__ == "__main__":
    # We must ensure the arguments are passed through
    sys.argv[0] = "vllm_omni.entrypoints.openai.api_server"
    runpy.run_module("vllm_omni.entrypoints.openai.api_server", run_name="__main__")
