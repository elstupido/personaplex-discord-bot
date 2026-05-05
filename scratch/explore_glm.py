import sys
import os

GLM_PATHS = [
    "/app/glm",
    "/app/glm/cosyvoice",
    "/app/glm/third_party/Matcha-TTS"
]
for p in GLM_PATHS:
    if p not in sys.path:
        sys.path.append(p)

try:
    import flow_inference
    print(f"flow_inference path: {flow_inference.__file__}")
    
    # Try to find other modules in the same root
    root = os.path.dirname(os.path.dirname(flow_inference.__file__))
    print(f"GLM root: {root}")
    
    import glob
    print("--- Files in GLM root ---")
    for f in glob.glob(os.path.join(root, "*.py")):
        print(os.path.basename(f))
        
    print("--- Subdirectories in GLM root ---")
    for d in glob.glob(os.path.join(root, "*/")):
        print(os.path.basename(os.path.normpath(d)))

except Exception as e:
    print(f"Error: {e}")
