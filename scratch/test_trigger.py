import os
import sys
import numpy as np
import wave

# Add src to path
sys.path.append(os.path.abspath("src"))

from voice.trigger import TriggerEngine, FishTemplateBackend
from core.logger import setup_logger

logger = setup_logger("test.trigger")

def test_template_loading():
    print("--- Testing Template Loading ---")
    backend = FishTemplateBackend(template_dir="src/assets/positives")
    backend._ensure_model()
    print(f"Loaded {len(backend.templates)} templates.")
    
    if len(backend.templates) > 0:
        print("✅ Template loading successful.")
    else:
        print("❌ No templates loaded.")

def test_trigger_detection():
    print("\n--- Testing Trigger Detection ---")
    backend = FishTemplateBackend(template_dir="src/assets/positives")
    
    # Load one of the templates to test against itself
    template_path = "src/assets/positives/hey_stupid_final.wav"
    if not os.path.exists(template_path):
        print(f"❌ Template {template_path} not found.")
        return

    with wave.open(template_path, 'rb') as w:
        orig_sr = w.getframerate()
        raw_data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32767.0
        # Resample to 16kHz for the backend
        duration = len(raw_data) / orig_sr
        num_target_samples = int(duration * 16000)
        x_orig = np.linspace(0, duration, len(raw_data))
        x_target = np.linspace(0, duration, num_target_samples)
        data_16k = np.interp(x_target, x_orig, raw_data).astype(np.float32)

    print(f"Feeding {len(data_16k)} samples ({duration:.2f}s) to detector...")
    
    # Feed in 1280 chunks
    found = False
    for i in range(0, len(data_16k), 1280):
        chunk = data_16k[i:i+1280]
        if len(chunk) < 1280:
            chunk = np.pad(chunk, (0, 1280 - len(chunk)))
        
        result = backend.detect(chunk)
        if result:
            print(f"✅ TRIGGER DETECTED: {result} at chunk {i//1280}")
            found = True
            break
    
    if not found:
        print("❌ Trigger NOT detected.")

if __name__ == "__main__":
    try:
        test_template_loading()
        test_trigger_detection()
    except Exception as e:
        print(f"💥 Test failed: {e}")
        import traceback
        traceback.print_exc()
