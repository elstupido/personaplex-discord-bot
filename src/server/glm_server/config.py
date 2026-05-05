import sys
import os
import types
import torch
from core.logger import setup_logger

logger = setup_logger("server.config")

# 1. RUNTIME SHIMS: Fix Python 3.12 compatibility
try:
    import pkg_resources
except ImportError:
    mock_pkg = types.ModuleType("pkg_resources")
    mock_pkg.declare_namespace = lambda name: None
    mock_pkg.get_distribution = lambda name: types.SimpleNamespace(version="0.0.0")
    sys.modules["pkg_resources"] = mock_pkg

# 2. MONKEY-PATCH: Fix ruamel.yaml
try:
    import ruamel.yaml
    if not hasattr(ruamel.yaml.Loader, 'max_depth'):
        ruamel.yaml.Loader.max_depth = None
except Exception:
    pass

# --- MODEL PATHS (The Physical Location of the AI Weights) ---
# WHY: These are the local paths where the multi-gigabyte weight tensors reside. 
# We use absolute paths to avoid ambiguity across different working directories.
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
GLM_MODEL_PATH = os.path.join(BASE_DIR, "model_cache/hub/models--THUDM--glm-4-voice-9b/snapshots/352e5a64063448d5e46394753fb65bf1dbab975b")
GLM_TOKENIZER_PATH = os.path.join(BASE_DIR, "model_cache/hub/models--THUDM--glm-4-voice-tokenizer/snapshots/a5f2404e63c84e92f5238908e1706316324ebafa")
GLM_DECODER_PATH = os.path.join(BASE_DIR, "model_cache/glm-4-voice-decoder")

# --- FISH AUDIO PATHS ---
FISH_CODEC_PATH = os.path.join(BASE_DIR, "checkpoints/s2-pro")

# 3. Path Configuration
GLM_PATHS = [
    os.path.join(BASE_DIR, "src/ai/providers/glm"),
    "/app/glm",
    "/app/glm/cosyvoice",
    "/app/glm/third_party/Matcha-TTS"
]
for p in GLM_PATHS:
    if os.path.exists(p) and p not in sys.path:
        sys.path.append(p)

# 4. Fish Audio Path Configuration
FISH_PATHS = [
    "/root/fish-speech"
]
for p in FISH_PATHS:
    if p not in sys.path:
        sys.path.append(p)

class Colors:
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    MAGENTA = '\033[95m'
    GRAY = '\033[90m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'

# 4. Model Constants
DISCORD_SAMPLE_RATE = 48000
GLM_MODEL_SAMPLE_RATE = 16000
DECODER_SAMPLE_RATE = 22050

# --- ACOUSTIC ALIGNMENT CONFIGURATION ---
class AlignmentConfig:
    """
    The Physics of Temporal & Spectral Synchronization.
    
    WHY THIS EXISTS:
    A neural voice pipeline is a 'Distributed Relay Race'. One model produces 
    features, another consumes them. If they don't agree on the fundamental 
    dimensions of time and frequency, the 'Handover' fails.
    
    CONCEPT: THE TEMPORAL RESOLUTION
    The choice of window and hop lengths determines our 'Temporal Resolution'. 
    A finer grid (higher rate) allows us to capture the micro-textures of 
    human phonemes, while a coarser grid (lower rate) reduces the 
    computational entropy of the system. This config acts as the 'Universal 
    Constants' that ensure all subsystems—Brain, Vault, and Larynx—are 
    aligned to the same coordinate system.
    """
    def __init__(self):
        # --- SPECTRAL PHYSICS ---
        self.mel_mean = 0.0
        self.mel_std = 1.0
        self.mel_scale = "ln" 
        self.mel_min_clamp = 1e-5
        self.sample_rate = 22050
        self.n_mels = 80
        self.f_min = 0.0
        self.f_max = 8000.0 
        self.n_fft = 1024
        self.hop_length = 441  # 20ms at 22050Hz (50Hz frame rate)
        self.win_length = 1024
        
        # --- IDENTITY & EMBEDDINGS ---
        self.emb_scalar = 35.0
        self.emb_l2_norm = True
        self.emb_mvn = True 
        self.whisper_sr = 16000
        self.audio_offset = 152353
        self.true_vocab_size = 168960
        
        # --- RESOURCE MANAGEMENT ---
        self.enable_diagnostics = True
        self.lazy_load_models = True
        self.enabled_vocoder_backends = ['glm', 'fish']
        
        # --- FISH AUDIO S2 EXTENSIONS ---
        self.fish_codec_path = "/root/personaplex-discord-bot/checkpoints/s2-pro/codec.pth"
        self.fish_llama_path = "/root/personaplex-discord-bot/checkpoints/s2-pro"

    def to_dict(self):
        return self.__dict__

    def update(self, params):
        for k, v in params.items():
            if hasattr(self, k):
                setattr(self, k, v)
                logger.info(f"[AlignmentConfig] Updated {k} = {v}")
