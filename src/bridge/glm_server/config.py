import sys
import types
import torch

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

# 3. Path Configuration
GLM_PATHS = [
    "/app/glm",
    "/app/glm/cosyvoice",
    "/app/glm/third_party/Matcha-TTS"
]
for p in GLM_PATHS:
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
