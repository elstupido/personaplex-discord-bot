"""
The Static Reality of GLM.

WHY THIS FILE EXISTS:
Constants are the 'laws of physics' for our bridge. By centralizing them, 
we ensure that every deconstructed module is operating under the same 
universal truths, such as whether we are in 'Echo Mode' or where the 
inference gateway lives.
"""

import os
from core.logger import setup_logger

logger = setup_logger("ai.providers.glm")

# ---------------------------------------------------------------------------
# Feature flags
# ---------------------------------------------------------------------------

# WHY PROMPT_ECHO_MODE: 
# Sometimes you just want to hear the AI's internal representation 
# without waiting for the LLM to hallucinate. This skips the 'Big Brain' 
# and loops tokens straight back to the vocoder.
PROMPT_ECHO_MODE: bool = os.getenv("PROMPT_ECHO_MODE", "false").lower() == "true"

# WHY RAW_ECHO_MODE: 
# For when you suspect the microphone is absolute garbage. 
# Plays back intake bytes with zero processing so you can hear the noise floor.
RAW_ECHO_MODE: bool = os.getenv("RAW_ECHO_MODE", "false").lower() == "true"

# WHY DIAGNOSTICS: 
# Because neural alignment is hard, and sometimes you need to see the 
# RMS energy of a clone sample to understand why the AI sounds like a robot.
DIAGNOSTICS_ENABLED: bool = os.getenv("GLM_DIAGNOSTICS", "true").lower() == "true"

# WHY LAZY_LOAD: 
# Because we respect your RAM. If true, we don't load the weights until 
# someone actually tries to speak.
LAZY_LOAD_MODELS: bool = os.getenv("LAZY_LOAD_MODELS", "true").lower() == "true"

# Colors for pedagogical logging
class Colors:
    CYAN    = "\033[96m"
    MAGENTA = "\033[95m"
    RED     = "\033[91m"
    RESET   = "\033[0m"
