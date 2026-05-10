"""
WHY THIS FILE EXISTS:
The 'StupidConfig' (The Central Sigil Registry). 📜🏗️

WHY:
In a complex ETL system, hardcoded pipelines are the 'Spaghetti of the 
Modern Age'. If the bot needs to switch from GLM-4 to Moshi, or add a 
new diagnostic step, we shouldn't have to hunt through five different 
modules. 

StupidConfig acts as the 'Single Source of Truth' (SSOT) or the 'Master 
Blueprint'. It defines the 'Recipes' (Jobs) that the runner executes 
against the data river.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Any

@dataclass
class StupidBlueprint:
    """
    A declarative 'Recipe' for a specific model type.
    """
    name: str
    steps: List[str]
    description: str = ""

class StupidConfig:
    """
    The Orchestrator of Constants. 🧠
    
    WHY NOT JUST ENV VARS?
    Environment variables are great for secrets and deployment flags, but 
    they are 'Flat'. A Blueprint requires structure (a list of steps). 
    StupidConfig bridges the gap between 'Flat Env' and 'Structured Logic'.
    """

    @classmethod
    def model_type(cls) -> str:
        return os.getenv("MODEL_TYPE", "Qwen/Qwen2.5-Omni-7B-Instruct").lower()
    
    @classmethod
    def vocoder_type(cls) -> str:
        return os.getenv("VOCODER_TYPE", "glm").lower()
    
    # --- RESOURCE THRESHOLDS ---
    VRAM_THRESHOLD = float(os.getenv("VRAM_THRESHOLD", "0.90"))
    
    # --- THE SIGIL BLUEPRINTS ---
    BLUEPRINTS = {
        "glm-4": StupidBlueprint(
            name="GLM-4 Standard Flow",
            steps=["downsampler", "whisper_tokenizer", "glm-4", "upsampler"],
            description="The high-fidelity flow: Downsample -> Tokenize -> Inference -> Upsample."
        ),
        "moshi": StupidBlueprint(
            name="Moshi Real-time",
            steps=["resampler", "moshi"],
            description="--model Qwen/Qwen2.5-Omni-7B-Instruct --port 8000 --host 0.0.0.0 --trust-remote-code --gpu-memory-utilization 0.85 --max-model-len 4096 --enforce-eager"
        ),
        "mini-omni": StupidBlueprint(
            name="Mini-Omni2 Experimental",
            steps=["resampler", "mini-omni"],
            description="Next-gen multimodal interaction."
        ),
        "vllm-omni": StupidBlueprint(
            name="vLLM-Omni Disaggregated",
            steps=["downsampler", "vllm-omni", "upsampler"],
            description="High-performance stage-graph serving via vLLM-brain. Upsampler converts 24kHz→48kHz stereo for Discord."
        ),
        "qwen-fish": StupidBlueprint(
            name="Sacred Fish Hybrid",
            steps=["downsampler", "vllm-omni-asr", "vllm_reasoning", "vllm-omni-tts", "upsampler"],
            description="ASR (SenseVoice GPU) -> Think (vLLM GPU) -> Talk (Fish 19GB GPU) -> Upsample (Discord)."
        ),
        "diagnostics": StupidBlueprint(
            name="Maintenance Mode",
            steps=["diagnostics"],
            description="Used by verify_foundation.py to ensure the dam isn't leaking."
        )
    }

    # --- THE EXPERT MAP ---
    EXPERT_MODULES = {
        "diagnostics": "ai.audit.diagnostics",
        "resampler": "ai.transform.resampler",
        "downsampler": "ai.transform.resampler",
        "upsampler": "ai.transform.resampler",
        "whisper_tokenizer": "ai.transform.tokenizer",
        "glm-4": "ai.transform.glm",
        "moshi": "ai.transform.moshi",
        "mini-omni": "ai.transform.mini_omni",
        "vllm-omni": "ai.transform.vllm_omni",
        "vllm-omni-asr": "ai.transform.vllm_omni_asr",
        "vllm-omni-tts": "ai.transform.vllm_omni_tts",
        "vllm_reasoning": "ai.transform.vllm_reasoning"
    }

    @classmethod
    def get_active_blueprint(cls) -> StupidBlueprint:
        model = cls.model_type()
        return cls.BLUEPRINTS.get(model, cls.BLUEPRINTS["glm-4"])

    @classmethod
    def get_vram_limit(cls) -> float:
        return cls.VRAM_THRESHOLD

    @classmethod
    def get_all_experts_for_model(cls, model_type: str = None) -> List[str]:
        mt = model_type or cls.model_type()
        blueprint = cls.BLUEPRINTS.get(mt, cls.BLUEPRINTS["glm-4"])
        return [s for s in blueprint.steps if not s.startswith("$")]
