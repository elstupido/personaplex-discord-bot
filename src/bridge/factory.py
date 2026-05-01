import os
import logging
# Future adapters will be imported lazily within create_bridge

logger = logging.getLogger("bridge.factory")

def create_bridge(voice_preset: str, text_prompt: str, audio_source=None):
    """
    Factory to instantiate the correct AI bridge based on MODEL_TYPE env var.
    """
    model_type = os.getenv("MODEL_TYPE", "moshi").lower()
    
    logger.info(f"Instantiating bridge for model: {model_type}")
    
    if model_type == "moshi":
        try:
            from .moshi import MoshiBridge
            return MoshiBridge(voice_preset, text_prompt, audio_source)
        except ImportError as e:
            logger.error(f"Failed to load Moshi dependencies: {e}")
            raise
    elif model_type == "glm-4":
        from .glm import GLMBridge
        return GLMBridge(voice_preset, text_prompt, audio_source)
    elif model_type == "mini-omni":
        raise NotImplementedError("Mini-Omni2 bridge coming soon!")
    else:
        logger.warning(f"Unknown MODEL_TYPE '{model_type}'. Defaulting to GLM-4.")
        from .glm import GLMBridge
        return GLMBridge(voice_preset, text_prompt, audio_source)
