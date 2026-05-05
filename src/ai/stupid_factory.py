"""
WHY THIS FILE EXISTS:
The Factory Pattern implementation for AI bridges. It allows the bot to 
switch between different AI backends (GLM-4, Moshi, etc.) based on 
environment configuration without changing the high-level Discord logic.

WHY LAZY IMPORTS?
AI model bridges often have heavy dependencies (like transformers or moshi). 
By importing them only when needed inside create_bridge(), we keep the 
initial bot startup lightweight and avoid loading unused models into VRAM.
"""
import os
import importlib
import asyncio
from core.logger import setup_logger
from .stupid_base import StupidRegistry, logger
from .stupid_config import StupidConfig

logger = setup_logger("ai.stupid_factory")

async def create_bridge(voice_preset: str, text_prompt: str, audio_source=None, vocoder: str = 'glm'):
    """
    Factory to instantiate the correct AI bridge or StupidExpert.
    
    WHY DATA-DRIVEN?
    By consulting the StupidConfig blueprint, we can dynamically load the 
    required experts for ANY model without changing this factory's code.
    """
    model_type = StupidConfig.model_type()
    blueprint = StupidConfig.get_active_blueprint()
    
    # 1. Dynamic Expert Registration 🔔
    # WHY: We iterate through the blueprint steps and import the corresponding 
    # modules. This triggers the @StupidRegistry.register decorators.
    experts_to_load = StupidConfig.get_all_experts_for_model(model_type)
    for expert_id in experts_to_load:
        module_path = StupidConfig.EXPERT_MODULES.get(expert_id)
        if module_path:
            try:
                # WHY TO_THREAD? Imports of AI modules (transformers/torch) are slow 
                # and can block the Discord heartbeats.
                await asyncio.to_thread(importlib.import_module, module_path)
                logger.debug(f"✨ [Factory] Registered expert: {expert_id}")
            except Exception as e:
                logger.error(f"💥 [Factory] Failed to load expert '{expert_id}': {e}")

    logger.info(f"🚀 Instantiating bridge for model: {model_type} ({blueprint.name})")
    
    # 2. Try the New StupidRunner path first 🌊
    try:
        # Check if the primary model expert is registered
        StupidRegistry.get_expert(model_type)
        logger.info(f"✅ [Factory] Found Expert in Registry: {model_type}. Using StupidBridgeAdapter.")
        from .bridge_adapter import StupidBridgeAdapter
        return StupidBridgeAdapter(model_type)
    except (ValueError, KeyError):
        logger.warning(f"⚠️ [Factory] Expert '{model_type}' not found in Registry. Falling back to legacy bridge.")

    # 3. Legacy Fallbacks (Deprecated) 🏛️
    if model_type == "moshi":
        try:
            from .providers.moshi import MoshiBridge
            return MoshiBridge(voice_preset, text_prompt, audio_source)
        except ImportError as e:
            logger.error(f"💥 Failed to load Moshi dependencies: {e}")
            raise
    elif model_type == "glm-4":
        from .providers.glm.core import GLMBridge
        return GLMBridge(voice_preset, text_prompt, audio_source, vocoder=vocoder)
    else:
        logger.warning(f"⚠️ Unknown MODEL_TYPE '{model_type}'. Defaulting to GLM-4 legacy.")
        from .providers.glm.core import GLMBridge
        return GLMBridge(voice_preset, text_prompt, audio_source, vocoder=vocoder)
