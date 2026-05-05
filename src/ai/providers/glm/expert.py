"""
WHY THIS FILE EXISTS: The GLMExpert is the neural conduit between the 'Stupid' Recursive ETL 
River and the high-fidelity GLM-4-Voice inference kernels.

THE NEURAL CONDUIT:
This expert acts as a protocol transmuter. It takes atomic 
StupidData particles from the data river and reconstructs them into 
the stateful payloads required by the GLM-4-Voice decoder.
"""

from ...stupid_base import StupidStep, StupidData, StupidRegistry, logger
from .core import GLMBridge

@StupidRegistry.register("glm-4")
class GLMExpert(StupidStep):
    """
    The GLM-4-Voice Expert Wrapper.
    
    THE CONVERSION LAYER:
    This expert transmutes StupidData (PCM) into the specific 
    spectral feature payloads required by the inference server.
    """
    def __init__(self, name: str):
        super().__init__(name)
        # THE RECURSION HAZARD:
        # We instantiate the bridge directly to avoid a circular dependency 
        # with the Factory. The Factory calls the Registry, which calls the 
        # Expert; if the Expert calls the Factory, the stack overflows.
        import os
        self.bridge = GLMBridge(
            voice_preset=os.getenv("VOICE_PRESET", "VARM3"),
            text_prompt=os.getenv("TEXT_PROMPT", ""),
            vocoder=os.getenv("VOCODER_BACKEND", "glm")
        )

    async def process(self, data: StupidData):
        """
        The Processing Conduit.
        
        WHY: We convert the StupidData back into the raw payload 
        the legacy bridge expects, then yield the response.
        """
        logger.debug(f"[GLMExpert] Processing packet from user {data.context.user_id}")
        
        # Prepare the turn payload
        payload = {
            'user_id': data.context.user_id,
            'audio': data.content,
            'duration_s': len(data.content) / (data.context.sample_rate * 2), # Assuming int16
            'event_type': 'TURN'
        }
        
        # Hand off to bridge (this is the legacy entry point)
        await self.bridge.send_audio_packet(payload)
        
        # Since the bridge is async and handles its own output, 
        # we don't yield anything back for now. 
        # (In Phase 3, the bridge will yield tokens back into the river).
        if False: yield # To keep it a generator
