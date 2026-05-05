"""
The Ghost of Transcription Past.

WHY THIS FILE EXISTS:
Backward compatibility is a necessary evil. This module holds the 'Stubs' 
for components we have killed or offloaded, ensuring that the rest of the 
bot doesn't throw a tantrum when it looks for an 'Ear' that is now 
living on the inference server.
"""

class Transcriber:
    """
    The Legacy Ear (DEPRECATED).
    
    WHY IT IS DISABLED:
    To be 'Respectful' of VRAM, we no longer run Whisper inside the Discord 
    bot process. All acoustic analysis has been migrated to the 
    Inference Gateway (app.py). 
    """
    def __init__(self):
        self._model = None
        self._backend = "disabled"

    def transcribe(self, stereo_pcm: bytes) -> str:
        # WHY: Because why do it yourself when the server has a 5090?
        return "[Analysis Offloaded to Server]"

    def warmup(self) -> str:
        return "[Warmup Deferred to Server]"
