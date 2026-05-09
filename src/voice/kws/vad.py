"""
WHY THIS FILE EXISTS:
Stage 1: Voice Activity Detection (VAD).

WHY: 
Neural keyword spotting is expensive. We use Silero VAD as a primary gate
to ensure we only run inference when actual speech is present.
"""
import torch
import numpy as np
from core.logger import setup_logger

logger = setup_logger("voice.kws.vad")

class SileroVAD:
    """
    Lightweight VAD gating for the KWS pipeline.
    Expects 16kHz audio.
    """
    def __init__(self):
        # Load Silero VAD from torch hub (cached locally)
        self.model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad', 
            model='silero_vad', 
            trust_repo=True
        )
        (_, _, self.read_audio, *_) = utils
        self.model.eval()

    def is_speech(self, chunk: np.ndarray, threshold: float = 0.3) -> bool:
        """
        Returns True if the chunk contains speech.
        chunk: 1D float32 array @ 16kHz.
        """
        with torch.no_grad():
            # Silero expects (batch, samples)
            audio_tensor = torch.from_numpy(chunk).unsqueeze(0)
            speech_prob = self.model(audio_tensor, 16000).item()
            return speech_prob > threshold
