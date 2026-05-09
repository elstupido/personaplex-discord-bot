"""
WHY THIS FILE EXISTS:
Stage 2: Neural Keyword Spotting (KWS) Backend.

WHY:
This module encapsulates the Sherpa-ONNX Zipformer engine. It manages 
transducer state and handles phoneme-level keyword matching.
"""
import os
import sherpa_onnx
import numpy as np
from abc import ABC, abstractmethod
from .vad import SileroVAD
from core.logger import setup_logger

logger = setup_logger("voice.kws.backend")

class TriggerBackend(ABC):
    @abstractmethod
    def detect(self, audio_data: np.ndarray) -> str:
        pass
    
    @abstractmethod
    def reset(self):
        pass

class SherpaKWSBackend(TriggerBackend):
    """
    Open-Vocabulary Keyword Spotting via Sherpa-ONNX.
    Uses a Zipformer model trained on GigaSpeech.
    """
    def __init__(self, model_dir: str = "src/assets/models/kws"):
        self.model_dir = model_dir
        self.spotter = None
        self.stream = None
        self.vad = SileroVAD()
        self._initialized = False
        
        # VAD State Management
        self.is_active = False
        self.silence_chunks = 0
        self.HANGOVER_CHUNKS = 15  # ~0.5 seconds of hangover (15 * 32ms)
        
    def _initialize(self):
        if self._initialized: return
        
        kw_path = f"{self.model_dir}/keywords.txt"
        with open(kw_path, 'w') as f:
            # Always update with latest sensitivity settings
            # Lowered threshold to 0.1 and increased boosting to 2.0
            f.write("▁HE Y ▁ST U P ID :2.0 #0.1\n")
                
        required = ["encoder-epoch-12-avg-2-chunk-16-left-64.onnx", 
                    "decoder-epoch-12-avg-2-chunk-16-left-64.onnx", 
                    "joiner-epoch-12-avg-2-chunk-16-left-64.onnx", 
                    "tokens.txt"]
        for r in required:
            if not os.path.exists(f"{self.model_dir}/{r}"):
                raise FileNotFoundError(f"Missing Sherpa model file: {r}")

        # WHY: Positional arguments are safer for this specific Python wrapper.
        self.spotter = sherpa_onnx.KeywordSpotter(
            f"{self.model_dir}/tokens.txt",
            f"{self.model_dir}/encoder-epoch-12-avg-2-chunk-16-left-64.onnx",
            f"{self.model_dir}/decoder-epoch-12-avg-2-chunk-16-left-64.onnx",
            f"{self.model_dir}/joiner-epoch-12-avg-2-chunk-16-left-64.onnx",
            kw_path,
            num_threads=2,
            sample_rate=16000,
            feature_dim=80,
            max_active_paths=8, # More paths = better recognition of noisy audio
            keywords_score=2.0, # Stronger boost
            keywords_threshold=0.1, # Global fallback threshold
            provider="cpu"
        )
        self.stream = self.spotter.create_stream()
        self._initialized = True
        logger.info("📡 [SherpaKWS] Engine activated.")

    def detect(self, audio_data: np.ndarray) -> str:
        try:
            self._initialize()
            
            # WHY: Continuous Audio Stream Integrity.
            # We cannot just drop silent chunks. Zipformer relies on a continuous 
            # acoustic context. If we drop chunks, we splice the audio and destroy 
            # the phoneme timings.
            # Instead, we use VAD to open a "gate" with a hangover.
            is_speech = self.vad.is_speech(audio_data, threshold=0.2)
            
            if is_speech:
                self.is_active = True
                self.silence_chunks = 0
            else:
                self.silence_chunks += 1
                if self.silence_chunks > self.HANGOVER_CHUNKS:
                    if self.is_active:
                        # Reset stream after prolonged silence to save CPU/VRAM
                        self.reset()
                        self.is_active = False
                    return ""
            
            if not self.is_active:
                return ""
            
            # Feed the continuous stream
            self.stream.accept_waveform(16000, audio_data)
            
            while self.spotter.is_ready(self.stream):
                self.spotter.decode_stream(self.stream)
                
            result = self.spotter.get_result(self.stream)
            if result:
                logger.info(f"✨ [SherpaKWS] TRIGGER: {result}")
                self.reset()
                self.is_active = False
                return "hey_stupid"
                
            return ""
        except Exception as e:
            logger.error(f"❌ [SherpaKWS] Detection error: {e}", exc_info=True)
            return ""

    def reset(self):
        if self.spotter:
            self.stream = self.spotter.create_stream()
