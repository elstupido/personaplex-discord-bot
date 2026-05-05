import os
import torch
import logging
from abc import ABC, abstractmethod
from core.logger import setup_logger
from .config import AlignmentConfig, Colors

logger = setup_logger("server.vocoder")

class VocoderBase(ABC):
    """
    The Abstract Mouthpiece (The Architectural Blueprint).
    
    WHY:
    In the physics of speech synthesis, the 'LLM' is the brain that thinks 
    in discrete semantic units (tokens). The 'Vocoder' is the physical 
    vocal tract that converts those abstract concepts into moving air (waveforms).
    
    By abstracting this into a base class, we ensure that our 'Synthesis Engine' 
    is not biologically dependent on any single vocal tract implementation. 
    This is the 'Inversion of Control' that allows us to support future 
    technologies like Fish Audio S2 or BigVGAN without heart surgery on the engine.
    """
    @abstractmethod
    def token2wav(self, tokens, **kwargs):
        """
        Transform discrete tokens into raw waveform tensors.
        
        WHY: 
        The primary responsibility of any vocoder is to solve the 'Acoustic 
        Inversion Problem'—how to reconstruct continuous time-domain signals 
        from discrete, compressed representations.
        """
        pass

    @abstractmethod
    def warmup(self):
        """
        Prime the model weights and CUDA kernels.
        
        WHY: 
        The first time a GPU executes a new kernel, there's a significant 
        'Cold-Start' overhead for JIT compilation and memory allocation. 
        In a real-time voice chat, a 500ms delay on the first word is the 
        difference between immersion and frustration. Warmup 'eager-loads' 
        these kernels into VRAM.
        """
        pass

class GLMVocoder(VocoderBase):
    """
    The Flow-Matching Larynx (The THUDM Specialization).
    
    WHY:
    This implementation uses 'Flow-Matching'—a concept from continuous-time 
    normalizing flows. Think of it as a way to learn a 'Vector Field' that 
    smoothly deforms a simple noise distribution into the complex, multi-modal 
    distribution of human speech.
    
    CONCEPT: THE TEMPORAL GRID
    Sound is a continuous wave, but neural models are discrete processors. 
    A 'Temporal Grid' is the quantization of time into fixed intervals (frames). 
    The 'Rate' of this grid is a fundamental trade-off:
    
    1. RESOLUTION: A higher frequency rate captures finer acoustic details—the 
       shimmer of a 's' sound or the micro-fluctuations in pitch.
    2. LATENCY: Every frame requires GPU compute. A denser grid increases 
       the workload, potentially breaking real-time 'Streaming' performance.
    
    WHY ALIGNMENT IS SACRED:
    The Brain (LLM) and the Larynx (Vocoder) must share the same temporal 
    map. If the Brain generates tokens at one rate but the Larynx reconstructs 
    them at another, the resulting audio will suffer from 'Acoustic Drift'—
    appearing either unnaturally sped up or slowed down. We enforce this 
    grid not because the number is magical, but because it is the 
    shared language of our acoustic pipeline.
    """
    def __init__(self, flow_path, device='cuda'):
        self.device = device
        self.flow_path = flow_path
        self._decoder = None
        self._init_decoder()

    def _init_decoder(self):
        """
        The Initialization Ritual.
        
        WHY THE CONFIGS?
        The decoder requires three distinct artifacts:
        1. config.yaml: The 'Architectural Schematic' of the layers.
        2. flow.pt: The 'Learned Flows' (The Flow-Matching weights).
        3. hift.pt: The 'HiFT' weights (The High-Fidelity Transform that 
           turns Mel-spectrograms into actual samples).
        """
        logger.info(f"[Vocoder] Initializing GLM Flow Decoder: {self.flow_path}")
        from flow_inference import AudioDecoder
        
        cfg_path = os.path.join(self.flow_path, "config.yaml")
        flow_ckpt = os.path.join(self.flow_path, "flow.pt")
        hift_ckpt = os.path.join(self.flow_path, "hift.pt")
        
        # WHY AUTOMATIC DISCOVERY?
        # Different environments (WSL, Docker, Native Linux) might name 
        # checkpoint files differently (e.g. .pth vs .pt). This logic 
        # makes the bot 'Agnostic' to minor filesystem variations.
        for f in ["flow.pt", "flow.pth", "mp_rank_00_model_states.pt"]:
            p = os.path.join(self.flow_path, f)
            if os.path.exists(p): 
                flow_ckpt = p
                break
        
        self._decoder = AudioDecoder(cfg_path, flow_ckpt, hift_ckpt, self.device)
        logger.info("[Vocoder] GLM Decoder loaded and ready.")

    def token2wav(self, tokens, **kwargs):
        """
        The Generative Act.
        
        WHY THE CONDITIONING?
        A 'Zero-Shot' vocoder cannot just read tokens; it needs an 
        'Identity Anchor'. 
        - prompt_token/feat: The 'Acoustic Memory' of how the speaker 
          SOUNDS in a reference clip.
        - embedding: The 'Identity Vector' that ensures the voice 
          doesn't drift into a different person mid-sentence.
        """
        with torch.inference_mode():
            # We use torch.inference_mode() because it's even faster than 
            # no_grad()—it completely bypasses the autograd graph construction.
            tts_speech, tts_mel = self._decoder.token2wav(
                tokens, 
                uuid=kwargs.get('uuid', 'default'),
                prompt_token=kwargs.get('prompt_token'),
                prompt_feat=kwargs.get('prompt_feat'),
                embedding=kwargs.get('embedding'),
                finalize=kwargs.get('finalize', True)
            )
            return tts_speech, tts_mel

    def warmup(self):
        """
        Force-fire the CUDA kernels.
        
        WHY THE DUMMY DATA?
        We use zeros because the value doesn't matter for compilation. 
        We just need the GPU to trace the execution path once so it can 
        optimize the parallel compute kernels for the real data.
        """
        logger.info("[Vocoder] Warming up kernels...")
        dummy_tokens = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        dummy_prompt = torch.zeros((1, 0), dtype=torch.long, device=self.device)
        dummy_feat = torch.zeros((1, 0, 80), device=self.device)
        dummy_emb = torch.zeros((1, 192), device=self.device)
        
        self.token2wav(
            dummy_tokens, 
            prompt_token=dummy_prompt, 
            prompt_feat=dummy_feat, 
            embedding=dummy_emb
        )
        logger.info("[Vocoder] Kernels hot.")

class FishVocoder(VocoderBase):
    """
    The Vector-Quantized Larynx (The Fish Audio S2 Implementation).
    
    WHY:
    Fish Audio S2 utilizes the Descript Audio Codec (DAC) as its 
    reconstruction engine. Unlike GLM's Flow-Matching, which deforms 
    continuous noise into speech, DAC works by decoding 'Indices' from a 
    discrete Vector-Quantized (VQ) codebook.
    
    CONCEPT: THE ACOUSTIC ATOM
    By mapping semantic intent to a finite set of learned 'Atoms' (VQ codes), 
    the model gains extreme robustness. It doesn't have to 'Imagine' the 
    physics of air vibration from scratch; it just has to select the right 
    sequence of pre-calculated high-fidelity building blocks.
    
    TRADE-OFF: LATENCY VS VARIATION
    While VQ-based models often provide 'Crisper' output, they are 
    computationally intensive during the 'Auto-Regressive' sampling phase. 
    We integrate this as a premium backend for users who prioritize 
    'Acoustic Texture' over raw throughput.
    """
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = device
        self.checkpoint_path = checkpoint_path
        self._model = None
        self._init_model()

    def _init_model(self):
        """
        The Hydra Initialization.
        
        WHY HYDRA?
        Fish Audio uses the Hydra configuration framework to manage its 
        complex graph of neural components. We must initialize a global 
        Hydra instance to compose the schema before instantiating the 
        model weights.
        """
        logger.info(f"[Vocoder] Loading Fish Audio S2 (DAC): {self.checkpoint_path}")
        try:
            # We must reach into the fish_speech namespace to leverage 
            # their specialized DAC inference logic.
            from fish_speech.models.dac.inference import load_model
            
            # Note: We assume the 'modded_dac_vq' config name as per 
            # the official S2-Pro documentation.
            self._model = load_model(
                config_name="modded_dac_vq",
                checkpoint_path=self.checkpoint_path,
                device=self.device
            )
            logger.info("[Vocoder] Fish Larynx operational.")
        except ImportError:
            logger.error("│ 💥 Fish-Speech not found. Run: pip install -e /root/fish-speech")
            raise
        except Exception as e:
            logger.error(f"│ 🛑 Fish Larynx Failure: {e}")
            raise

    def token2wav(self, tokens, **kwargs):
        """
        Reconstruct audio from Fish-specific VQ indices.
        
        WHY INDICES?
        In the Fish ecosystem, 'Tokens' are not just semantic identifiers; 
        they are indices into a high-dimensional codebook. 
        """
        with torch.inference_mode():
            # If we are passed indices (2D tensor [K, T]), we unsqueeze 
            # for the batch dimension [1, K, T].
            indices = tokens
            if indices.ndim == 2:
                indices = indices.unsqueeze(0)
            
            # The 'from_indices' method is the heart of the VQ-Decoder. 
            # It performs the codebook lookup and runs the GAN-based generator.
            fake_audios = self._model.from_indices(indices)
            return fake_audios[0], None # Fish DAC doesn't return Mel by default

    def warmup(self):
        """Prime the VQ-GAN kernels."""
        logger.info("[Vocoder] Warming up Fish kernels...")
        # A typical DAC sequence length for a 1s clip.
        dummy_indices = torch.zeros((1, 4, 50), dtype=torch.long, device=self.device)
        self.token2wav(dummy_indices)
        logger.info("[Vocoder] Fish kernels hot.")

class VocoderManager:
    """
    The Orchestrator of Sound (The Central Dispatch).
    
    WHY:
    The Engine shouldn't care about WHICH vocoder is running. It should 
    just ask the 'Manager' to reconstruct a voice. This manager handles 
    the logic of choosing between the standard GLM vocoder or future 
    premium options.
    """
    def __init__(self, backend_type='glm', device='cuda', **kwargs):
        self.device = device
        self.kwargs = kwargs
        self.vocoders = {}
        self.active_backend = backend_type
        self.alignment_config = AlignmentConfig()
        
        # WHY: We don't initialize the backend here anymore. 
        # We wait until someone actually needs to speak.
        logger.info(f"[VocoderManager] Configured with default backend: {backend_type} (Lazy)")

    def get_active_vocoder(self):
        """
        Retrieves the active vocoder, initializing it if necessary.
        
        WHY: 
        This is the 'Just-In-Time' entry point. It ensures that VRAM is 
        only consumed at the moment of first synthesis.
        """
        if self.active_backend not in self.vocoders:
            logger.info(f"[VocoderManager] Lazy-loading backend: {self.active_backend}")
            self.vocoders[self.active_backend] = self._create_vocoder(self.active_backend, **self.kwargs)
        return self.vocoders[self.active_backend]

    def _create_vocoder(self, backend_type, **kwargs):
        """
        The Factory Method.
        
        WHY:
        We gate model instantiation behind the 'enabled_vocoder_backends' 
        list. This allows the user to 'De-Scope' certain models at build 
        time, ensuring that experimental or heavy backends (like Fish) 
        cannot be accidentally loaded into VRAM.
        """
        if backend_type not in self.alignment_config.enabled_vocoder_backends:
            logger.error(f"│ 🛑 VOCODER BACKEND '{backend_type.upper()}' IS DISABLED IN CONFIG.")
            raise ValueError(f"Backend {backend_type} is not enabled for this build.")

        if backend_type == 'glm':
            return GLMVocoder(kwargs.get('flow_path'), kwargs.get('device', 'cuda'))
        elif backend_type == 'fish':
            # Sarcastic Note: Upgrading to the premium Fish Larynx. 
            # Hope your VRAM is ready for the deep end.
            return FishVocoder(kwargs.get('codec_path'), kwargs.get('device', 'cuda'))
        else:
            raise ValueError(f"Unknown vocoder backend: {backend_type}")

    def reconstruct(self, tokens, **kwargs):
        """
        The synthesis hot-path.
        """
        vocoder = self.get_active_vocoder()
        return vocoder.token2wav(tokens, **kwargs)

    def warmup(self):
        """
        The explicit kernel priming trigger.
        """
        logger.info(f"[VocoderManager] Warming up active backend: {self.active_backend}")
        vocoder = self.get_active_vocoder()
        vocoder.warmup()
