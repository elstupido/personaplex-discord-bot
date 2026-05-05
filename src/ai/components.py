"""
WHY THIS FILE EXISTS:
The foundational building blocks of the AI pipeline. This file contains 
heavyweight components like AudioResamplers, Tokenizers, and Decoders 
that perform the actual DSP and neural inference operations.

WHY SEPARATE FROM THE BRIDGES?
By decoupling the 'Logic' (Bridge) from the 'Machinery' (Components), 
we can unit-test the resampler or the tokenizer without needing to 
spin up a full Discord client or a WebSocket server.
"""
import sys
import os
import torch
import torchaudio.functional as F_audio
import numpy as np
import time
from core.logger import setup_logger

logger = setup_logger("ai.components")

# WHY: Muzzle noisy library-internal logs.
import logging
logging.getLogger("faster_whisper").setLevel(logging.WARNING)
logging.getLogger("funasr").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.ERROR)

# Apply paths from local environment so we can find GLM dependencies
# WHY: We support both native WSL and Docker. We search for 'src/ai/providers/glm'
# in the current workspace if /app/glm is missing.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GLM_PATHS = [
    os.path.join(BASE_DIR, "src/ai/providers/glm"),
    "/app/glm",
    "/app/glm/cosyvoice",
    "/app/glm/third_party/Matcha-TTS"
]
for p in GLM_PATHS:
    if os.path.exists(p) and p not in sys.path:
        sys.path.append(p)

from server.glm_server.config import Colors

class AudioResampler:
    """
    The Harmonic Frequency Translator (The Temporal Bridge).
    
    WHY THIS EXISTS:
    Sound exists in multiple digital 'Realities'. Discord's reality is 48kHz—
    optimized for the limits of human hearing. Our AI's reality is often 
    much lower (e.g., 22.05kHz)—optimized for the limits of VRAM and 
    computational throughput.
    
    CONCEPT: FREQUENCY DOMAINS
    Resampling is the act of 'Translating' a signal from one temporal domain 
    to another. If we don't resample, the audio will appear stretched or 
    compressed (aliasing). By using GPU-accelerated linear interpolation, 
    we ensure this translation happens with minimal phase distortion and 
    maximum speed.
    """
    
    def __init__(self, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        logger.debug(f"✨ [AudioResampler] Initialized on {self.device}")

    def warmup(self):
        """Warm up the GPU kernels for resampling."""
        # WHY: The first time torchaudio.functional.resample is called, 
        # it might initialize CUDA kernels which can cause a small stall.
        dummy = torch.zeros(16000, device=self.device)
        self.downsample(dummy.cpu().numpy().tobytes())
        self.upsample(dummy.cpu().numpy().tobytes(), 16000, 48000)
        logger.info("[AudioResampler] GPU Kernels Warmed.")

    def downsample(self, pcm_data: bytes) -> np.ndarray:
        """
        Compressing Reality (48kHz Stereo -> 16kHz Mono).
        
        WHY: To feed the AI's intake layer, we must discard high-frequency 
        data and collapse stereo into mono.
        """
        # 1. Convert int16 bytes to float32 tensor
        arr = np.frombuffer(pcm_data, dtype=np.int16)
        samples = torch.tensor(arr, device=self.device).float() / 32768.0
        
        # 2. collapse to mono if stereo
        if len(samples) % 2 == 0:
            left_channel = samples[0::2]
            tensor_48k = left_channel.unsqueeze(0)
        else:
            tensor_48k = samples.unsqueeze(0)
            
        # 3. Resample on GPU
        tensor_16k = F_audio.resample(tensor_48k, 48000, 16000).squeeze(0)
        return tensor_16k.cpu().numpy()

    def upsample(self, pcm_data: bytes, orig_sr: int = 22050, target_sr: int = 48000) -> bytes:
        """
        Expanding Reality (Variable Mono -> 48kHz Stereo).
        
        WHY: To make the AI audible to humans on Discord, we must project its 
        low-resolution mono output back into the high-resolution 48kHz 
        stereo space.
        """
        # 1. Convert int16 bytes to float32 tensor
        arr = np.frombuffer(pcm_data, dtype=np.int16)
        mono = torch.tensor(arr, device=self.device).float() / 32768.0
        mono = mono.unsqueeze(0) # (1, T)
        
        # 2. Resample
        resampled = F_audio.resample(mono, orig_sr, target_sr)
        
        # 3. Make stereo by duplicating channel
        stereo = torch.cat([resampled, resampled], dim=0)
        
        # 4. Clamp, scale, convert to int16 bytes for Discord
        stereo_int16 = (stereo.clamp(-1, 1) * 32767).to(torch.int16).cpu()
        return stereo_int16.T.contiguous().numpy().tobytes()

class WhisperTokenizer:
    """
    The Semantic Phoneme Decoder (The Information Compressor).
    
    WHY THIS EXISTS:
    Raw audio is too 'dense' for an LLM to process efficiently. A 1-second 
    clip contains thousands of samples. A 'Tokenizer' converts this 
    continuous wave into a sequence of 'Tokens'—discrete symbols that represent 
    the semantic and acoustic essence of the sound.
    
    CONCEPT: LATENT SPACE COMPRESSION
    We use a VQ (Vector Quantized) Encoder to project audio into a 'Latent 
    Space' where sounds with similar meanings (phonemes) are grouped together. 
    This allows the LLM to 'Read' audio just as it reads text.
    """
    
    def __init__(self, tokenizer_path="THUDM/glm-4-voice-tokenizer", device="cuda"):
        # WHY: We keep __init__ strictly 'Lightweight'. 
        # No weights are loaded here. This ensures that expert instantiation
        # can happen on the main thread without triggering a Heartbeat Failure.
        self.device = device if torch.cuda.is_available() else "cpu"
        self.tokenizer_path = tokenizer_path
        self._whisper_model = None
        self._feature_extractor = None
        self._transcriber = None

    def warmup(self):
        """Force load models and perform dummy inference."""
        self._ensure_loaded()
        dummy_audio = np.zeros(16000, dtype=np.float32)
        logger.info("[WhisperTokenizer] Warming up...")
        features = self.extract_features(dummy_audio)
        self.get_vq_tokens(features)
        self.transcribe(dummy_audio)
        logger.info("[WhisperTokenizer] Warm.")

    def _ensure_loaded(self):
        if self._whisper_model is not None: return
        
        load_start = time.time()
        logger.info(f"[WhisperTokenizer] Loading VQ Encoder from {self.tokenizer_path}...")
        
        try:
            # 1. Load VQ Encoder
            vq_start = time.time()
            try:
                from speech_tokenizer.modeling_whisper import WhisperVQEncoder
                self._whisper_model = WhisperVQEncoder.from_pretrained(self.tokenizer_path).to(self.device)
                self._whisper_model.eval()
                logger.info(f"✅ [WhisperTokenizer] Neural VQ Encoder loaded from {self.tokenizer_path}")
            except ImportError:
                logger.warning("⚠️ [WhisperTokenizer] speech_tokenizer module not found! Entering SIMULATION mode.")
                self._whisper_model = "MOCK_VQ"
            
            vq_duration = (time.time() - vq_start) * 1000
            logger.debug(f"[METRIC] op=load_vq expert=whisper_tokenizer duration_ms={vq_duration:.2f}")
            
            # 2. Load Feature Extractor
            fe_start = time.time()
            try:
                from transformers import WhisperFeatureExtractor
                self._feature_extractor = WhisperFeatureExtractor.from_pretrained(self.tokenizer_path)
            except ImportError:
                logger.warning("⚠️ [WhisperTokenizer] transformers module not found! Mocking feature extractor.")
                self._feature_extractor = "MOCK_FE"
            
            fe_duration = (time.time() - fe_start) * 1000
            logger.debug(f"[METRIC] op=load_fe expert=whisper_tokenizer duration_ms={fe_duration:.2f}")
            
            # 3. Load Transcriber (Lazy)
            if self._transcriber is None:
                try:
                    from .providers.glm.stubs import Transcriber
                    self._transcriber = Transcriber()
                except ImportError:
                    # Fallback if relative import fails in some test contexts
                    self._transcriber = type('Mock', (), {'transcribe': lambda x: "[MOCK]", 'warmup': lambda: None})()
                
            load_total = (time.time() - load_start) * 1000
            logger.info(f"[WhisperTokenizer] Warmup complete (Mode: {'SIMULATION' if self._whisper_model == 'MOCK_VQ' else 'NEURAL'}) in {load_total:.2f}ms.")
        except Exception as e:
            logger.error(f"[WhisperTokenizer] Critical failure during load: {e}")
            raise e

    def extract_features(self, audio_16k: np.ndarray) -> list:
        self._ensure_loaded()
        # Input expected: list of tuples (tensor, sample_rate)
        # Tensor must be 2D [channels, samples]
        tensor_16k = torch.from_numpy(audio_16k).unsqueeze(0).to(self.device)
        return [(tensor_16k, 16000)]

    def get_vq_tokens(self, features: list) -> list:
        self._ensure_loaded()
        if self._whisper_model == "MOCK_VQ":
            # Simulate 25Hz token rate (1 token per 40ms)
            num_samples = features[0][0].shape[1] if features else 16000
            num_tokens = int(np.ceil((num_samples / 16000.0) * 25.0))
            return [0] * num_tokens # Return a sequence of silence tokens

        try:
            from speech_tokenizer.utils import extract_speech_token
            with torch.no_grad():
                audio_tokens_list = extract_speech_token(
                    self._whisper_model, 
                    self._feature_extractor, 
                    features
                )
                if not audio_tokens_list: return []
                tokens = audio_tokens_list[0]
                
                # Truncate to match audio duration. 
                # Reverting to 25Hz to resolve doubling artifacts.
                num_samples = features[0][0].shape[1]
                expected_tokens = int(np.ceil((num_samples / 16000.0) * 25.0))
                
                return tokens[:expected_tokens]
        except Exception as e:
            logger.error(f"[WhisperTokenizer] Tokenization error: {e}")
            return []

    def transcribe(self, audio_16k: np.ndarray) -> str:
        # Use existing faster-whisper/openai-whisper model
        try:
            # WHY: Because why do it yourself when the server has a 5090?
            # We call the stub which informs the caller that analysis is offloaded.
            return self._transcriber.transcribe(audio_16k.tobytes())
        except Exception as e:
            return f"[Transcription failed: {e}]"


class GLMAudioDecoder:
    """GPU-bound Vocoder (Flow + HiFT)."""
    
    def __init__(self, flow_path="/app/glm/glm-4-voice-decoder", device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.flow_path = flow_path
        self._decoder = None
        # Provide the same monkey patch that engine.py uses
        self._apply_monkey_patch()

    def _apply_monkey_patch(self):
        try:
            import flow_inference
            def patched_fade_in_out(fade_in_mel, fade_out_mel, window):
                if not isinstance(window, torch.Tensor):
                    window = torch.from_numpy(window).to(fade_in_mel.device)
                else:
                    window = window.to(fade_in_mel.device)
                actual_len = min(fade_in_mel.shape[-1], fade_out_mel.shape[-1], window.shape[0])
                if actual_len > 0:
                    fade_in_mel[..., :actual_len] = fade_in_mel[..., :actual_len] * window[:actual_len] + \
                                                    fade_out_mel[..., :actual_len] * (1 - window[:actual_len])
                return fade_in_mel
            flow_inference.fade_in_out = patched_fade_in_out
        except Exception as e:
            logger.error(f"[GLMAudioDecoder] Could not apply fade patch: {e}")

    def warmup(self):
        """Force load models and perform dummy decoding."""
        self._ensure_loaded()
        logger.info("[GLMAudioDecoder] Warming up...")
        dummy_tokens = [0]
        tensor = self.prepare_tokens(dummy_tokens)
        mel = self.generate_flow_mel(tensor)
        self.vocode_to_waveform(mel)
        logger.info("[GLMAudioDecoder] Warm.")

    def _ensure_loaded(self):
        if self._decoder is not None: return
        logger.info(f"[GLMAudioDecoder] Loading from {self.flow_path}...")
        
        try:
            from flow_inference import AudioDecoder
            config_path = os.path.join(self.flow_path, "config.yaml")
            flow_ckpt = os.path.join(self.flow_path, "flow.pt")
            hift_ckpt = os.path.join(self.flow_path, "hift.pt")
            
            for f in ["flow.pt", "flow.pth", "mp_rank_00_model_states.pt"]:
                p = os.path.join(self.flow_path, f)
                if os.path.exists(p): flow_ckpt = p; break
            
            for f in ["hift.pt", "hift.pth"]:
                p = os.path.join(self.flow_path, f)
                if os.path.exists(p): hift_ckpt = p; break
                
            self._decoder = AudioDecoder(config_path, flow_ckpt, hift_ckpt, self.device)
            logger.info("[GLMAudioDecoder] Loaded.")
        except Exception as e:
            logger.error(f"[GLMAudioDecoder] Load failed: {e}")
            raise e

    def prepare_tokens(self, tokens: list) -> torch.Tensor:
        self._ensure_loaded()
        if not tokens:
            # Return a single silent/padding token tensor of type long
            return torch.tensor([[0]], device=self.device, dtype=torch.long)
        return torch.tensor(tokens, device=self.device, dtype=torch.long).unsqueeze(0)

    def generate_flow_mel(self, token_tensor: torch.Tensor):
        self._ensure_loaded()
        # Logically separate the flow step (tokens -> mel)
        # In actual flow_inference, token2wav combines these.
        # We will emulate this by doing the flow step if exposed, or passing through.
        # Assuming flow_inference provides a way to get mel:
        # For now, we will return the token tensor and handle it in vocode_to_waveform.
        return token_tensor

    def vocode_to_waveform(self, 
                           token_tensor: torch.Tensor, 
                           prompt_token: torch.Tensor = None, 
                           prompt_feat: torch.Tensor = None, 
                           embedding: torch.Tensor = None) -> torch.Tensor:
        self._ensure_loaded()
        import uuid
        session_uuid = str(uuid.uuid4())
        
        # Default empty conditioning if None
        if prompt_token is None: prompt_token = torch.zeros(1, 0, dtype=torch.long, device=self.device)
        if prompt_feat is None: prompt_feat = torch.zeros(1, 0, 80, device=self.device)
        if embedding is None: embedding = torch.zeros(1, 192, device=self.device)

        try:
            with torch.inference_mode():
                # Note: token2wav expects [1, N] tokens and returns waveform
                tts_speech, _ = self._decoder.token2wav(
                    token_tensor, 
                    uuid=session_uuid, 
                    prompt_token=prompt_token,
                    prompt_feat=prompt_feat,
                    embedding=embedding,
                    finalize=True
                )
            return tts_speech
        except Exception as e:
            print(f"{Colors.RED}[GLMAudioDecoder] Decoding Error: {e}{Colors.RESET}")
            return torch.zeros(1) # fallback silence
