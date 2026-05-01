import sys
import os
import torch
import torchaudio.functional as F_audio
import numpy as np
from utils.logger import setup_logger

logger = setup_logger("bridge.components")

# Apply paths from glm_server config so we can find GLM dependencies
# (Assumes this code runs in the same environment as the GLM Server)
GLM_PATHS = [
    "/app/glm",
    "/app/glm/cosyvoice",
    "/app/glm/third_party/Matcha-TTS"
]
for p in GLM_PATHS:
    if p not in sys.path:
        sys.path.append(p)

from bridge.glm_server.config import Colors

class AudioResampler:
    """GPU-accelerated audio resampler."""
    
    def __init__(self, device="cuda"):
        self.device = device

    def downsample(self, pcm_data: bytes) -> np.ndarray:
        """48kHz stereo int16 -> 16kHz mono float32"""
        arr = np.frombuffer(pcm_data, dtype=np.int16)
        samples_stereo = torch.tensor(arr, device=self.device).float() / 32768.0
        
        # Extract left channel (L, R, L, R...)
        left_channel = samples_stereo[0::2]
        
        # Resample on GPU
        tensor_48k = left_channel.unsqueeze(0)
        tensor_16k = F_audio.resample(tensor_48k, 48000, 16000).squeeze(0)
        return tensor_16k.cpu().numpy()

    def upsample(self, generated_audio_tensor: torch.Tensor, orig_sr: int = 22050, target_sr: int = 48000) -> bytes:
        """Model output rate mono -> 48kHz stereo int16 Discord format"""
        # Ensure it's on the correct device and shaped right
        mono = generated_audio_tensor.to(self.device).unsqueeze(0) if generated_audio_tensor.dim() == 1 else generated_audio_tensor.to(self.device)
        
        resampled = F_audio.resample(mono, orig_sr, target_sr)
        
        # Make stereo by duplicating channel
        stereo = torch.cat([resampled, resampled], dim=0)
        
        # Clamp, scale, convert to bytes
        # Must move back to CPU for raw bytes export
        stereo_int16 = (stereo.clamp(-1, 1) * 32767).to(torch.int16).cpu()
        return stereo_int16.T.contiguous().numpy().tobytes()

    def upsample_bytes(self, pcm_data: bytes, orig_sr: int = 22050, target_sr: int = 48000) -> bytes:
        """Helper to upsample raw 22kHz mono int16 bytes to 48kHz stereo bytes."""
        arr = np.frombuffer(pcm_data, dtype=np.int16)
        tensor = torch.tensor(arr, device=self.device).float() / 32768.0
        return self.upsample(tensor, orig_sr, target_sr)


class WhisperTokenizer:
    """GPU-bound Tokenization & Transcription."""
    
    def __init__(self, tokenizer_path="THUDM/glm-4-voice-tokenizer", device="cuda"):
        self.device = device
        self.tokenizer_path = tokenizer_path
        self._whisper_model = None
        self._feature_extractor = None
        
        # Standard transcriber fallback (reusing bridge's existing transcriber logic)
        from bridge.glm import _transcriber
        self._transcriber = _transcriber

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
        logger.info(f"[WhisperTokenizer] Loading VQ Encoder from {self.tokenizer_path}...")
        
        try:
            from speech_tokenizer.modeling_whisper import WhisperVQEncoder
            from transformers import WhisperFeatureExtractor
            
            self._whisper_model = WhisperVQEncoder.from_pretrained(self.tokenizer_path).to(self.device).eval()
            self._feature_extractor = WhisperFeatureExtractor.from_pretrained(self.tokenizer_path)
            logger.info("[WhisperTokenizer] Loaded VQ Encoder.")
        except Exception as e:
            logger.error(f"[WhisperTokenizer] Failed to load VQ Encoder: {e}")
            raise e

    def extract_features(self, audio_16k: np.ndarray) -> list:
        self._ensure_loaded()
        # Input expected: list of tuples (tensor, sample_rate)
        # Tensor must be 2D [channels, samples]
        tensor_16k = torch.from_numpy(audio_16k).unsqueeze(0).to(self.device)
        return [(tensor_16k, 16000)]

    def get_vq_tokens(self, features: list) -> list:
        self._ensure_loaded()
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
            # Re-enable the real transcribe logic if it was disabled in the bridge
            # or just call the model directly
            if self._transcriber._model is None:
                self._transcriber._ensure_loaded()
            
            return self._transcriber._run_model(audio_16k)
        except Exception as e:
            return f"[Transcription failed: {e}]"


class GLMAudioDecoder:
    """GPU-bound Vocoder (Flow + HiFT)."""
    
    def __init__(self, flow_path="/app/glm/glm-4-voice-decoder", device="cuda"):
        self.device = device
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

    def vocode_to_waveform(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        self._ensure_loaded()
        import uuid
        session_uuid = str(uuid.uuid4())
        try:
            with torch.inference_mode():
                tts_speech, _ = self._decoder.token2wav(mel_spectrogram, uuid=session_uuid, finalize=True)
            return tts_speech
        except Exception as e:
            print(f"{Colors.RED}[GLMAudioDecoder] Decoding Error: {e}{Colors.RESET}")
            return torch.zeros(1) # fallback silence
