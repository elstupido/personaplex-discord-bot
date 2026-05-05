import os
import torch
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F
import torch.nn.functional as F_nn
import tempfile
import base64
import json
import numpy as np
import time
import sys
import io
from threading import Thread, Lock

from .config import GLM_MODEL_SAMPLE_RATE, Colors, AlignmentConfig
from .cloning import IdentityManager
from .vocoder import VocoderManager

from core.logger import setup_logger
logger = setup_logger("server.engine")

# WHY: Muzzle noisy library-internal logs that pollute the real-time stream.
import logging
logging.getLogger("faster_whisper").setLevel(logging.WARNING)
logging.getLogger("funasr").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.ERROR)

"""
WHY THIS FILE EXISTS:
The Central Nervous System of Synthesis.

WHY THIS CLASS EXISTS:
GLMVoiceEngine is the 'Orchestrator'. In the complex biological act of 
speaking, the brain doesn't just 'make sound'—it coordinates semantic 
intent, acoustic memory, and muscular execution.
"""
class GLMVoiceEngine:
    def __init__(self, model_path, tokenizer_path, flow_path, device='cuda'):
        self.device = device
        self.lock = Lock()
        self.gen_lock = Lock()
        self.alignment_config = AlignmentConfig()
        
        self._init_decoder(flow_path)
        
        # State: Model containers
        self._glm_model = None
        self._glm_tokenizer = None
        self._whisper_model = None
        self._sensevoice_model = None  # Fish Audio ASR
        self._feature_extractor = None
        self._speaker_encoder = None
        self._model_path = model_path
        self._tokenizer_path = tokenizer_path

        self._init_ids()
        self.identity_manager = IdentityManager()
        self.active_identity = None 
        self._update_mel_basis()
        self.warmup()

    def _check_vram(self, threshold=0.90):
        """
        WHY: The Engine needs to be its own VRAM steward when loading 
        heavyweight weights (SenseVoice, GLM-4-Voice).
        """
        if not torch.cuda.is_available(): return
        reserved = torch.cuda.memory_reserved(0)
        total = torch.cuda.get_device_properties(0).total_memory
        if reserved / total > threshold:
            logger.warning(f"[Engine] VRAM Pressure Detected ({reserved/total*100:.1f}%). Cleaning cache...")
            torch.cuda.empty_cache()

    @property
    def glm_model(self):
        if self._glm_model is None:
            self._check_vram()
            logger.info(f"{Colors.CYAN}[Engine] Lazy-Loading GLM: {self._model_path}{Colors.RESET}")
            from transformers import AutoModel
            self._glm_model = AutoModel.from_pretrained(self._model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(self.device).eval()
        return self._glm_model

    @property
    def glm_tokenizer(self):
        if self._glm_tokenizer is None:
            from transformers import AutoTokenizer
            self._glm_tokenizer = AutoTokenizer.from_pretrained(self._model_path, trust_remote_code=True)
        return self._glm_tokenizer

    @property
    def whisper_model(self):
        if self._whisper_model is None:
            self._check_vram()
            logger.info(f"{Colors.CYAN}[Engine] Lazy-Loading Whisper: {self._tokenizer_path}{Colors.RESET}")
            from speech_tokenizer.modeling_whisper import WhisperVQEncoder
            self._whisper_model = WhisperVQEncoder.from_pretrained(self._tokenizer_path).to(self.device).eval()
        return self._whisper_model

    @property
    def feature_extractor(self):
        if self._feature_extractor is None:
            from speech_tokenizer.utils import WhisperFeatureExtractor
            self._feature_extractor = WhisperFeatureExtractor.from_pretrained(self._tokenizer_path)
        return self._feature_extractor

    @property
    def sensevoice_model(self):
        """The 'Big Fish' ASR Model (SenseVoiceSmall)."""
        if self._sensevoice_model is None:
            self._check_vram()
            logger.info(f"{Colors.CYAN}[Engine] Lazy-Loading SenseVoiceSmall...{Colors.RESET}")
            try:
                from funasr import AutoModel
                self._sensevoice_model = AutoModel(model="iic/SenseVoiceSmall", device=self.device, disable_update=True)
                logger.info("[Engine] SenseVoice operational.")
            except ImportError:
                logger.error("│ ⚠️ funasr not installed. SenseVoice fallback to Whisper.")
                self._sensevoice_model = "FALLBACK"
        return self._sensevoice_model

    @property
    def speaker_encoder(self):
        if self._speaker_encoder is None:
            logger.info(f"{Colors.CYAN}[Engine] Lazy-Loading Speaker Encoder (CAM++){Colors.RESET}")
            from .cloning import SpeakerEncoder
            model_id = 'iic/speech_eres2net_sv_zh-cn_16k-common'
            self._speaker_encoder = SpeakerEncoder(model_id, self.device)
            self._speaker_encoder.alignment_config = self.alignment_config
        return self._speaker_encoder

    def _init_decoder(self, flow_path):
        self.vocoder_manager = VocoderManager(
            backend_type='glm', 
            flow_path=flow_path, 
            codec_path=self.alignment_config.fish_codec_path,
            device=self.device
        )

    def _init_ids(self):
        self.system_id = 151335
        self.user_id = 151336
        self.assistant_id = 151337
        self.begin_audio_id = 151343
        self.end_audio_id = 151344

    def _update_mel_basis(self):
        cfg = self.alignment_config
        self.mel_basis = torchaudio.functional.melscale_fbanks(
            n_freqs=cfg.n_fft // 2 + 1,
            f_min=cfg.f_min, f_max=cfg.f_max,
            n_mels=cfg.n_mels, sample_rate=cfg.sample_rate,
            norm="slaney", mel_scale="slaney"
        ).to(self.device).T

    def _compute_matcha_mel(self, audio: torch.Tensor, sr: int) -> torch.Tensor:
        cfg = self.alignment_config
        if sr != cfg.sample_rate:
            audio = torchaudio.functional.resample(audio, sr, cfg.sample_rate)
        
        spec = torch.stft(
            audio.squeeze(0), 
            n_fft=cfg.n_fft, 
            hop_length=cfg.hop_length, 
            win_length=cfg.win_length,
            window=torch.hann_window(cfg.win_length).to(self.device),
            center=True, return_complex=True
        ).abs()
        
        mel = torch.matmul(self.mel_basis, spec)
        log_mel = torch.log(torch.clamp(mel, min=cfg.mel_min_clamp))
        if cfg.mel_scale == "ln":
            log_mel = (log_mel - cfg.mel_mean) / cfg.mel_std
        return log_mel.unsqueeze(0)

    # --- Identity Delegation Properties ---
    @property
    def ref_prompt_token(self): return self.active_identity.get("prompt_token") if self.active_identity else None
    @property
    def ref_prompt_feat(self): return self.active_identity.get("prompt_feat") if self.active_identity else None
    @property
    def ref_prompt_embedding(self): return self.active_identity.get("embedding") if self.active_identity else None
    @property
    def ref_stats(self): return self.active_identity.get("stats") if self.active_identity else {}

    def set_voice_reference(self, pcm_b64: str, name: str, sample_rate: int = 48000, prompt_text: str = None):
        self.active_identity = self.identity_manager.crystallize_identity(self, pcm_b64, name, sample_rate, prompt_text)
        return {"status": "ok", "name": name}

    def load_voice_profile(self, name):
        profile = self.identity_manager.load_profile(name)
        if profile:
            self.active_identity = profile
            return {"status": "ok", "name": name}
        return {"status": "error", "message": "Identity not found in vault."}

    def _tokenize_b64(self, pcm_b64: str) -> list:
        from speech_tokenizer.utils import extract_speech_token
        pcm_bytes = base64.b64decode(pcm_b64)
        audio, sr = torchaudio.load(io.BytesIO(pcm_bytes))
        if audio.shape[0] > 1: audio = audio.mean(dim=0, keepdim=True)
        audio_16k = torchaudio.functional.resample(audio.to(self.device), sr, 16000)
        tokens = extract_speech_token(self.whisper_model, self.feature_extractor, [(audio_16k, 16000)])[0]
        return tokens.tolist()

    def transcribe(self, pcm_b64: str) -> str:
        """The Remote ASR Gateway using SenseVoiceSmall."""
        pcm_bytes = base64.b64decode(pcm_b64)
        # WHY np.frombuffer? 
        # The TriggerEngine sends raw int16 bytes to avoid the overhead of 
        # wrapping every 0.5s window in a WAV header. We decode directly 
        # into a NumPy array for maximum performance.
        pcm_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        audio_16k = pcm_int16.astype(np.float32) / 32767.0
        
        model = self.sensevoice_model
        if model == "FALLBACK":
            return "[SenseVoice Unavailable] Install funasr"
        
        try:
            # SHINY: We pass the 1D array and let funasr handle the batching.
            # We use a comprehension-style extraction to keep it tight.
            res = model.generate(input=audio_16k, cache={}, language="auto")
            
            return (
                res[0].get('text', '') if isinstance(res[0], dict) 
                else str(res[0]) if res 
                else ""
            )
        except Exception as e:
            logger.error(f"│ ⚠️ SenseVoice Failure: {e}")
            return ""

    def prepare_input(self, messages):
        input_ids = []
        idnt = self.active_identity
        if idnt is not None:
            input_ids.append(self.system_id)
            msg = f"TRANSCRIPTION: {idnt['prompt_text']}\nACT AS this speaker." if idnt.get("prompt_text") else "ACT AS the speaker in the following audio:"
            input_ids.extend(self.glm_tokenizer.encode(msg, add_special_tokens=False))
            input_ids.append(self.begin_audio_id)
            input_ids.extend([t + self.alignment_config.audio_offset for t in idnt["prompt_token"][0].tolist()])
            input_ids.append(self.end_audio_id)
        
        input_ids = self._add_history(input_ids, messages)
        input_ids.append(self.assistant_id)
        input_ids = [min(tid, self.alignment_config.true_vocab_size - 1) for tid in input_ids]
        return {"input_ids": torch.tensor([input_ids], device=self.device)}

    def _add_history(self, ids, messages):
        role_map = {"system": self.system_id, "user": self.user_id, "assistant": self.assistant_id}
        for msg in messages:
            ids.append(role_map.get(msg.get("role", "user"), self.user_id))
            if msg.get("content"): ids.extend(self.glm_tokenizer.encode(msg["content"], add_special_tokens=False))
            audio_tokens = msg.get("audio_tokens", [])
            if not audio_tokens and msg.get("audio_b64"): audio_tokens = self._tokenize_b64(msg["audio_b64"])
            if audio_tokens:
                ids.append(self.begin_audio_id)
                ids.extend([t + self.alignment_config.audio_offset for t in audio_tokens])
                ids.append(self.end_audio_id)
        return ids

    def process_segment(self, segment):
        try:
            from speech_tokenizer.utils import extract_speech_token
            pcm_bytes = base64.b64decode(segment.get("audio"))
            audio, sr = torchaudio.load(io.BytesIO(pcm_bytes))
            if audio.shape[0] > 1: audio = audio.mean(dim=0, keepdim=True)
            audio_16k = torchaudio.functional.resample(audio.to(self.device), sr, 16000)
            tokens = extract_speech_token(self.whisper_model, self.feature_extractor, [(audio_16k, 16000)])[0]
            return [t + self.alignment_config.audio_offset for t in tokens]
        except Exception as e:
            logger.error(f"│ ⚠️ SEGMENT TOKENIZATION FAIL: {e}")
            return []

    def warmup(self, force=False):
        # WHY: We support Eager Warmup to ensure the bot is 'Ready to Talk' before the first user command.
        if not force and self.alignment_config.lazy_load_models: 
            logger.info("[Engine] Eager Warmup skipped (Lazy Mode enabled).")
            return
            
        logger.info("[Engine] 🛰️  Priming Neural Kernels (Eager Mode)...")
        try:
            _ = self.glm_model
            _ = self.whisper_model
            _ = self.speaker_encoder
            self.vocoder_manager.warmup()
            logger.info("[Engine] Neural chain is hot.")
        except Exception as e: 
            logger.error(f"Engine Warmup failed: {e}")
