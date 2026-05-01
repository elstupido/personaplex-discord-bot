import os
import torch
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F_audio
import torch.nn.functional as F
import tempfile
import base64
import json
import numpy as np
import time
import sys
from threading import Thread, Lock
from transformers import AutoModel, AutoTokenizer, WhisperFeatureExtractor
from .config import GLM_MODEL_SAMPLE_RATE

from utils.logger import setup_logger

logger = setup_logger("bridge.glm_server.engine")

# Lazy imports for GLM specific modules (they are added to sys.path in config.py)
from speech_tokenizer.modeling_whisper import WhisperVQEncoder
from speech_tokenizer.utils import extract_speech_token
from flow_inference import AudioDecoder
import flow_inference

# MONKEY-PATCH: Fix GLM-4-Voice broadcast error in streaming mode
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

VOICE_PROFILES_DIR = "/app/voice_profiles"

class GLMVoiceEngine:
    def __init__(self, model_path, tokenizer_path, flow_path, device='cuda'):
        self.device = device
        self.lock = Lock()
        self.gen_lock = Lock()
        
        logger.info(f"Loading Model: {model_path}")
        
        # 1. Main GLM-4-Voice Model
        self.glm_model = AutoModel.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
        ).to(device).eval()
        
        # 2. Tokenizers & Extractors
        self.glm_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.whisper_model = WhisperVQEncoder.from_pretrained(tokenizer_path).to(device).eval()
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(tokenizer_path)
        
        # 3. Flow-based Audio Decoder
        logger.info(f"Loading Audio Decoder from {flow_path}")
        
        config_path = flow_path
        
        if os.path.isdir(flow_path):
            config_path = os.path.join(flow_path, "config.yaml")
            flow_ckpt = os.path.join(flow_path, "flow.pt")
            hift_ckpt = os.path.join(flow_path, "hift.pt")
            
            for f in ["flow.pt", "flow.pth", "mp_rank_00_model_states.pt"]:
                p = os.path.join(flow_path, f)
                if os.path.exists(p):
                    flow_ckpt = p
                    break
            for f in ["hift.pt", "hift.pth"]:
                p = os.path.join(flow_path, f)
                if os.path.exists(p):
                    hift_ckpt = p
                    break
            self.audio_decoder = AudioDecoder(config_path, flow_ckpt, hift_ckpt, device)
            
        self.system_id = 151335
        self.user_id = 151336
        self.assistant_id = 151337
        self.observation_id = 151338
        self.begin_audio_id = 151343
        self.end_audio_id = 151344
        
        self._scan_audio_vocab()
        # 3. Matcha-style Mel Spectrogram basis (Slaney)
        # GLM-4-Voice/CosyVoice uses Matcha-TTS mel extraction which is very specific:
        # center=False, Magnitude (not power), and natural log.
        self.mel_basis = torchaudio.functional.melscale_fbanks(
            n_freqs=1024 // 2 + 1,
            f_min=0, f_max=8000, n_mels=80,
            sample_rate=22050, norm="slaney", mel_scale="slaney"
        ).to(device).T  # [80, 513]

        # Voice cloning state
        self.ref_prompt_feat  = None  # [1, T, 80] float32 log-mel
        self.ref_prompt_token = None  # [1, N] int32 Whisper VQ tokens
        self.ref_name         = None  # str label (derived from Discord username)
        
    def _scan_audio_vocab(self):
        logger.info("Scanning vocabulary for audio tokens...")
        audio_zero_id = self.glm_tokenizer.convert_tokens_to_ids("<|audio_0|>")
        if audio_zero_id and audio_zero_id != self.glm_tokenizer.unk_token_id:
            self.audio_offset = audio_zero_id
            logger.info(f"Found audio_offset via lookup: {self.audio_offset}")
        else:
            self.audio_offset = 151552 
            for i in range(150000, 162560):
                try:
                    tname = self.glm_tokenizer.decode([i])
                    if "<|audio_0|>" in tname:
                        self.audio_offset = i
                        logger.info(f"Found audio_offset via scan: {i}")
                        break
                except:
                    continue
        
        self.true_vocab_size = self.glm_model.get_input_embeddings().weight.shape[0]
        logger.info(f"Ready! Vocab size: {self.true_vocab_size} | Offset: {self.audio_offset}")

    # ------------------------------------------------------------------
    # Voice reference / cloning
    # ------------------------------------------------------------------

    def _compute_matcha_mel(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Replicates matcha.utils.audio.mel_spectrogram exactly.
        This is the feature space GLM-4-Voice/CosyVoice was trained on.
        """
        # audio: [1, T] float32 on device
        # 1. Mel parameters aligned with CosyVoice standards (50Hz frame rate)
        # 22050 / 441 = 50 Hz. This aligns with the model's acoustic expectations.
        n_fft, hop_size, win_size = 1024, 441, 1024
        window = torch.hann_window(win_size).to(audio.device)
        
        # 2. STFT with centering (standard for CosyVoice)
        # This prevents the time-shift/ghosting artifacts seen with center=False.
        spec = torch.stft(
            audio, n_fft, hop_length=hop_size, win_length=win_size,
            window=window, center=True, pad_mode="reflect",
            normalized=False, onesided=True, return_complex=True
        )
        spec = torch.view_as_real(spec) # [1, 513, T', 2]
        
        # 3. Power Spectrogram (Power, not Magnitude!)
        # Switching from Magnitude to Power doubles the log-scale energy.
        spec = spec.pow(2).sum(-1) + 1e-9 # [1, 513, T']
        
        # 4. Mel Projection
        mel = torch.matmul(self.mel_basis, spec) # [1, 80, T']
        
        # 5. Spectral Normalization (Natural Log of Power)
        mel = torch.log(torch.clamp(mel, min=1e-5))

        # 6. Standardization (Z-Score)
        # Center the features around 0 with unit variance. This ensures the 
        # reference features are in the "sweet spot" of the model's sensitivity,
        # helping to override the default voice bias and fixing "scratchiness".
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)
        
        return mel.transpose(1, 2) # [1, T', 80]

    def set_voice_reference(self, pcm_b64: str, name: str, sample_rate: int = 48000):
        """
        Compute and GPU-pin a voice reference from raw PCM audio.

        Accepts either:
          - 48000Hz stereo int16 (from Discord, sample_rate=48000)
          - 16000Hz mono float32 (legacy, sample_rate=16000)

        Mel spectrogram parameters match CosyVoice's mel_feat_conf exactly.
        Both resamples (to 22050Hz and 16kHz) are done from the original source
        in a single GPU pass — no double-resampling.
        """
        pcm_bytes = base64.b64decode(pcm_b64)

        if sample_rate == 48000:
            # Discord 48kHz stereo int16 → mono float32 on GPU
            samples = torch.frombuffer(bytearray(pcm_bytes), dtype=torch.int16).to(self.device).float() / 32768.0
            mono_48k = samples[0::2].unsqueeze(0)  # left channel, [1, T]

            # Single-step GPU resamples from 48kHz source
            audio_22k = F_audio.resample(mono_48k, 48000, 22050)  # for Mel
            audio_16k = F_audio.resample(mono_48k, 48000, 16000)  # for Whisper VQ

            # Peak normalization: Ensure the reference audio reaches full scale [-1, 1]
            # to provide the strongest stylistic conditioning signal to the model.
            max_val = audio_22k.abs().max()
            if max_val > 1e-6:
                audio_22k = audio_22k / max_val
                audio_16k = audio_16k / max_val
        else:
            # Legacy 16kHz mono float32 path
            audio_16k = torch.frombuffer(bytearray(pcm_bytes), dtype=torch.float32).to(self.device).unsqueeze(0)
            
            # Peak normalization
            max_val = audio_16k.abs().max()
            if max_val > 1e-6:
                audio_16k = audio_16k / max_val

            audio_22k = F_audio.resample(audio_16k, 16000, 22050)  # for Mel

        # 80-dim Matcha-style Mel spectrogram.
        # This uses Magnitude (linear), center=False, and natural Log.
        # This aligns perfectly with the decoder's flow matching conditioning.
        ref_feat = self._compute_matcha_mel(audio_22k)  # [1, T', 80] float32


        # VQ tokens from 16kHz audio via Whisper encoder (GPU)
        with torch.no_grad():
            tokens = extract_speech_token(
                self.whisper_model, self.feature_extractor, [(audio_16k, 16000)]
            )[0]
        ref_token = torch.tensor(tokens, device=self.device, dtype=torch.int32).unsqueeze(0)

        # Cache on GPU
        self.ref_prompt_feat  = ref_feat
        self.ref_prompt_token = ref_token
        self.ref_name         = name

        self._save_voice_profile(name, ref_feat, ref_token)

        n_frames   = ref_feat.shape[1]
        n_tokens   = ref_token.shape[1]
        duration_s = round(n_frames * 256 / 22050, 1)
        logger.info(
            f"[Engine] Voice profile '{name}' ready: "
            f"{n_tokens} VQ tokens, {n_frames} Mel frames "
            f"(\u2248 {duration_s}s of reference audio) \u2014 pinned on GPU."
        )


    def _save_voice_profile(self, name: str, prompt_feat: torch.Tensor, prompt_token: torch.Tensor):
        """Persist a voice profile to disk so it survives container restarts."""
        os.makedirs(VOICE_PROFILES_DIR, exist_ok=True)
        path = os.path.join(VOICE_PROFILES_DIR, f"{name}.pt")
        torch.save({
            "prompt_feat":  prompt_feat.cpu(),
            "prompt_token": prompt_token.cpu(),
        }, path)
        logger.info(f"[Engine] Voice profile saved → {path}")

    def load_voice_profile(self, name: str) -> bool:
        """Load a previously saved voice profile from disk onto the GPU."""
        path = os.path.join(VOICE_PROFILES_DIR, f"{name}.pt")
        if not os.path.exists(path):
            logger.warning(f"[Engine] Voice profile '{name}' not found at {path}")
            return False
        data = torch.load(path, map_location=self.device)
        self.ref_prompt_feat  = data["prompt_feat"].to(self.device).float()  # must be float32
        self.ref_prompt_token = data["prompt_token"].to(self.device)
        self.ref_name         = name
        logger.info(f"[Engine] Voice profile '{name}' loaded from disk → GPU.")
        return True

    def clear_voice_reference(self):
        """Reset to default (no voice cloning)."""
        self.ref_prompt_feat  = None
        self.ref_prompt_token = None
        self.ref_name         = None
        logger.info("[Engine] Voice reference cleared.")

    def prepare_input(self, messages):
        role_map = {
            "system": self.system_id,
            "user": self.user_id,
            "assistant": self.assistant_id
        }
        
        input_ids = []
        
        # 1. Anchor the LLM's identity to the active voice reference if set.
        # This prevents the "default voice" from dominating the output tokens.
        if self.ref_prompt_token is not None:
            input_ids.append(self.system_id)
            input_ids.extend(self.glm_tokenizer.encode("ACT AS the speaker in the following audio. Match their pitch, timbre, and cadence EXACTLY in your response:", add_special_tokens=False))
            input_ids.append(self.begin_audio_id)
            # Flatten [1, N] tensor to list
            input_ids.extend([t + self.audio_offset for t in self.ref_prompt_token[0].tolist()])
            input_ids.append(self.end_audio_id)

        for i, msg in enumerate(messages):
            role = msg.get("role", "user")
            
            role_token = role_map.get(role, self.user_id)
            input_ids.append(role_token)
            
            content = msg.get("content", "")
            if content:
                content_ids = self.glm_tokenizer.encode(content, add_special_tokens=False)
                input_ids.extend(content_ids)
            
            audio_tokens = msg.get("audio_tokens", [])
            if audio_tokens and all(t < 30000 for t in audio_tokens):
                audio_tokens = [t + self.audio_offset for t in audio_tokens]
                
            audio_segments = msg.get("audio_segments", [])
            if audio_segments:
                for segment in audio_segments:
                    seg_tokens = self.process_segment(segment)
                    if seg_tokens: audio_tokens.extend(seg_tokens)

            if role == "user" and not content and not audio_tokens:
                logger.info("Skipping empty user turn in prompt assembly.")
                continue

            if audio_tokens:
                input_ids.append(self.begin_audio_id)
                input_ids.extend(audio_tokens)
                input_ids.append(self.end_audio_id)
                    
        input_ids.append(self.assistant_id)
        
        try:
            log_parts = []
            in_audio = False
            self._logged_raw_audio_marker = False
            trans_text = " [AUDIO TOKENS] "
            for m in messages:
                if m.get("transcription"):
                    trans_text = f" [TRANSCRIPTION: {m['transcription'].strip()}] "
                    
            for tid in input_ids:
                if tid == self.begin_audio_id:
                    log_parts.append(self.glm_tokenizer.decode([tid]))
                    log_parts.append(trans_text)
                    in_audio = True
                elif tid == self.end_audio_id:
                    in_audio = False
                    log_parts.append(self.glm_tokenizer.decode([tid]))
                elif not in_audio:
                    if self.audio_offset <= tid < self.audio_offset + 30000:
                        # Raw audio token without a wrapper
                        if not getattr(self, '_logged_raw_audio_marker', False):
                            log_parts.append(" [RAW AUDIO TOKENS] ")
                            self._logged_raw_audio_marker = True
                    else:
                        text = self.glm_tokenizer.decode([tid])
                        log_parts.append(text)
            
            prompt_str = "".join(log_parts)
            logger.info("╔══ PROMPT SENT TO MODEL ══════════════════════════════════════")
            logger.info(prompt_str)
            logger.info("╚══════════════════════════════════════════════════════════════")
        except Exception as e:
            logger.error(f"Prompt logging failed: {e}")

        input_ids = [min(tid, self.true_vocab_size - 1) for tid in input_ids]
        
        return {"input_ids": torch.tensor([input_ids]).to(self.device)}


    def process_segment(self, segment):
        try:
            import torchaudio.functional as F_audio
            audio_b64 = segment.get("audio")
            if not audio_b64: return []
            
            audio_data = base64.b64decode(audio_b64)
            samples_stereo = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            left_channel = samples_stereo[0::2]
            
            tensor_48k = torch.from_numpy(left_channel).to(self.device).unsqueeze(0)
            tensor_16k = F_audio.resample(tensor_48k, 48000, 16000)
            
            with torch.no_grad():
                audio_tokens = extract_speech_token(
                    self.whisper_model, 
                    self.feature_extractor, 
                    [(tensor_16k, 16000)]
                )[0]
                
                audio_tokens = [t + self.audio_offset for t in audio_tokens]
                return audio_tokens
        except Exception as e:
            logger.error(f"Audio processing error: {e}")
            return []

    @torch.inference_mode()
    def safe_generate(self, **kwargs):
        with self.gen_lock:
            try:
                logger.info("Lock acquired. Starting generation...")
                return self.glm_model.generate(**kwargs)
            except Exception as e:
                logger.error(f"Generation Error: {e}")
                return None
