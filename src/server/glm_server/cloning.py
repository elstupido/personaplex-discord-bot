"""
WHY THIS FILE EXISTS:
The Identity Crystallization Vault & Profile Manager.

WHY THE DISK I/O EXCEPTION:
[SPECIAL DISPENSATION GRANTED BY OPERATOR]
Voice cloning requires 'Identity DNA' persistence. While the hot-path must 
be zero-copy, the 'Crystallization' phase is allowed to touch the disk 
to ensure that reference WAVs and .pt tensors are safely journaled for 
future sessions. We trade a few milliseconds of setup time for a lifetime 
of spectral accuracy.

WHY CRYSTALLIZATION?
Because a human voice isn't just data; it's a 'Spectral Signature'. This 
module extracts the stable timbre (the 'Soul') from transient linguistic 
noise, storing it as a 192-dimensional vector.
"""
import torch
import torch.nn.functional as F
import numpy as np
import os
import json
import torchaudio
from .config import AlignmentConfig, Colors
from core.logger import setup_logger

logger = setup_logger("server.engine")
import tempfile
import torch.nn.functional as F_nn

class SpeakerEncoder:
    """
    The Identity DNA Extractor (CAM++).
    
    WHY: 
    A human voice is more than just words; it's a 'Spectral Signature'. 
    CAM++ allows us to extract this signature as a 192-dimensional vector.
    
    WHY L2 NORMALIZATION?
    If we don't normalize, a loud recording would produce a 'bigger' identity 
    than a quiet one. We want the IDENTITY, not the VOLUME. L2 Norm ensures 
    that every speaker, regardless of how loud they spoke, exists on the 
    surface of the same 192-dimensional hypersphere. It's equality for voices.
    """
    def __init__(self, model_dir: str, device: str):
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks
        logger.info(f"[SpeakerEncoder] Loading identity engine...")
        self.device = device
        self.pipeline = pipeline(
            task=Tasks.speaker_verification,
            model=model_dir,
            model_revision='v1.0.0',
            device=device
        )

    def __call__(self, audio: torch.Tensor, sr: int) -> torch.Tensor:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            # WHY CPU TRANSFER?
            # ModelScope/CAM++ pipelines often expect CPU-bound file paths. 
            # We detach and move to CPU here to satisfy the legacy interface.
            audio_cpu = audio.detach().cpu()
            if audio_cpu.ndim == 1: audio_cpu = audio_cpu.unsqueeze(0)
            
            # WHY SAVE TO DISK? 
            # [EXCEPTION RULE] The speaker verification pipeline requires a 
            # file path. We use a temporary WAV as the 'Sacrificial Lamb' 
            # to bridge the gap between our tensor river and the model's 
            # file-based intake.
            torchaudio.save(tmp_path, audio_cpu, sr)
            
            with torch.no_grad():
                res = self.pipeline.forward(self.pipeline.preprocess([tmp_path]))
                val = res.get('spk_embedding') if isinstance(res, dict) else res
                emb = torch.as_tensor(val, device=self.device, dtype=torch.float32).clone().detach()
                
                cfg = getattr(self, 'alignment_config', None)
                if cfg:
                    if cfg.emb_l2_norm:
                        emb = F_nn.normalize(emb, p=2, dim=-1)
                    
                    # WHY MVN? Mean-Variance Normalization ensures the embedding
                    # distribution matches the expected input of the GLM transformer.
                    if cfg.emb_mvn:
                        emb = (emb - emb.mean()) / (emb.std() + 1e-6)
                        
                    emb = emb * cfg.emb_scalar
                
                return emb[:, :192] if emb.shape[-1] > 192 else emb
        finally:
            if os.path.exists(tmp_path): os.remove(tmp_path)

class IdentityManager:
    """
    The Identity Crystallization Vault (The Essence Extractor).
    
    WHY THIS CLASS EXISTS:
    Voice cloning is fundamentally an 'Embedding' problem. A human voice is a 
    high-dimensional signal containing linguistic data, emotional state, and 
    'Timbre' (the unique physical resonance of your throat and mouth). 
    
    This class exists to perform 'Identity Crystallization'—the process of 
    isolating the stable, physical timbre from the transient linguistic 
    noise. By capturing this as a 'Character Profile', we allow the synthesis 
    engine to project any semantic intent (text) through the lens of a 
    specific physical identity.
    
    WHY MODULARITY?
    If the 'Synthesis Engine' handled identity, it would have to re-extract 
    features on every single turn. By moving this to a dedicated 'Vault', 
    we treat identities as immutable 'Crystals'. We extract them once, 
    normalize them to a standard 'Energy Density', and store them for 
    instant retrieval. This separation of 'Who' (Identity) from 'What' 
    (Synthesis) is what makes our pipeline both fast and physically accurate.
    
    Sarcastic Note: We're basically building a digital soul-catcher for NPCs. 
    It captures the 'Acoustic Signature' without the 'Existential Dread'. 
    Use responsibly, or at least don't clone your boss without a good lawyer.
    """
    
    def __init__(self, profile_dir="/app/voice_profiles"):
        self.profile_dir = profile_dir
        os.makedirs(self.profile_dir, exist_ok=True)
        self.cfg = AlignmentConfig()
        
        # Identity Cache: Because loading from disk is for people who like waiting.
        self.vault = {}
        
        # We don't hold the models here; the Engine does. 
        # But we provide the logic to process them.

    def crystallize_identity(self, engine, pcm_b64, name, sample_rate=48000, prompt_text=""):
        """
        Extracts the speaker's essence and stores it in the vault.
        
        WHY THE GROUNDING TEXT?
        Zero-shot cloning is a 'Multimodal Rosetta Stone' problem. If we provide 
        the model with just audio, it has to guess the phonemes. By providing the 
        text ('Rosetta Stone'), the model can perfectly map the speaker's 
        unique timbre to the known linguistic structure. This results in a 
        'Fidelity Trust Index' that would make a banker weep.
        """
        import base64
        import io

        # 1. Decode and Normalization
        # We expect 48kHz (Discord standard) but we need to verify.
        pcm_bytes = base64.b64decode(pcm_b64)
        audio, sr = torchaudio.load(io.BytesIO(pcm_bytes))
        
        # Ensure mono float32 for the extraction pipeline
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        
        # Move to GPU for processing
        audio_gpu = audio.to(engine.device)

        # 2. Extract Acoustic Tokens (Linguistic structure)
        # WHY: 
        # This layer extracts the 'What'—the discrete linguistic units and 
        # their temporal structure (cadence, prosody). We use a resolution 
        # optimized for human speech intelligibility. This ensures the model 
        # understands the phonetic boundaries of the speaker.
        # WHY 16kHz?
        # Whisper and SenseVoice were trained on the 'Telephonic Standard' 
        # of 16kHz. Passing 48kHz would be like trying to read a book through 
        # a telescope—too much detail, not enough context.
        audio_16k = torchaudio.functional.resample(audio_gpu, sr, 16000)
        
        from speech_tokenizer.utils import extract_speech_token
        tokens = extract_speech_token(
            engine.whisper_model, 
            engine.feature_extractor, 
            [(audio_16k, 16000)]
        )[0]
        ref_tokens = torch.as_tensor(tokens, device=engine.device, dtype=torch.long).unsqueeze(0)

        # 3. Extract Mel Features (Timbre conditioning)
        # WHY: 
        # This layer extracts the 'How'—the high-resolution spectral 
        # fingerprint of the voice. Unlike linguistic tokens, timbre 
        # requires a much finer temporal grid to capture the subtle 
        # textures that make a voice sound human rather than synthetic. 
        # We must align this grid with the vocoder's expectations to 
        # avoid temporal warping or artifacts.
        ref_feat = engine._compute_matcha_mel(audio_gpu, sr)
        
        # 4. Extract Speaker Embedding (The Identity Vector)
        # WHY: We delegate the heavy lifting (and normalization) to the 
        # speaker_encoder itself, as it knows the 'Acoustic Physics' 
        # required for the embedding space.
        embedding = engine.speaker_encoder(audio_gpu, sr)
        
        # 5. Save to Vault
        profile = {
            "name": name,
            "embedding": embedding,
            "prompt_token": ref_tokens,
            "prompt_feat": ref_feat,
            "prompt_text": prompt_text,
            "stats": {
                "rms": torch.sqrt(torch.mean(audio**2)).item(),
                "duration": audio.shape[1] / sr
            }
        }
        
        self.vault[name] = profile
        self._persist_profile(name, profile)
        
        logger.info(f"{Colors.GREEN}[IdentityManager] Identity '{name}' crystallized. Trust Index: High.{Colors.RESET}")
        return profile

    def _persist_profile(self, name, profile):
        """Save the digital soul to disk."""
        path = os.path.join(self.profile_dir, f"{name}.pt")
        # We save as a .pt file because it handles tensors natively. 
        # JSON is for mortals; Tensors are for gods.
        torch.save(profile, path)

    def load_profile(self, name):
        """Retrieve a soul from the vault."""
        if name in self.vault:
            return self.vault[name]
            
        path = os.path.join(self.profile_dir, f"{name}.pt")
        if os.path.exists(path):
            profile = torch.load(path)
            self.vault[name] = profile
            return profile
            
        return None

    def list_profiles(self):
        """Show me the collection."""
        return [f.replace(".pt", "") for f in os.listdir(self.profile_dir) if f.endswith(".pt")]
