import torch
import torchaudio
import torchaudio.functional as F_audio
import numpy as np
from core.logger import setup_logger

logger = setup_logger("server.diagnostics")

class AcousticAnalyzer:
    """
    The Forensic Voice Laboratory (The Identity Audit).
    
    WHY THIS EXISTS:
    Voice cloning is a game of 'Probabilistic Approximation'. To ensure the 
    AI isn't hallucinating a generic persona, we need a way to measure the 
    'Acoustic Residue' of the generated signal against the physical 
    constants of the reference speaker. 
    
    CONCEPT: FORENSIC PHYSICS
    We don't just look at the wave; we look at its 'Spectral Mass' and 
    'Geometry'. This class performs a forensic audit of the voice to prove—
    with numbers—that the bot is actually 'acting' correctly.
    """
    
    @staticmethod
    def calculate_similarity(ref_emb, gen_emb):
        """
        The Geometry of Identity (The Hyperspace Laser).
        
        WHY: 
        A 192-dimensional identity vector defines a specific point in 
        'Speaker Hyperspace'. Cosine similarity measures the angular 
        alignment between the reference 'Laser' and the generated one. 
        If they point in the same direction, the 'Soul' of the voice matches.
        """
        if ref_emb is None or gen_emb is None: return 0.0
        v1 = ref_emb / (ref_emb.norm(dim=-1, keepdim=True) + 1e-6)
        v2 = gen_emb / (gen_emb.norm(dim=-1, keepdim=True) + 1e-6)
        return torch.mm(v1, v2.t()).item()

    @staticmethod
    def get_pitch_stats(audio: torch.Tensor, sr: int):
        """
        WHY F0 (Fundamental Frequency)?
        F0 is the frequency at which your vocal cords vibrate. It's the 'baseline' 
        of your voice. If you speak at 120Hz but the bot clones you at 200Hz, 
        you're going to sound like you just inhaled a balloon of helium. 
        We track this delta to ensure the bot stays in your 'Vocal Register'.
        Identity similarity might capture your 'texture', but if the pitch is off,
        it breaks the illusion of roleplay.
        """
        try:
            pitch = F_audio.detect_pitch_frequency(audio, sr)
            voiced = pitch[pitch > 0]
            if voiced.numel() == 0: return 0.0, 0.0
            return voiced.mean().item(), voiced.std().item()
        except Exception as e:
            logger.warning(f"Pitch analysis failed: {e}")
            return 0.0, 0.0

    @staticmethod
    def get_spectral_centroid(audio: torch.Tensor, sr: int):
        """
        WHY Spectral Centroid?
        The centroid is the 'Center of Mass' of the spectrum. 
        Higher Centroid = Brighter, sharper, more high-frequency energy.
        Lower Centroid = Darker, muffled, bassier.
        If the bot's centroid is 20% higher than yours, it's probably 
        introducing 'robotic' artifacts or over-sharpening the vocoder output.
        It's basically a 'Quality Control' check for audio texture.
        """
        try:
            spec = torchaudio.transforms.Spectrogram(n_fft=1024).to(audio.device)(audio)
            freqs = torch.linspace(0, sr / 2, spec.shape[-2]).to(audio.device)
            centroid = (spec.transpose(-1, -2) @ freqs.unsqueeze(-1)) / (spec.sum(dim=-2, keepdim=True).transpose(-1, -2) + 1e-6)
            return centroid.mean().item()
        except Exception as e:
            logger.warning(f"Spectral analysis failed: {e}")
            return 0.0

    @classmethod
    def run_full_report(cls, engine, gen_audio_16k, ref_emb, ref_stats=None, ref_audio_16k=None):
        """
        WHY the Trust Index?
        Users (that's you) don't want to look at a spreadsheet of Hz and dB. 
        The Trust Index is a weighted average of Identity, Pitch, and Color. 
        - Identity (75%): If the person isn't YOU, nothing else matters.
        - Pitch (12.5%): Keeps you in your register.
        - Color (12.5%): Keeps you sounding natural.
        We scale it so that a 90+ score means you could probably rob a bank 
        using this bot's voice (not that we recommend it).
        """
        # --- FORENSIC ANALYSIS BEGINS ---
        gen_emb = engine.speaker_encoder.extract_embedding(gen_audio_16k)
        similarity = cls.calculate_similarity(ref_emb, gen_emb)
        
        gen_pitch_mean, _ = cls.get_pitch_stats(gen_audio_16k, 16000)
        gen_centroid = cls.get_spectral_centroid(gen_audio_16k, 16000)
        
        ref_pitch_mean = ref_stats.get("pitch_mean", 0.0) if ref_stats else 0.0
        ref_centroid = ref_stats.get("centroid", 0.0) if ref_stats else 0.0
        
        # Scaling logic: 0.6 similarity is 'Base', 0.9 is 'Peak'.
        pitch_score = max(0, 1 - (abs(gen_pitch_mean - ref_pitch_mean) / 60)) if ref_pitch_mean > 0 else 1.0
        color_delta_pct = (abs(gen_centroid - ref_centroid) / ref_centroid) if ref_centroid > 0 else 0.0
        color_score = max(0, 1 - (color_delta_pct / 0.4)) 
        
        adj_similarity = max(0, min(1.0, (similarity - 0.6) / 0.3))
        fidelity_score = (adj_similarity * 75) + (pitch_score * 12.5) + (color_score * 12.5)

        # Hard Measurements: Energy & Density
        gen_rms = torch.sqrt(torch.mean(gen_audio_16k**2)).item()
        ref_rms = ref_stats.get("rms", 0.0) if ref_stats else 0.0
        
        # Identity Density (Norm)
        gen_norm = gen_emb.norm(dim=-1).mean().item()
        ref_norm = ref_emb.norm(dim=-1).mean().item()

        # 4. Logging the forensic results
        logger.info(f"┌─── CLONE FIDELITY REPORT ────────────────────────────")
        
        # CONCEPT: The 'Fidelity Trust' Branch
        # We call it the 'Fidelity Trust Index' because, much like a bank, 
        # we're managing your most valuable asset: your identity. 
        # But remember, as the boss said, we're not here to rob banks—we're just 
        # opening a high-interest identity account for your RPG NPCs.
        # If your 'Fidelity' is high enough, you might even pass a biometric 
        # withdrawal test (Disclaimer: do not actually try to rob banks).
        logger.info(f"│ Trust Index: {fidelity_score:.1f}/100 " + ("🌟 (Identity Theft level)" if fidelity_score > 90 else "✅ (Good Enough for RPG)" if fidelity_score > 70 else "⚠️ (Generic Robot)"))
        logger.info(f"│ Identity Similarity: {similarity:.4f} (Density: {gen_norm:.2f} vs Ref: {ref_norm:.2f})")
        logger.info(f"│ Sub-Scores: Identity={adj_similarity*100:.1f}, Pitch={pitch_score*100:.1f}, Color={color_score*100:.1f}")
        
        if ref_pitch_mean > 0:
            pitch_delta = abs(gen_pitch_mean - ref_pitch_mean)
            logger.info(f"│ Pitch Register: {gen_pitch_mean:.1f}Hz (Ref: {ref_pitch_mean:.1f}Hz, Δ: {pitch_delta:.1f}Hz)")
        if ref_centroid > 0:
            color_delta_p = color_delta_pct * 100
            logger.info(f"│ Spectral Color: {gen_centroid:.0f} (Δ: {color_delta_p:.1f}%)")
            
        # Loudness Balance Sheet
        logger.info(f"│ Energy Check: {gen_rms:.4f} RMS (Ref: {ref_rms:.4f} RMS)")
        if gen_rms > 0.1:
            logger.warning("│ ⚠️ OVERDRAFT: Generated audio is clipping (RMS > 0.1).")
        elif gen_rms < 0.005:
            logger.warning("│ ⚠️ INSOLVENT: Generated audio is too quiet/silent.")
        
        # Actionable Forensic Advice
        if similarity < 0.75:
            logger.warning("│ 🔎 FORENSIC: Identity drift detected. Check grounding transcription.")
        if ref_pitch_mean > 0 and abs(gen_pitch_mean - ref_pitch_mean) > 30:
            logger.warning("│ 🔎 FORENSIC: Register shift. Pitch is not matching reference baseline.")
            
        logger.info(f"└──────────────────────────────────────────────────────")
        
        return fidelity_score
