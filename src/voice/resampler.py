"""
WHY THIS FILE EXISTS:
A high-performance audio resampling utility designed to bridge the gap between 
Discord's 48kHz audio and various AI model sample rates (16kHz, 24kHz, etc.).

WHY USE TORCHAUDIO?
Because standard scipy/numpy resampling is slow and doesn't leverage the GPU.
By using torchaudio.transforms.Resample, we get high-quality Sinc interpolation 
that can run on CUDA if the tensors are already there.
"""
import torch
import torchaudio
import logging

logger = logging.getLogger("voice.resampler")

class UniversalResampler:
    """
    A high-quality audio resampler utility to handle conversions between
    Discord (48kHz) and various AI models (16kHz, 24kHz, etc.).
    
    This class explicitly logs its configuration to avoid "resampling confusion."
    """
    def __init__(self, src_rate: int, dst_rate: int, name: str = "Audio"):
        self.src_rate = src_rate
        self.dst_rate = dst_rate
        self.name = name
        
        if src_rate == dst_rate:
            self.resampler = None
            logger.info(f"[{name} Resampler] Passthrough mode enabled ({src_rate}Hz -> {dst_rate}Hz)")
        else:
            # We use torchaudio's high-quality Resample transform
            self.resampler = torchaudio.transforms.Resample(src_rate, dst_rate)
            # Pre-warm the resampler to avoid latency on first call
            dummy = torch.zeros(1, 1024)
            self.resampler(dummy)
            
            ratio = dst_rate / src_rate
            logger.info(f"[{name} Resampler] Configured: {src_rate}Hz -> {dst_rate}Hz (Ratio: {ratio:.4f})")

    def __call__(self, audio_tensor: torch.Tensor) -> torch.Tensor:
        """
        Resample the input tensor.
        Input shape: (C, T) or (T,)
        Output shape: (C, T_new)
        """
        if self.resampler is None:
            return audio_tensor
            
        # Ensure 2D (C, T)
        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
            
        return self.resampler(audio_tensor)

    def resample_bytes(self, pcm_bytes: bytes, channels: int = 1, src_dtype: str = "float32") -> bytes:
        """
        Helper to resample raw bytes and return raw bytes.
        
        WHY: Most AI models output float32, but Discord needs int16. 
        This helper handles the conversion gymnastics.
        """
        if self.resampler is None:
            return pcm_bytes
            
        # 1. Convert bytes to tensor 📥
        if src_dtype == "int16":
            import numpy as np
            audio_np = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            audio_tensor = torch.from_numpy(audio_np).unsqueeze(0)
        else:
            # Assume float32
            audio_tensor = torch.frombuffer(pcm_bytes, dtype=torch.float32).view(channels, -1)
        
        # 2. Resample ⚡
        resampled = self.resampler(audio_tensor)
        
        # 3. Convert back to int16 bytes for Discord 📤
        # WHY: Discord requires 48kHz stereo int16. 
        # We duplicate the mono channel to create stereo.
        stereo = torch.cat([resampled, resampled], dim=0) # (2, T)
        int16_audio = (stereo.clamp(-1, 1) * 32767.0).to(torch.int16)
        
        return int16_audio.flatten().numpy().tobytes()
