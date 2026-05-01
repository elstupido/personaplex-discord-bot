import torch
import torchaudio
import logging

logger = logging.getLogger("utils.resampler")

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

    def resample_bytes(self, pcm_bytes: bytes, channels: int = 1, dtype: torch.dtype = torch.float32) -> bytes:
        """
        Helper to resample raw bytes and return raw bytes.
        Assumes float32 input/output.
        """
        if self.resampler is None:
            return pcm_bytes
            
        # Convert bytes to tensor
        audio_np = torch.from_blob(bytearray(pcm_bytes), (len(pcm_bytes) // 4,), dtype=dtype)
        audio_tensor = audio_np.view(channels, -1)
        
        # Resample
        resampled = self.resampler(audio_tensor)
        
        # Convert back to bytes
        return resampled.flatten().numpy().tobytes()
