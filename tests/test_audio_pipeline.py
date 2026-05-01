import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import torch
import base64
import asyncio

# Mock the imports that don't exist in the test environment
import sys

# Mock speech_tokenizer
mock_speech_tokenizer = MagicMock()
sys.modules['speech_tokenizer'] = mock_speech_tokenizer
sys.modules['speech_tokenizer.modeling_whisper'] = mock_speech_tokenizer
sys.modules['speech_tokenizer.utils'] = mock_speech_tokenizer

# Mock transformers
mock_transformers = MagicMock()
sys.modules['transformers'] = mock_transformers

# Mock flow_inference
mock_flow_inference = MagicMock()
sys.modules['flow_inference'] = mock_flow_inference

# Mock bridge.glm so it doesn't import discord or load real whisper
mock_bridge_glm = MagicMock()
sys.modules['bridge.glm'] = mock_bridge_glm

# Now import our components safely
from bridge.components import AudioResampler, WhisperTokenizer, GLMAudioDecoder

class TestAudioPipeline(unittest.TestCase):
    def setUp(self):
        self.resampler = AudioResampler()
        self.tokenizer = WhisperTokenizer(device="cpu")
        self.decoder = GLMAudioDecoder(device="cpu")
        
        # 48kHz stereo int16 dummy data (0.5 seconds)
        self.dummy_pcm = np.zeros(int(48000 * 0.5 * 2), dtype=np.int16).tobytes()

    def test_audio_resampler(self):
        # Test Downsample
        audio_16k = self.resampler.downsample(self.dummy_pcm)
        self.assertEqual(audio_16k.shape[0], int(16000 * 0.5))
        self.assertEqual(audio_16k.dtype, np.float32)

        # Test Upsample
        dummy_generated = torch.zeros(int(22050 * 0.5))
        audio_48k = self.resampler.upsample(dummy_generated, orig_sr=22050, target_sr=48000)
        self.assertEqual(len(audio_48k), int(48000 * 0.5 * 2 * 2)) # frames * channels * 2 bytes

    @patch('bridge.components.WhisperTokenizer._ensure_loaded')
    @patch('speech_tokenizer.utils.extract_speech_token')
    def test_whisper_tokenizer(self, mock_extract, mock_ensure_loaded):
        audio_16k = np.zeros(8000, dtype=np.float32)
        
        # Test feature extraction
        features = self.tokenizer.extract_features(audio_16k)
        self.assertEqual(len(features), 1)
        self.assertEqual(features[0][1], 16000)
        
        # Test tokenization
        mock_extract.return_value = [[1, 2, 3]]
        tokens = self.tokenizer.get_vq_tokens(features)
        self.assertEqual(tokens, [1, 2, 3])

    @patch('bridge.components.GLMAudioDecoder._ensure_loaded')
    def test_glm_audio_decoder(self, mock_ensure_loaded):
        tokens = [1, 2, 3]
        
        # Test prepare tokens
        tensor = self.decoder.prepare_tokens(tokens)
        self.assertEqual(tensor.shape, (1, 3))
        
        # Test generate flow mel
        mel = self.decoder.generate_flow_mel(tensor)
        self.assertEqual(mel.shape, (1, 3))
        
        # Test vocode to waveform
        self.decoder._decoder = MagicMock()
        self.decoder._decoder.token2wav.return_value = (torch.zeros(11025), None) # 0.5s at 22050
        
        waveform = self.decoder.vocode_to_waveform(mel)
        self.assertEqual(waveform.shape[0], 11025)

if __name__ == '__main__':
    unittest.main()
