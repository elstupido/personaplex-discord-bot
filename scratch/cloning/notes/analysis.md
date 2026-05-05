# Research Notes: Voice Cloning Fidelity Analysis (GLM-4-Voice)

## Overview
This research compares our current voice cloning implementation in `personaplex-discord-bot` with the `GLM-TTS` project from `zai-org`. The goal is to identify discrepancies in acoustic feature extraction and conditioning that contribute to identity drift.

## 1. Acoustic Feature Calibration (Mel Spectrograms)
The most significant discrepancy found is in the **temporal resolution** and **normalization** of the Mel spectrograms used for Flow-Matching conditioning.

### GLM-TTS (Remote Implementation)
- **Hop Size**: 20ms (480 samples @ 24kHz, 640 @ 32kHz).
- **Window Size**: 80ms (1920 samples @ 24kHz, 2560 @ 32kHz).
- **Normalization**: `spec = torch.log(torch.clamp(spec, min=1e-5))`. (Natural log).
- **Frequency Range**: 0 Hz to 8000 Hz.
- **Frame Rate**: 50 frames per second (FPS).

### Our Project (Local Implementation)
- **Hop Size**: 10ms (160 samples @ 16kHz via Whisper Feature Extractor).
- **Normalization**: `norm_mel = (mel - (-15.0)) / 1.0`. (Based on `log10` scaling in config).
- **Frame Rate**: 100 frames per second (FPS).
- **Impact**: We are providing twice as many Mel frames as the remote implementation for the same audio duration. If the decoder was trained on 50 FPS Mels, this 100 FPS input will cause severe identity misalignment.

## 2. The Role of Text Prompts
The `GLM-TTS` implementation highlights that **accurate prompt text improves speaker similarity**.

- **Remote Method**: Captures the exact text spoken in the reference audio, normalizes it, and includes it in the LLM context paired with the audio tokens.
- **Local Method**: Uses a hardcoded instruction: `"ACT AS the speaker in the following audio..."`.
- **Opportunity**: By transcribing the clone reference (using Whisper) and providing that text to the LLM, we can ground the "identity" in a specific phoneme-to-audio mapping, which significantly stabilizes timbre.

## 3. Speaker Embedding Model
There is a model mismatch in the speaker verification encoder:
- **Remote**: Uses **CAM++** (`campplus.onnx`).
- **Local**: Uses **ERes2Net** (`iic/speech_eres2net_sv_zh-cn_16k-common`).
- **Alignment**: ERes2Net is generally more robust, but if the Flow-Matching decoder was conditioned on CAM++ embeddings during training, using ERes2Net will result in an "out-of-distribution" embedding, leading to timbre degradation.

## 4. Key Parameters for `_process_voice_clone`
To achieve high-fidelity cloning, the following parameters should be aligned in our `_process_voice_clone` and server-side `set_voice_reference`:

| Parameter | Recommended Value | Rationale |
| :--- | :--- | :--- |
| **Mel Hop** | 20ms (441 @ 22050Hz) | Matches 50 FPS training target of GLM Flow models. |
| **Mel Window** | 80ms (1764 @ 22050Hz) | Standard window for Flow-Matching acoustic features. |
| **Normalization** | `ln(x)` with `1e-5` clamp | Avoids the arbitrary `-15.0` mean shift if the model expects raw log-mels. |
| **Prompt Text** | Transcribed from audio | Provides the LLM with a concrete few-shot example of the voice. |
| **Embedding Scalar** | 35.0 (with L2 Norm) | Corrects the magnitude of the speaker vector for the decoder's cross-attention. |

## 5. Summary of Differences
Our current method relies heavily on discrete tokens and a generic prompt. The `GLM-TTS` method treats voice cloning as a **multimodal few-shot task**, where the paired (Text, Discrete Tokens, Continuous Mel Features, Speaker Embedding) tuple defines the identity.

### Recommendation for next steps:
1.  **Instrument the server** to dump the Mel statistics of the reference audio.
2.  **Verify the frame rate** of the `AudioDecoder`. If it's 50Hz, our 100Hz input is the "smoking gun" for identity drift.
3.  **Implement auto-transcription** for clone references to replace the hardcoded "ACT AS" prompt.
