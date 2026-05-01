# Voice Cloning Design — PersonaPlex Discord Bot

## What We Discovered From Source Code

After reading the actual GLM-4-Voice and CosyVoice source files on GitHub (`flow.py`, `flow_inference.py`, `web_demo.py`, `processor.py`, `frontend.py`, `generator.py`), here is the ground-truth architecture.

---

## How Zero-Shot Voice Cloning Works In CosyVoice

The flow decoder (`MaskedDiffWithXvec`) is a **conditional flow-matching model** that generates 80-dim mel spectrograms. It has three conditioning inputs:

| Input | Shape | What it is | How it conditions |
|---|---|---|---|
| `token` | `[1, N]` | VQ tokens (content) | Encoded by transformer → drives *what* is said |
| `prompt_token` | `[1, P]` | Reference VQ tokens | Prepended to `token`; shifts encoder attention toward reference speaker style |
| `prompt_feat` | `[1, T, 80]` | Reference mel frames | Fills `conds[:, :T, :]` in the diffusion conditioning signal; anchors the *starting acoustic trajectory* |
| `embedding` | `[1, 192]` | X-vector speaker embedding | Projected into the decoder at every diffusion step (`spks` argument) |

All three work together. `prompt_token` alone gives partial style transfer. `prompt_feat` provides the mel-space anchor. `embedding` provides global speaker timbre.

---

## The `prompt_feat` Mel Format

From `frontend.py` `_extract_speech_feat()` and `frontend_zero_shot()`:

```python
# Reference audio resampled to 22050 Hz FIRST
prompt_speech_22050 = torchaudio.transforms.Resample(16000, 22050)(prompt_speech_16k)
speech_feat = feat_extractor(prompt_speech_22050).squeeze(0).transpose(0, 1)
# Result shape: [T, 80] → padded to [1, T, 80]
```

The `feat_extractor` is defined in the training config YAML as:

```yaml
feat_extractor: !apply:torchaudio.transforms.MelSpectrogram
  sample_rate: 22050
  n_fft: 1024
  win_length: 1024
  hop_length: 256
  n_mels: 80
  f_min: 0
  f_max: 8000
```

**It does NOT apply log by default.** The raw power mel spectrogram is used as `speech_feat`. However, the `compute_fbank` pipeline normalizes audio to `[-1, 1]` first (`max_val = speech.abs().max(); speech /= max_val`), bounding the power values.

> **Open question:** Does the actual deployed GLM-4-Voice decoder's `config.yaml` apply log or not? This can be confirmed by running `cat /app/glm/glm-4-voice-decoder/config.yaml` inside the container.

---

## The Static Bug We Had — Root Cause

Our code computed:
```python
mel = torch.log(mel.clamp(min=1e-5))   # log-scale values: [-11, +13]
ref_feat = mel.half()                   # fp16
```

Two problems:
1. **Wrong dtype:** `conds` in `flow.inference` is `float32`. Assigning `fp16` into `float32` works via auto-cast, but if the model runs in `bfloat16` mode, the precision mismatch can introduce error.
2. **Possibly wrong scale:** If the model was trained on **linear power mels** (no log), our log-scaled values are in a completely different numerical range (~[-11,+13] vs [0,~50]) — causing the decoder to produce garbage → static.

---

## Correct Implementation Plan

### Step 1 — Reference Mel Extraction (engine.py `set_voice_reference`)

```python
# Resample 48kHz → 22050Hz (done via GPU torchaudio functional resample)
audio_22k = F_audio.resample(audio_48k, 48000, 22050)

# Normalize to [-1, 1] (matching training pipeline)
max_val = audio_22k.abs().max()
if max_val > 0:
    audio_22k = audio_22k / max_val

# Extract mel with EXACT training parameters — no log (linear power)
mel = self.mel_transform(audio_22k)          # [1, 80, T]
ref_feat = mel.transpose(1, 2).float()       # [1, T, 80] float32

# Store on GPU
self.ref_prompt_feat  = ref_feat
self.ref_prompt_token = vq_tokens            # [1, N] int32
```

`self.mel_transform` is pre-built as:
```python
T.MelSpectrogram(sample_rate=22050, n_fft=1024, win_length=1024,
                 hop_length=256, n_mels=80, f_min=0, f_max=8000)
```

### Step 2 — Streaming Generation (app.py `stream_response`)

For each streaming block:

```
Block 1 (first audio chunk):
  prompt_token = [ref_token]                          # reference VQ tokens only
  prompt_feat  = [ref_feat]                           # reference mel frames
  → flow generates: ref_frames + block1_frames
  → slices off ref_frames → outputs block1_frames audio

Block 2+:
  prompt_token = [ref_token | gen_tokens_block1]      # reference + generated so far
  prompt_feat  = [ref_feat  | gen_mel_block1]         # reference mel + generated mel
  → flow generates total frames, slices off accumulated prompt frames
  → outputs only the new block's audio
```

This is identical to `CosyVoiceFrontEnd.frontend_zero_shot()` plus the streaming continuity pattern from `web_demo.py`.

### Step 3 — Slicing Math Verification

With `input_frame_rate=25` (GLM-4-Voice VQ encoder outputs 25 tok/sec):

For T seconds of reference audio and block_size=25 tokens:
- `ref_token.shape[1]` = T × 25 tokens
- `ref_feat.shape[1]` = T × 86 frames (≈ 22050/256)
- `feat_len = (T×25 + 25) / 25 × 86 = (T+1) × 86`
- Output frames = feat_len − ref_feat.shape[1] = `(T+1)×86 − T×86 = 86` frames ✓

86 mel frames = 1 second of audio. Correct for 25-token block (1 second of speech at 25 tok/sec).

---

## What We Are NOT Doing (And Why)

### X-Vector / Speaker Embedding (`flow_embedding`)
CosyVoice uses a `campplus` ONNX model to extract 192-dim speaker embeddings at 16kHz via `torchaudio.compliance.kaldi.fbank`. GLM-4-Voice's deployed model server (`model_server.py`) does NOT use this — it uses the default `torch.zeros(1, 192)`. Adding campplus would require downloading and integrating a separate ONNX model, which is out of scope for now.

### Root Cause: Mel Feature Space Mismatch
The "static" audio issue is caused by the `prompt_feat` being in the wrong numerical space. Our previous investigation into "linear power" was incorrect. 
**Discovery**: GLM-4-Voice's decoder is based on **CosyVoice**, which uses the mel extraction logic from the **Matcha-TTS** project. 

Key differences between Matcha Mel and standard `torchaudio` Mel:
1.  **STFT Centering**: Matcha uses `center=False` with manual `reflect` padding. `torchaudio` defaults to `center=True`.
2.  **Magnitude vs Power**: Matcha uses **Linear Magnitude** (`sqrt(real^2 + imag^2)`). `torchaudio` defaults to **Power** (`real^2 + imag^2`).
3.  **Log Compression**: Matcha uses **Natural Log** (`torch.log(clamp(mel, min=1e-5))`). 
4.  **Mel Basis**: Matcha uses `librosa`'s Slaney mel filterbank.

Passing Power-Mel or Uncentered-Mel to a model trained on Centered-Log-Magnitude-Mel results in the "pure static" the user is seeing.

## Proposed Changes

### [Component] GLM Server Engine

#### [MODIFY] [engine.py](file:///root/personaplex-discord-bot/src/bridge/glm_server/engine.py)
- Implement `_compute_matcha_mel()` helper method that exactly replicates `matcha.utils.audio.mel_spectrogram`.
- Initialize `self.mel_basis` in `__init__` using `torchaudio.functional.melscale_fbanks(norm='slaney', mel_scale='slaney')`.
- Update `set_voice_reference()` to use the new Matcha mel extraction.
- Preserve `float32` precision throughout.

### [Component] GLM Server App

#### [MODIFY] [app.py](file:///root/personaplex-discord-bot/src/bridge/glm_server/app.py)
- Ensure `prompt_feat` is correctly seeded with the reference mel in block 1.
- Ensure continuity by prepending the reference mel to generated mels in subsequent blocks.
- **Note**: Since the model produces Matcha-style mels, the generated mels will naturally match the reference mel space.

---

## Fallback: Confirming Mel Format

If static persists after removing the log, we need to verify the deployed config.yaml. Ask the user to run:
```bash
docker exec <container_id> cat /app/glm/glm-4-voice-decoder/config.yaml | grep -A5 feat_extractor
```
This will definitively tell us whether log is applied inside `feat_extractor`.
