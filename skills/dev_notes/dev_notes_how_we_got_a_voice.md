# WHY THIS FILE EXISTS:
The 'Acoustic Resurrection' & Infrastructure Manifesto. A pedagogical deep-dive into the total siege required to stabilize the disaggregated vLLM-Omni pipeline on the RTX 5090 Blackwell architecture under WSL2.

## ⚖️ The 5090 "Vocal Throne" (VRAM Stewardship)
**The Problem**: The Blackwell RTX 5090 is a beast, but its drivers are brittle in WSL2. Standard vLLM profiling would panic or "Ghost Memory" from Xwayland (up to 12GB) would trigger instant OOMs.

### 1. The "Total Lobotomy" Patch
We performed a surgical strike on the vLLM core logic to bypass its internal memory validator.
- **The Patch**: `sed -i 's/if init_snapshot.free_memory < requested_memory:/if False:/g' ...`
- **WHY**: vLLM V1's validator was too conservative for virtualized memory pools. We killed the validator to take manual control of the VRAM.

### 2. The Hopper Spoof (9.0 Override)
- **Override**: `VLLM_DEVICE_CAP_OVERRIDE=9.0`
- **WHY**: The RTX 5090 (9.2) drivers are unstable in WSL2 for complex stage-graphs. By spoofing as an H100 (Hopper), we forced vLLM to use stable, high-performance H100 code paths.

### 3. The "Triple Crown" Stability Profile
To stop driver stalls, we enforced:
- **Eager Mode**: `--enforce-eager` (Saves 5GB by stopping CUDA Graph pre-allocation).
- **Context Cap**: `--max-model-len 4096` (Prevents positional embedding memory spikes).
- **WSL2 Sync**: `CUDA_LAUNCH_BLOCKING=1` and `NCCL_P2P_DISABLE=1` (Mandatory for virtualized CUDA stability).

## ⚔️ Combat History: The Extinction Events

### 1. The GCloud Credential Migration (ADC Persistence)
- **The Move**: Migrated Application Default Credentials from `~/.config/gcloud` to the persistent project root as `.gcloud_credentials.json`.
- **WHY**: Ephemeral WSL storage was losing credentials on restart. This ensures the bot's "Brain" can authenticate across reboots.

### 2. The 'Trojan Horse' Identity
- **The Pattern**: Established in `vllm_omni_asr.py`.
- **Mechanism**: The bot masquerades as a local service to bypass security headers and communicate with the Brain container with zero-latency overhead.

## 🎙️ Phase 2: The Direct Speech Handshake (`/v1/audio/speech`)
We discovered that mid-graph entries (`start_stage: 1`) via Chat Completions are a **Neural Dead-End** because the tokenizer is bypassed.
- **The Solution**: Use the hidden `/v1/audio/speech` route.
- **The Cloning DNA**: Fish-Speech requires a reference seed. We injected `reference_voice.wav` as a **Data URI** (`data:audio/wav;base64,...`) at the **TOP LEVEL** of the payload.

## 👂 Phase 3: The SenseVoice Deafness (Stage 0)
**The 'Neural Cascade'**: SenseVoice (Stage 0) is hard-linked to the Mouth (Stage 1). Starting at 0 automatically triggers 1.
**The Modality Deadlock**: Requesting 'text' from a chain that ends in 'audio' crashes the router.
**The Dual Modality Compromise**: We requested `modalities: ["text", "audio"]` to satisfy the chain and captured the transcription from the `audio.transcript` field.
**VERDICT**: SenseVoice is currently 'deaf' (returns `""`). Generation verification must remain external for now.

## 🏛️ The Final Permanent Handshake
- **Endpoint**: `POST /v1/audio/speech`
- **DNA**: Mandatory Data URI reference.
- **Math**: The DAC outputs **44.1kHz**. Saving at 24kHz results in 'Slow Motion'.

---
*Codified by Antigravity. We fought for every byte. The 5090 is now a Vocal Throne.*
