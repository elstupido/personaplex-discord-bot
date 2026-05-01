# Blackwell (RTX 5090) Optimization Roadmap

This document tracks planned and implemented optimizations for the PersonaPlex Discord bot running on Blackwell architecture.

## Phase 1: Stability & Baseline (DONE/IN-PROGRESS)
- [x] **Native Wheel Builds**: Compiled `TransformerEngine` and `torchaudio` with `sm_120a` support.
- [x] **Memory Management**: Implemented `low_cpu_mem_usage=True` and `4-bit` quantization to fit within WSL2 constraints.
- [x] **Flash Attention 2**: Enabled via `attn_implementation="flash_attention_2"` to leverage TE kernels.
- [x] **VRAM Stability**: Set `expandable_segments:True` in `PYTORCH_CUDA_ALLOC_CONF`.

## Phase 2: High Performance (NEXT STEPS)
- [ ] **FP8 Autocast & Scaling**: 
    - Use `transformer_engine.pytorch.fp8_autocast(enabled=True)`.
    - Implement `te.pytorch.FP8GlobalScale` to maintain precision during long audio-heavy turns.
- [ ] **Fused Operators**:
    - Replace standard `LayerNorm` with `te.LayerNorm` (already optimized for Blackwell).
- [ ] **Async Token Decoding**:
    - Reduce `buffer_size` in `glm_server.py` from 25 to 12. 
    - *Rationale*: The 5090 is fast enough to generate audio chunks at half the current latency without underruns.

## Hardware-Specific Tuning (RTX 5090 + 15GB RAM)
- [ ] **Pinned Memory (Host-to-Device)**:
    - In `StreamingSink`, use `audio_tensor.pin_memory()` before sending to the GLM worker.
    - *Rationale*: Speeds up the transfer from the CPU-bound Discord socket to the GPU.
- [ ] **WSL2 Hugepages**:
    - Configure `/etc/sysctl.conf` for `vm.nr_hugepages`.
    - *Rationale*: Prevents the Linux kernel from paging out model weights during silence, eliminating "wake-up" lag.
- [ ] **Process Affinity**:
    - Use `taskset` to pin the GLM-Server to specific high-performance CPU cores (P-cores).

## Phase 3: Ultra-Low Latency & Scalability
- [ ] **TensorRT-LLM 0.15+**:
    - Target the native Blackwell backend (once officially released in TRT-LLM).
- [ ] **Multi-User KV Management**:
    - Implement PagedAttention (vLLM style) if handling more than 5 simultaneous users.
    - *Rationale*: Maximize the 32GB VRAM by sharing common persona prompts.

---
*Note: Always monitor `nvidia-smi -l 1` during multi-user sessions to check for power-limit throttling.*

---
*Note: Ensure baseline stability in `glm-4` mode before moving to Phase 2.*
