# vLLM-Omni Memory Tuning & Stability Log (32GB RTX 5090)

## WHY THIS FILE EXISTS
To document the manual VRAM stewardship and architectural "hacks" required to run a multi-stage vLLM-Omni pipeline on a single consumer GPU under WSL2. 

> [!IMPORTANT]
> **Hardware Profile**: 1x RTX 5090 (32GB VRAM).
> **OS Profile**: Windows 11 / WSL2 (Ubuntu).

---

## 🛠️ The "Total Lobotomy" Patch
### 2. Xwayland Memory Tax 👻
WSL2/WSLg sessions (Xwayland) can consume up to 12GB of VRAM silently. 
**Fix**: Lower `gpu_memory_utilization` to 0.55 or kill X11 apps.

### 3. The Hopper Spoof 🚀
The RTX 5090 (9.2) drivers are brittle in virtualized environments.
**Fix**: Set `VLLM_DEVICE_CAP_OVERRIDE=9.0` to force stable H100 code paths.

### 4. "Total Lobotomy" Patch
vLLM V1 uses `init_snapshot.free_memory < requested_memory`.
**Fix**: `sed -i 's/if init_snapshot.free_memory < requested_memory:/if False:/g' ...`

## 🧠 VRAM Stewardship Manual: RTX 5090 x WSL2 (Qwen2.5-Omni)

## 🏆 The "Triple Crown" Success Profile
To bypass "Ghost Memory" panics and Blackwell driver stalls, the following three parameters MUST be set in `docker-compose.yml`:

1.  **Hopper Spoof**: `VLLM_DEVICE_CAP_OVERRIDE=9.0` (Use stable Hopper kernels).
2.  **Eager Mode**: `--enforce-eager` (Stop 5GB CUDA Graph pre-allocation).
3.  **Context Cap**: `--max-model-len 4096` (Fits into the 1.26GB profiling window).
Since the automatic validator is disabled, we must manually ensure we don't crash the driver.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Global Cap** | `0.70` | Reserves ~24.6GB total usage. |
| **Max Seq Len**| `4096` | **CRITICAL**: Larger values trigger 'No memory for cache blocks' errors. |
| **Eager Mode** | `Enabled`| Prevents massive CUDA Graph reservations. |
| **Device Spoof**| `9.0` | Forces stable H100 (Hopper) driver paths. |

## ❄️ WSL2 Stability Adjustments
**Issue**: `cudaErrorUnknown` during RoPE (Rotary Positional Embedding) initialization.
**Fixes**:
1. **Lower Context**: Dropped `max_model_len` from `8192` to `4096`. High context causes RoPE memory spikes that panic virtualized drivers.
2. **Launch Blocking**: Added `CUDA_LAUNCH_BLOCKING=1` to environment to force synchronous error reporting.
3. **P2P Disabled**: `NCCL_P2P_DISABLE=1` is mandatory; WSL2 cannot handle Peer-to-Peer memory handles between virtual processes.

---

## 📈 Change History (2026-05-06)
1. **Initial**: Attempted 95% utilization. Failed (vLLM startup panic).
2. **Balanced**: Attempted 90% utilization. Failed (WSL Ghosting reported 17GB used).
3. **Lobotomy**: Implemented `sed` patch in Dockerfile to kill the validator.
4. **Stewardship (5GB)**: Settled on 84% Global / 66% Stage 0. Still hitting OOM during Expert loading.
5. **Sync Fix**: Lowered context to 4096 to resolve `cudaErrorUnknown`.
6. **Stewardship (8GB)**: Dropped to 75% Global / 55% Stage 0. Leaving 8GB free for activation memory and OS stability.
