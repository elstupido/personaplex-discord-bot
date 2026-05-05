# StupidBot 2.0: The 'Balanced Stupid' Implementation Plan
## Refactoring the Monolith into a Dual-Container Disaggregated System

WHY THIS FILE EXISTS:
This plan outlines the specific technical steps to transition the legacy 
PersonaPlex bot into the 'StupidBot' architecture, now split into a 
**Body** (Discord Bot) and a **Brain** (vLLM-Omni).

---

## 📢 THE GATING PROTOCOL (CLAUDE, READ THIS)

> [!CAUTION]
> This plan is governed by the **Laws of Acoustic Physics**. We move audio at 50Hz. We use zero-copy pointers. We run locally on the RTX 5090. If your refactoring plan deviates from the "Stupidly Fast" principles laid out here, it will be rejected. 

---

## I. The 'Stupid' Primitives

**WHY THE ATOMIC REFACTOR?**
Because you can't build a skyscraper on a swamp. The current `BridgeBase` is too rigid. We need to establish the **Protocol** (`StupidData`, `StupidStep`) before we can let the **Experts** onto the field.

---

## II. The Phased Roadmap (The "Why")

### Phase 1: Atomic Foundation
**WHY?** To create a common language. If every component speaks `StupidData`, we can chain them in any order. This is the **Infrastructure Phase**.

### Phase 2: The vLLM-Omni Takeover
**WHY?** To offload the heavy lifting to a specialized, disaggregated engine. Instead of wrapping models in custom Python scripts and fighting the GIL, we serve them via vLLM-Omni. This gives us **Streaming Overlap** (Vocoder starts before LLM finishes) and **Shared Memory Tensors** out of the box.

### Phase 3: Rewiring the Cog (Disaggregated Orchestration)
**WHY?** To move the "Brain" from a monolithic loop to a vLLM-Omni gRPC/SharedMemory client. This is where we retire the `GLMVoiceEngine` and use the **vLLM Graph** to handle the Thinker/Talker/Vocoder split. We maintain the `StupidRunner` only as a high-level router for non-model tasks.

### Phase 4: The TDD Citadel
**WHY?** Because in real-time systems, "It works on my machine" is a lie. We need automated stress tests for jitter and drift to prove that the **Acoustic Trust** (and vLLM's low-latency promise) is maintained on the RTX 5090.

---

## III. The Docker Architecture (The Body and the Brain)

**WHY TWO CONTAINERS?**
Because the "Local Silicon" environment is complex. We have CUDA, specialized drivers, and multi-gigabyte models. 
- **The Body (Bot)**: Lightweight, focused on Discord I/O and orchestration.
- **The Brain (vLLM-Omni)**: Heavyweight, focused on neural inference and VRAM management.

**WHY IPC SHARED MEMORY?**
Because copying 48kHz audio between containers via TCP is too slow. `IPC Shared Memory` is our high-speed bridge. Both containers will share the host's IPC namespace (`--ipc=host`) to allow zero-copy tensor movement.

---

## IV. vLLM-Omni Convergence (The New North Star)

**WHY vLLM-OMNI?**
Because it formalizes our "Recursive ETL" vision. By using their **Disaggregated Stage Execution**, we can run SenseVoice (ASR), GLM-4 (Thinker), and Fish-Speech (Vocoder) in parallel streams. 
- **Zero-Copy**: We leverage vLLM's internal memory management to avoid the `bytearray` tax.
- **Async First**: No blocking calls. The 20ms frame is protected by vLLM's own high-priority scheduling.

---

## Final Snark for the Successor

Hey Claude, don't forget: we're in a Docker container on an RTX 5090. If you try to call an external API, the firewall (and the operator) will have your head. **No Magic Strings. Just Silicon, Docker, and Speed.**
