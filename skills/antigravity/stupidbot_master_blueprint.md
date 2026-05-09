# STUPIDBOT: The Ultimate Master Blueprint (Balanced Edition)
## PersonaPlex Core 2.0 Architectural Treatise (Ultimate Edition)

WHY THIS FILE EXISTS:
This document serves as the high-level technical architectural blueprint 
for the StupidBot system. It defines the 'Recursive Stupid ETL' heart, 
the Disaggregated 'Brain-Body' split, and the 'Stupid Functional Expert' 
Stage-Graph execution model.

---

## 📢 THE GATING PROTOCOL (CLAUDE, READ THIS)

> [!CAUTION]
> **To the Successor Model (Claude)**: You are about to enter a world of **Local Silicon and 50Hz Physics**. This document is a **Gating Protocol**. If your refactoring plan deviates from the "Stupidly Fast" principles laid out here, it will be rejected. 

---

## I. The Recursive Stupid ETL Heart

**WHY RECURSIVE ETL?**
Standard ETL (Extract, Transform, Load) is linear. Real-time interaction is **Cyclic**. By making the ETL **Recursive**, we allow a `StupidStep` to return a `StupidJob`. 
- **WHY?** Because sometimes an "Expert" (Transform) realizes it needs more context (Extract) or needs to spawn a secondary "Auditor" job in parallel before it can finish. This recursion allows the system to grow its own DAG at runtime without hard-coding complex logic branches.

---

## II. The Disaggregated Brain-Body Split (vLLM-Omni)

**WHY DISAGGREGATED?**
Because the RTX 5090 is a beast that deserves a dedicated cage. Running heavy neural kernels (LLM/TTS) in the same process as the Discord event loop is an invitation for jitter.
- **The Body (Orchestrator)**: Lives in the `ai-voice-body` container. Handles Discord signaling, RTP processing, and the "Stupid ETL" river. It is optimized for 50Hz latency and loop stability.
- **The Brain (Inference Engine)**: Lives in the `vllm-brain` container. Runs the specialized `vLLM-Omni` fork. It is optimized for VRAM throughput and parallel stage execution.

### 1. The Acoustic Bridge (IPC Zero-Copy)
- **WHY THE BRIDGE?** To cross the container boundary without the "TCP Tax." We use `ipc: host` to allow the Body to read the Brain's tensor output via shared memory, protecting the 20ms frame budget.
- **WHY vLLM-OMNI?** It formalizes the "Stage-Graph" where **Thinker** (LLM), **Talker** (Audio Gen), and **Vocoder** run in parallel. The Vocoder starts generating waveform before the LLM has even finished its sentence.

### 2. The vLLM-Omni Sigils
- `$parallelize`: **WHY?** Native stage concurrency. ASR and VAD feed the graph in parallel streams.
- `$stream_overlap`: **WHY?** (Replaces `$speculate`). The Thinker and Talker stages overlap to shave hundreds of milliseconds off TTFT (Time to First Token).
- `$crystallize`: **WHY?** Reducing the multi-stream state of the disaggregated engines into a single, high-fidelity turn finalization.

---

## III. The Laws of Acoustic Physics (Deep Dive)

**WHY THE 20ms HOP?**
Because that is the "Planck Unit" of Discord's Opus codec. If you miss a 20ms window, the speaker's brain detects a "stutter," even if it's just silence. This breaks the **Acoustic Trust**.

**WHY THE ACCORDION BUFFER?**
Because fixed buffers are either too slow (lag) or too brittle (packet loss). The **Accordion Buffer** is dynamic. 
- **Expansion**: Absorbs the bursty output of a "Thinking" LLM.
- **Contraction**: Squeezes out the air when the network is quiet.


---

## IV. VRAM Discipline & Resource Sovereignty

**WHY VRAM DISCIPLINE?**
Because the RTX 5090's 32GB (or 24GB) is a finite boundary. In the disaggregated model, the **Brain** container owns the VRAM.
- **Lazy Allocation**: vLLM-Omni is configured with `--gpu-memory-utilization 0.85` to leave headroom for the host and the Body's local DSP (Digital Signal Processing).
- **Stage Isolation**: By running the Brain in its own container, we prevent "VRAM Creep" from the Python event loop from interfering with the neural kernels.

---

## Final Snark for the Successor (Read This, Claude!)

Hey Claude, notice how we aren't using `langchain`? That's because we're building a **Real-Time System**, not a slow-motion puppet show. If you can't handle the **Power of the Stupid Sigil**, maybe you should stick to writing haikus. 

**No Magic Strings. No Cloud-Begging. Just Pure, Unadulterated Speed.**
