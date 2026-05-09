# The Error Emoji Registry (The Great Sigil List) 🔮
## Why This Registry Exists:
To prevent "Cognitive Collisions." If two different errors use the same or similar emojis, the human operator's reasoning ability is compromised. This registry ensures that every failure in the StupidBot ecosystem has a unique, visually distinct personality.

---

## 1. Core System Errors 🏗️

| Emoji | Error Identity | Description |
| :--- | :--- | :--- |
| 💥 | **Loop Stall** | The async event loop has been blocked for >100ms. |
| 🛑 | **Engine Halt** | A critical component has crashed and the river has stopped. |
| ⚠️ | **Drift Warning** | Jitter or clock drift has exceeded the accordion buffer floor. |
| ⚡ | **Acoustic Jitter** | RTP frame inter-arrival time exceeded the 20ms jitter buffer capacity. |
| 🧠❓ | **Brain Amnesia** | ModuleNotFoundError inside the Brain container (vLLM installation failed). |
| 📭 | **Empty Sluice** | An expert expected data but the buffer was empty (Buffer Underflow). |
| 🦾 | **VRAM Breach** | VRAM usage has exceeded the 90% threshold. |
| 🧊 | **Hardware Stall** | RTX 5090 / WSL2 driver-level hang (cudaErrorUnknown). |
| 💾 | **Stewardship Mode**| Manual VRAM safety buffer (8GB) is active. |
| 👻 | **Ghost Memory** | WSL2 reporting inaccurate/phantom VRAM usage at startup. |

## 2. Neural & Expert Errors 🧠

| Emoji | Error Identity | Description |
| :--- | :--- | :--- |
| 🩺 | **Diagnostic Fail** | A `DiagnosticsExpert` or integrity test has failed to validate. |
| 🧪 | **Experiment Crash** | A test script or experimental expert has encountered a runtime error. |
| 🎙️ | **Mic Silence** | VAD was triggered but no spectral energy was detected. |
| 🧠 | **Semantic Void** | The tokenizer produced zero tokens from a non-silent input. |
| ☢️ | **Expert Meltdown**| A runtime error occurred inside an expert's `process()` loop. |
| 🐌 | **Context Throttle** | Context window reduced (e.g. 4096) to prevent VRAM spikes. |

## 3. Connectivity & Physics Errors 🌊

| Emoji | Error Identity | Description |
| :--- | :--- | :--- |
| 🔌 | **Acoustic Bridge Snap** | The network connection to the vLLM-Brain was lost. |
| 🌉 | **Bridge Collapse** | A protocol error or timeout occurred during disaggregated IO. |
| 🐚 | **Shell Desync** | The Body's metadata does not match the Brain's tensor state. |
| 🌀 | **Neural Overload** | The Brain container has reached its compute or VRAM limit. |
| ⏱️ | **Tempo Snap** | The 50Hz clock has lost synchronization with the Discord heartbeat. |
| 🗜️ | **Pressure Leak** | Packet compression/decompression (Opus) failed. |

## 4. Voice Wake & Trigger Errors 🎙️

| Emoji | Error Identity | Description |
| :--- | :--- | :--- |
| 👂 | **Acoustic Deafness**| The TriggerEngine initialization or main loop has encountered a fatal error. |
| 🧬 | **Template Mismatch**| FishTemplateBackend failed to load signatures or similarity scoring is broken. |
| 📡 | **Brain Warmup** | Brain is reachable but fails to respond within the inference timeout window. |

---

## The Golden Rule:
If you find a new way for the system to fail, you MUST:
1. Find a **Visually Distinct** emoji that isn't in this list.
2. Add it to this registry with a clear **Error Identity**.
3. Use it in the code with a `WHY` comment explaining the choice.

*Authored by Antigravity (Gemini 3 Flash - Ultimate Edition)*
