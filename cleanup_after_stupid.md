# 🗑️ THE STUPIDBOT CLEANUP LIST (Legacy Debt)

> [!IMPORTANT]
> **GATING PROTOCOL**: DO NOT delete any files listed here without an explicit request for review and approval from the operator. These files represent legacy state that must be fully replicated in the Recursive ETL experts before being purged.

## 🏚️ Candidates for Removal

These files have been superseded by the **Master Blueprint** and **Expert Registry** system.

| File / Directory | Status | Why it's "Stupid" to keep |
| :--- | :--- | :--- |
| `src/ai/interrogator.py` | 🏚️ LEGACY | Old diagnostic sweep script. Superseded by the **Testing Citadel** and `verify_output.py`. |
| `src/ai/providers/moshi.py` | 🏚️ LEGACY | Legacy Moshi bridge. Superseded by the `moshi` expert registration. |
| `src/ai/providers/glm/` | 🏚️ LEGACY | The monolithic legacy GLM implementation. Superseded by `GLMExpert` and the recursive ETL pipeline. |
| `src/scratch/` | 🧪 SCRATCH | Contains temporary scripts used during infrastructure stabilization. |

## 🛑 The "Wait-and-See" List

Do NOT delete these until Phase 3 (Native Orchestrator Integration) is 100% stable.

| File / Directory | Dependency | Remaining Work |
| :--- | :--- | :--- |
| `src/ai/orchestrator.py` | `src/voice/cog.py` | Conversation history management needs to be ported to a `checkpoint` expert. |
| `src/voice/resampler.py` | Legacy Voice Logic | High-level upsampling utilities still used by some voice components. |

---
*Authored by Antigravity (Gemini 3 Flash - Ultimate Edition)*
