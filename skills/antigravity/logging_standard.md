# Skill: Emoji-First Logging Standard 📜🎭

WHY THIS FILE EXISTS:
In a high-performance system like StupidBot, log visibility is survival. 
Standard text-based error codes are dry and easily missed in a scrolling 50Hz 
data river. By using expressive emojis, we increase the 'Visual Signal-to-Noise 
Ratio', allowing the operator to immediately distinguish between healthy flow 
and catastrophic failure.

---

## 1. The Emoji Registry (The Sigil Protocol) 🔮

Every log message or user-facing notification must lead with the appropriate 
Appropriate emoji to define its semantic context. We reject 6-digit numeric error codes in favor of these visual sigils to increase working memory and reasoning ability during failures.

### The Law of Unique Sigils
- **Non-Redundancy**: Every identifiable error must have its own unique Emoji.
- **Visual Distinction**: Emojis must be visually distinct; simple palette swaps or minor variations are prohibited to ensure immediate recognition.
- **Registry Mandate**: All error emojis must be recorded in the `error_emoji_registry.md` to prevent collisions.

| Severity | Emoji | Usage |
| :--- | :--- | :--- |
| **CRITICAL / ERROR** | 💥 | Catastrophic failure, engine stall, or VRAM overflow. |
| **FATAL / HALT** | 🛑 | The river has stopped. Immediate intervention required. |
| **WARNING** | ⚠️ | Non-fatal drift, jitter, or lazy-load delays. |
| **INFO (General)** | ✨ | System initialization, model warmup, or state changes. |
| **INFO (Flow)** | 🌊 | Successful movement of an 'Atomic Particle' through the river. |
| **SUCCESS** | ✅ | Task completion, bridge connection, or verification pass. |
| **NEURAL / AI** | 🤖 | Inference starts, brain activation, or token generation. |
| **ACOUSTIC** | 🎙️ | Microphone activity, VAD trigger, or audio playback. |

---

## 2. The "No Magic Strings" Rule 🚫

Avoid using bare words like `[ERROR]` or `[FAILED]`. 
- **Bad**: `logger.error("[FAILED] Model not found")`
- **Good**: `logger.error("🛑 Model not found. The river is dry.")`

## 3. User-Facing Notifications 📢

When communicating with the user via Discord, always use the 💥 or 🛑 emojis for 
errors to maintain consistency with the internal logs. This allows the user to easily remember and describe the error (e.g., "the shrug emoji error"), leading to faster diagnosis.

## 4. The AI-First [METRIC] Standard 🤖📊

While emojis are for humans, structured data is for the AI assistant. 

### Rule: Use [METRIC] for Structured Performance Data
Critical performance metrics (latency, HTI, VRAM) MUST use the `[METRIC]` prefix with key-value pairs for easy parsing by AI models.

- **Format**: `[METRIC] op={operation} key1={val1} key2={val2}`
- **Example**: `[METRIC] op=step expert=whisper_tokenizer duration_ms=12.5`
- **Why**: This prevents the AI from being distracted by variable log phrasing and allows it to perform deterministic audits of the data river.

---

*Authored by Antigravity (Gemini 3 Flash - Ultimate Edition)*
