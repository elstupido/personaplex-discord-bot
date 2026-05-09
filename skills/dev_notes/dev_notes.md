# PersonaPlex Dev Notes — Session 2026-04-27

## The Big Picture Problem

The bot was not transcribing voice correctly. The symptom: saying "this is a test message" three times produced
fragmented garbage ("you", "[silence]", etc.). This session was a battle to figure out why.

---

## Key Truths — Do Not Forget These

### 1. Discord Does VAD For Us — Period

Discord's client performs Voice Activity Detection. It only transmits UDP packets when the user is **actively speaking**.
Silence = no packets. This means:

- **We do NOT need an energy/RMS gate.** Every packet that arrives IS speech.
- **We do NOT need a voiced-ratio filter.** Every packet is voiced.
- **We DO need a silence timer** — but only to detect the END of a turn, because Discord sends no "stop talking" signal.
  When packets stop arriving for N seconds → user finished their turn → dispatch.

The silence timer is the ONLY thing we should be doing besides buffering.

---

### 2. The StreamingSink Architecture (What Actually Works)

Modeled directly on `C:\Users\marti\repos\transcribe-discord-bot\src\cogs\transcribe.py`.

```python
def write(self, data, user):          # NO type hints — discord.VoiceData doesn't exist in this Pycord branch
    pcm_bytes = data.pcm if hasattr(data, 'pcm') else ...
    # Discord spec: 48kHz, stereo interleaved int16, 3840 bytes per 20ms frame
    stereo = np.frombuffer(pcm_bytes, dtype=np.int16)
    mono = stereo[0::2].tobytes()     # LEFT CHANNEL ONLY — see Phase Cancellation below
    buffer[user_id].extend(mono)
    last_active[user_id] = now
```

Collector loop (background thread, sleeps 100ms):
```python
for uid where (now - last_active[uid]) > silence_threshold:
    dispatch(buffer[uid])
    buffer[uid].clear()
    del last_active[uid]
```

Bridge dispatch:
```python
self.loop.call_soon_threadsafe(self.bridge.audio_queue.put_nowait, segments)
# NOT self.bridge.queue_user_audio()  — that method does NOT exist on GLMBridge
```

---

### 3. Phase Cancellation — Why Left Channel Only

If you average L+R channels: `(left + right) // 2`...
...and the mic is phase-inverted (right = -left, common with Discord headsets), the average = 0. Silent audio.

Fix: **take only the left channel**: `stereo[0::2]`. This is what `transcribe-discord-bot` does.
The transcription-bot's comment: `# Use left channel only to prevent phase cancellation issues`.

---

### 4. DAVE / E2EE Support

We use Pycord's DAVE PR: `git+https://github.com/Pycord-Development/pycord@refs/pull/3159/head`
(Same as `transcribe-discord-bot`.)

Required for DAVE compatibility in the Sink:
```python
class StreamingSink(discord.sinks.Sink):
    def __init__(self, ...):
        super().__init__(...)
        self.__sink_listeners__ = []   # Required by Pycord DAVE PR PacketRouter
```

**Do NOT use `discord.VoiceData` as a type hint.** It is not exported in the top-level `discord` namespace
in this PR. Using it crashes the extension at load time with `AttributeError`.

---

### 5. `write()` Must Be Lean — No Blocking

The `write()` method is called on the Discord event loop thread (or a closely related thread).
**Any blocking call in write() causes packet loss.** This includes:
- Running Whisper/STT synchronously
- Heavy numpy operations
- Long sleeps

If you need STT for diagnostics, run it in a thread executor or a separate background thread.
The `_glm_collector_loop` is a background thread — that's the right place for heavier work.

---

### 6. Common Mistakes Made This Session

| Mistake | Consequence |
|---|---|
| Deafen check in write() (`if self.audio_source.is_playing`) | ~80% packet loss — most frames returned early |
| `discord.VoiceData` type hint | `AttributeError` crash at extension load |
| `self.bridge.queue_user_audio(segment)` | `AttributeError` — method doesn't exist; use `audio_queue.put_nowait` |
| Averaging L+R channels | Phase-cancelled to silence |
| Running Whisper synchronously in echo mode | Blocked event loop for 1-3s, missed voice packets, 80% packet loss |
| Energy/RMS filtering | Dropped valid speech, totally wrong approach for Discord |
| `_min_dispatch_bytes` floor | Dropped short utterances, violates "trust Discord's VAD" principle |

---

### 7. Logging Philosophy

**Write logs for an LLM to read.** A log line should answer:
1. **What was the system trying to do?** (INTENT)
2. **What did it observe?** (OUTCOME)
3. **Does the outcome match the intent?** (SIGNAL — only needed when it doesn't)

❌ Bad: `Packet received from user 12345`
✅ Good: `[Discord-Intake] INTENT: buffer this packet as voiced speech. OUTCOME: 3840B received from uid=12345. Geometry OK (20ms frame at 48kHz stereo).`

Failure paths must say what to investigate next.
Success paths confirm the contract was met.
The skill file is at: `C:\Users\marti\.gemini\antigravity\skills\logging_philosophy.md`

---

### 8. Duration Math (Mono, 48kHz, int16)

After left-channel extraction, stored data is **mono, 48kHz, int16**:
- Bytes per second = 48000 samples/sec × 2 bytes/sample = **96000 bytes/sec**
- Duration formula: `len(buffer_bytes) / (48000 * 2)`
- One 20ms Discord frame = 3840 bytes stereo → 1920 bytes mono → 0.02s

---

### 9. The Transcription-Bot Reference

**Path:** `C:\Users\marti\repos\transcribe-discord-bot\src\cogs\transcribe.py`

When our audio pipeline breaks, check here first. This bot:
- Works perfectly with the same Discord setup
- Uses the same Pycord DAVE PR
- Stores raw stereo (converts to mono at transcription time, not in write())
- Never calls `vc.play()` — receive-only

---

### 10. Current Status (End of Session)

- `StreamingSink` rewritten to match transcription-bot architecture ✅
- Phase cancellation fixed (left channel only) ✅
- `discord.VoiceData` type hint removed ✅
- Bridge dispatch uses `audio_queue.put_nowait` ✅
- `vc.play()` removed (WSL2 UDP congestion mitigation, re-add when bot speaks back) ✅
- `PROMPT_ECHO_MODE = True` in `src/bridge/glm.py` — safe diagnostic mode ✅
- Turns are still short (0.08s, 0.20s) — root cause not yet confirmed ⚠️

**Next step:** User has a new approach. Start fresh from that.
