"""
WHY THIS FILE EXISTS:
The Status Emoji Registry. 

WHY:
In a complex, disaggregated system like PersonaPlex (WSL2 + RTX 5090 + multi-process pipes), 
standard text logs are insufficient for real-time monitoring. We need a shorthand that 
communicates the PHYSICAL state of the silicon and the VIRTUAL state of the memory.
"""

class StatusEmoji:
    # --- HARDWARE & VRAM ---
    STEWARDSHIP = "💾" # WHY: Running with a manual VRAM safety buffer (The 8GB Buffer).
    GHOST       = "👻" # WHY: WSL2 is reporting phantom memory usage that doesn't exist.
    STALL       = "🧊" # WHY: CUDA driver/kernel synchronization failure (cudaErrorUnknown).
    THROTTLE    = "🐌" # WHY: Context window reduced (e.g. 4096) to prevent memory pressure.
    BREACH      = "🦾" # WHY: VRAM usage exceeded 90%.
    HALT        = "🛑" # WHY: Engine halt.
    DETECTED    = "✨" # WHY: Wake-word or trigger detected.
    
    # --- NEURAL PIPELINE ---
    BRAIN_PULSE = "🧠" # WHY: Successful handshake with the disaggregated inference server.
    MELTDOWN    = "☢️" # WHY: An AI Expert has crashed or yielded a trash result.
    WAKING      = "🔥" # WHY: Model is being eagerly loaded into VRAM.
    
    # --- DISCORD & SESSION ---
    ONLINE      = "✅"
    OFFLINE     = "🛑"
    JOINED      = "🎙️"
    LEAVING     = "🖕"
    SWITCH      = "✨"
    
    @classmethod
    def get_stewardship_msg(cls):
        """Returns the current hardware stewardship status string."""
        return f"{cls.STEWARDSHIP} Stewardship Active (8GB Buffer) | {cls.THROTTLE} Context: 4096"
