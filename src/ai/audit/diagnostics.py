"""
WHY THIS FILE EXISTS:
The 'DiagnosticsExpert' (The Stethoscope). 🩺

WHY:
Before we trust the RTX 5090 with heavy tensors, we need to prove that the 
conveyor belt works. This expert is a 'No-Op' transformation that logs 
the AcousticContext to ensure the 50Hz timing and TraceID are preserved.
"""

import time
from typing import Generator
from ..stupid_base import StupidStep, StupidData, StupidRegistry, logger

@StupidRegistry.register("diagnostics")
class DiagnosticsExpert(StupidStep):
    """
    A low-latency diagnostic step for pipeline verification. 🧪
    """
    def __init__(self, name: str):
        super().__init__(name)
        logger.info(f"✨ [DiagnosticsExpert] '{name}' initialized and ready to audit the river.")

    async def process(self, data: StupidData) -> Generator[StupidData, None, None]:
        """
        Audit the data and yield it unchanged. 🔍
        
        WHY:
        We don't want to copy the data (Zero-Copy!), but we do want to 
        calculate the 'River Latency'—the time from arrival at the bot 
        to processing by this expert.
        """
        ctx = data.context
        latency_ms = (time.time() - ctx.arrival_time) * 1000
        
        # Log the health of the particle 🩺
        logger.debug(
            f"🧪 [Diagnostics] Trace: {ctx.trace_id} | "
            f"Type: {data.type} | "
            f"River Latency: {latency_ms:.2f}ms | "
            f"SR: {ctx.sample_rate}Hz"
        )
        
        # Keep the river flowing 🌊
        yield data
