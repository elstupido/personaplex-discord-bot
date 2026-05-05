"""
WHY THIS FILE EXISTS:
The Heartbeat Integrity Monitor (The Anti-Stall Test). 💓🏗️🧪

WHY:
In an async system, 'Blocking' is the ultimate sin. If we spend 5 seconds 
importing 'transformers' on the main thread, the Discord Heartbeat stops, 
and the bot is disconnected. This test ensures our experts and runners 
remain 'Stupidly Fast' (or correctly threaded) so the loop never stalls.
"""

import asyncio
import time
import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from ai.stupid_base import StupidRegistry, StupidData, AcousticContext, logger

async def heartbeat_monitor(interval=0.01):
    """A task that 'ticks' at a high frequency to monitor loop health."""
    ticks = 0
    last_time = time.time()
    max_stall = 0
    while True:
        ticks += 1
        now = time.time()
        delta = now - last_time
        
        # --- DEEP DIAGNOSTICS ---
        # WHY: In a 100Hz loop, any delta > 15ms is a sign of jitter.
        # We log these as [JITTER] to identify precisely when the 
        # main thread was denied its CPU slice.
        if delta > 0.015 and ticks > 1:
            logger.debug(f"⚠️ [JITTER] Tick {ticks} delayed by {delta*1000:.2f}ms")
            max_stall = max(max_stall, delta)
            
        last_time = now
        await asyncio.sleep(interval)
        yield ticks, max_stall

async def verify_heartbeat():
    logger.info("🚀 Starting Heartbeat Integrity Monitor...")
    
    # 0. Pre-flight Checks (The 'Context Warmup')
    # WHY: Initializing CUDA or loading FFmpeg for the first time can trigger 
    # a process-wide stall. We do this BEFORE starting the monitor to ensure 
    # we are measuring Expert loading, not Driver initialization.
    import torch
    if torch.cuda.is_available():
        logger.debug("🛰️  Pre-initializing CUDA context...")
        torch.cuda.init()
        # Force a tiny allocation to solidify the context
        torch.zeros(1, device="cuda")
    
    try:
        import torchaudio
        logger.debug("🛰️  Pre-loading torchaudio backends...")
        # Accessing backends often triggers FFmpeg loading
        torchaudio.list_audio_backends()
    except Exception:
        pass
        
    # 1. Start the 50Hz Heartbeat Monitor
    monitor_ticks = 0
    max_stall_observed = 0
    async def run_monitor():
        nonlocal monitor_ticks, max_stall_observed
        async for t, ms in heartbeat_monitor():
            monitor_ticks = t
            max_stall_observed = ms
            
    monitor_task = asyncio.create_task(run_monitor())
    
    # 2. Trigger the Expert Warmup (The 'Stress' operation)
    # WHY: Importing modules like 'ai.transform.tokenizer' triggers 'torch' imports,
    # which take ~300ms. If done on the main thread, the heartbeat stalls.
    def get_expert_cls():
        import importlib
        importlib.import_module("ai.transform.tokenizer")
        return StupidRegistry.get_expert("whisper_tokenizer")

    logger.info("🧪 Testing Loop Responsiveness during Expert Instantiation...")
    
    start_ticks = monitor_ticks
    start_time = time.time()
    
    # 3. Offload both import and instantiation
    expert_cls = await asyncio.to_thread(get_expert_cls)
    instance = await asyncio.to_thread(expert_cls, "test_heartbeat")
    
    # 4. Perform the Eager Warmup (Neural Pre-heating)
    # WHY: We MUST call warmup here to ensure lazy-loaded weights and CUDA 
    # kernels are ready BEFORE the first audio packet hits the river.
    if hasattr(instance, "warmup"):
        logger.info(f"🔥 Warming up {instance.name} machinery...")
        await asyncio.to_thread(instance.warmup)
    
    # Wait a tiny bit to allow the monitor to catch up if it was queued
    await asyncio.sleep(0.05)
    
    end_ticks = monitor_ticks
    total_time = time.time() - start_time
    
    # Calculate the 'Heartbeat Trust Index' (HTI)
    # Expected ticks = total_time / interval
    expected_ticks = total_time / 0.01
    actual_ticks = end_ticks - start_ticks
    hti = (actual_ticks / expected_ticks) * 100 if expected_ticks > 0 else 0
    
    logger.info(f"[METRIC] op=warmup hti={hti:.2f} ticks_actual={actual_ticks} ticks_expected={int(expected_ticks)} duration_ms={total_time*1000:.1f} max_stall_ms={max_stall_observed*1000:.1f}")
    
    if hti < 80:
        logger.error(f"💥 HEARTBEAT FAILURE: The loop stalled during warmup! HTI={hti:.1f}%")
        monitor_task.cancel()
        return False
        
    logger.info("✅ Loop remained responsive during Warmup.")

    # 3. Test the 'Hot Path' (The Real Test) ⚡
    # WHY: Warmup is great, but if the first 'process' call still 
    # triggers a lazy import or CUDA initialization, we still fail.
    # 1. Create a larger dummy PCM buffer (2 seconds of 16kHz mono)
    # WHY: We want to give the machinery something to actually chew on.
    dummy_pcm = np.zeros(32000, dtype=np.float32) 
    logger.info("🧪 Testing First-Packet Latency (The Hot Path)...")
    
    start_ticks_hot = monitor_ticks
    start_time_hot = time.time()
    
    # Simulate 100 packets (The 'Stress Burn')
    # WHY: A single 35ms packet is too short for a 100Hz monitor to 
    # produce a statistically valid HTI (sampling error dominates). 
    # By looping 100 times, we smooth out the jitter and prove the 
    # sustained responsiveness of the core.
    for i in range(100):
        if i % 10 == 0:
            logger.debug(f"   [HotPath] Processing packet {i}/100...")
        async for _ in instance.process(StupidData(content=dummy_pcm, context=AcousticContext(sample_rate=16000), type="pcm")):
            pass
        
    end_ticks_hot = monitor_ticks
    hot_path_time = time.time() - start_time_hot
    
    # Calculate Hot-Path Trust Index
    expected_hot_ticks = hot_path_time / 0.01
    actual_hot_ticks = end_ticks_hot - start_ticks_hot
    hot_hti = (actual_hot_ticks / expected_hot_ticks) * 100 if expected_hot_ticks > 0 else 100
    
    logger.info(f"[METRIC] op=hot_path hti={hot_hti:.2f} ticks_actual={actual_hot_ticks} ticks_expected={int(expected_hot_ticks)} duration_ms={hot_path_time*1000:.1f}")

    if hot_hti < 90:
        logger.error(f"💥 HOT-PATH STALL: The first packet blocked the loop! HTI={hot_hti:.1f}%")
        monitor_task.cancel()
        return False

    monitor_task.cancel()
    logger.info("\n🏆 Heartbeat & Hot-Path Integrity Verified. 💓✨")
    return True

if __name__ == "__main__":
    # Ensure logs are visible
    import logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        success = asyncio.run(verify_heartbeat())
        sys.exit(0 if success else 1)
    except Exception as e:
        import traceback
        print(f"💥 Test Crash:")
        traceback.print_exc()
        sys.exit(1)
