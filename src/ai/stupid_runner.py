"""
WHY THIS FILE EXISTS:
The StupidRunner is the 'Hydroelectric Dam Operator' of the system. 
It takes a StupidJob (the Recipe) and executes it against the data river.

WHY ASYNCIO?
Because we are orchestrating multiple models that might be running 
in different containers or on different CUDA streams. We need 
non-blocking execution to maintain the 50Hz Loop.
"""

import asyncio
import torch
import time
import os
from typing import List, Any, Generator, Dict
from core.logger import setup_logger
from .stupid_base import StupidData, StupidStep, StupidJob, StupidRegistry

logger = setup_logger("ai.stupid_runner")

class VRAMGuard:
    """
    The Silicon Stewardship Protocol.
    
    WHY THIS EXISTS:
    The RTX 5090 is a finite resource. If we load every expert into VRAM, 
    the system will OOM and die. The VRAMGuard ensures we stay below 
    the 90% threshold.
    """
    def __init__(self, threshold=0.90):
        self.threshold = threshold

    async def monitor(self, active_experts: Dict[str, Dict[str, Any]]):
        if not torch.cuda.is_available(): return
        
        total = torch.cuda.get_device_properties(0).total_memory
        reserved = torch.cuda.memory_reserved(0)
        usage = reserved / total
        
        if usage > self.threshold:
            logger.warning(f"⚠️ [VRAMGuard] VRAM Critical ({usage*100:.1f}%). Triggering Eviction...")
            await self.evict_coldest(active_experts)

    async def evict_coldest(self, active_experts: Dict[str, Dict[str, Any]]):
        # Sort by last_used timestamp
        sorted_experts = sorted(active_experts.items(), key=lambda x: x[1]['last_used'])
        
        for name, data in sorted_experts:
            logger.info(f"✨ [VRAMGuard] Evicting Cold Expert: {name}")
            del data['instance']
            del active_experts[name]
            await asyncio.to_thread(torch.cuda.empty_cache)
            
            # Check if we've cleared enough
            if torch.cuda.memory_reserved(0) / torch.cuda.get_device_properties(0).total_memory < self.threshold:
                break

class StupidRunner:
    """
    The Executor of the Recursive ETL Pipeline.
    
    WHY THIS ARCHITECTURE?
    By separating the 'What' (StupidJob) from the 'How' (StupidRunner), 
    we allow for complex execution patterns (Parallelism, Speculation) 
    without polluting the Expert logic.
    """
    def __init__(self):
        self.active_experts: Dict[str, Dict[str, Any]] = {}
        self.vram_guard = VRAMGuard()

    async def warmup(self, steps: List[str]):
        """
        Eagerly load and warmup all experts in the blueprint.
        
        WHY THREADED?
        Initial instantiation of AI experts (especially Tokenizers) 
        involves massive synchronous imports (transformers, torch). 
        By using to_thread, we keep the Discord heartbeat alive during load.
        """
        for step_id in steps:
            if isinstance(step_id, str) and not step_id.startswith("$"):
                if step_id not in self.active_experts:
                    expert_cls = StupidRegistry.get_expert(step_id)
                    logger.info(f"🔥 [StupidRunner] Eagerly Warming Expert: {step_id}")
                    
                    warmup_start = time.time()
                    # Instantiate in thread to avoid blocking the loop with imports
                    instance = await asyncio.to_thread(expert_cls, step_id)
                    warmup_duration = (time.time() - warmup_start) * 1000
                    
                    logger.debug(f"[METRIC] op=expert_warmup expert={step_id} duration_ms={warmup_duration:.2f}")
                    
                    self.active_experts[step_id] = {
                        'instance': instance,
                        'last_used': time.time()
                    }
                    
                    # Optional: Trigger expert-specific warmup logic
                    if hasattr(instance, "warmup"):
                        if asyncio.iscoroutinefunction(instance.warmup):
                            await instance.warmup()
                        else:
                            await asyncio.to_thread(instance.warmup)

    async def execute_job(self, job: StupidJob):
        """
        Execute a StupidJob recursively.
        
        WHY RECURSIVE?
        Because a step might yield a new job. This allows the pipeline 
        to 'Branch' mid-stream based on the content of the audio.
        """
        current_data = [job.data]
        
        for step_id in job.steps:
            if isinstance(step_id, StupidJob):
                # Recurse!
                current_data = await self.execute_job(step_id)
                continue
            
            # Handle Sigils
            if step_id.startswith("$"):
                await self._handle_sigil(step_id, job)
                continue

            # Standard Step Execution
            next_data = []
            expert_cls = StupidRegistry.get_expert(step_id)
            
            # Lazy Load Expert with VRAM Stewardship
            if step_id not in self.active_experts:
                await self.vram_guard.monitor(self.active_experts)
                logger.info(f"✨ [StupidRunner] Lazy Loading Expert (Emergency): {step_id}")
                instance = await asyncio.to_thread(expert_cls, step_id)
                self.active_experts[step_id] = {
                    'instance': instance,
                    'last_used': time.time()
                }
            
            expert_data = self.active_experts[step_id]
            expert_data['last_used'] = time.time()
            expert = expert_data['instance']
            
            # --- STUPID DIAGNOSTICS ---
            # WHY: If enabled, we trace the exact particle flow and timing 
            # between experts to find bottlenecks in the river.
            diagnostic_mode = os.getenv("STUPID_DIAGNOSTICS") == "1"
            
            for d in current_data:
                step_start = time.time()
                try:
                    async for result in expert.process(d):
                        if diagnostic_mode:
                            duration = (time.time() - step_start) * 1000
                            logger.debug(
                                f"[METRIC] op=step expert={step_id} "
                                f"trace_id={d.context.trace_id} "
                                f"duration_ms={duration:.2f} "
                                f"result_type={getattr(result, 'type', 'unknown')}"
                            )
                        
                        if isinstance(result, StupidJob):
                            # The Expert spawned a sub-job! Execute it and merge results.
                            sub_results = await self.execute_job(result)
                            next_data.extend(sub_results)
                        else:
                            next_data.append(result)
                except Exception as e:
                    # WHY: ☢️ Expert Meltdown.
                    # We catch all expert-level errors here to prevent a single 
                    # malfunctioning model from killing the whole bot.
                    logger.error(f"☢️ [StupidRunner] Expert Meltdown in '{step_id}': {e}")
                    # In a linear pipeline, we stop this branch, but the core lives.
                    continue
            
            current_data = next_data
            
        return current_data

    async def _handle_sigil(self, sigil: str, job: StupidJob):
        """
        The Sigil Dispatcher.
        
        WHY SIGILS?
        Because Perl was right. Declarative symbols for complex execution 
        patterns keep the 'Recipe' readable.
        """
        if sigil == "$parallelize":
            # Implementation for parallel execution of next steps
            # (Currently a placeholder for Phase 3)
            logger.debug("[StupidRunner] Sigil $parallelize detected. Spawning ghost tasks...")
        elif sigil == "$speculate":
            # Implementation for speculative execution
            logger.debug("[StupidRunner] Sigil $speculate detected. Looking into the future...")
        else:
            logger.warning(f"⚠️ [StupidRunner] Unknown Sigil: {sigil}")
