"""
WHY THIS FILE EXISTS:
The Foundation Verification Harness (The Core Integrity Suite). 🏗️🧪

WHY:
In the age of local silicon and recursive ETL, 'It builds' is a joke. 
This script proves that the ENTIRE core framework—Registry, Config, 
Runner, Factory, and VRAMGuard—works together as a single, coherent 
'Data River' orchestrator.

WE TEST:
1. Config & Blueprint Discovery 📜
2. Dynamic Expert Loading (Factory integration) 🔔
3. Non-blocking Eager Warmup 💓
4. VRAM Stewardship (Eviction logic) 🦾
5. Recursive Job Splicing (Data flow integrity) 🌊
6. Full Acoustic Pipeline (The Hot Path) 🎙️
"""

import asyncio
import time
import sys
import os
import json
import base64
import importlib
import torch
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from ai.stupid_base import StupidData, AcousticContext, StupidJob, StupidRegistry, logger
from ai.stupid_runner import StupidRunner
from ai.stupid_config import StupidConfig
from ai.stupid_factory import create_bridge

async def test_config_and_blueprints():
    logger.info("📡 [CoreTest] Phase 1: Validating Config & Blueprints...")
    
    blueprint = StupidConfig.get_active_blueprint()
    if not blueprint or not blueprint.steps:
        logger.error("💥 Config Failure: Active blueprint is empty or missing!")
        return False
        
    logger.info(f"✅ Blueprint Discovery: {blueprint.name} has {len(blueprint.steps)} steps.")
    
    # Ensure diagnostics blueprint exists
    diag_bp = StupidConfig.BLUEPRINTS.get("diagnostics")
    if not diag_bp:
        logger.error("💥 Config Failure: 'diagnostics' blueprint missing from StupidConfig!")
        return False
    
    return True

async def test_dynamic_loading():
    logger.info("📡 [CoreTest] Phase 2: Validating Dynamic Expert Loading...")
    
    # Use StupidConfig to find the diagnostics module
    module_path = StupidConfig.EXPERT_MODULES.get("diagnostics")
    if not module_path:
        logger.error("💥 Registry Failure: 'diagnostics' expert not found in StupidConfig mapping!")
        return False
        
    logger.info(f"👉 Loading module: {module_path}")
    # Trigger registration via factory-like logic
    await asyncio.to_thread(importlib.import_module, module_path)
    
    try:
        expert_cls = StupidRegistry.get_expert("diagnostics")
        logger.info(f"✅ Dynamic Registration: Expert 'diagnostics' ({expert_cls.__name__}) is now in Registry.")
    except (ValueError, KeyError):
        logger.error("💥 Registry Failure: Expert 'diagnostics' failed to register!")
        return False
        
    return True

async def test_warmup_integrity():
    logger.info("📡 [CoreTest] Phase 3: Validating Non-blocking Warmup...")
    
    runner = StupidRunner()
    experts_to_warm = ["diagnostics"]
    
    # We want to ensure it doesn't crash and actually populates active_experts
    start_time = time.time()
    await runner.warmup(experts_to_warm)
    elapsed = (time.time() - start_time) * 1000
    
    if "diagnostics" not in runner.active_experts:
        logger.error("💥 Warmup Failure: 'diagnostics' expert not in runner's active pool!")
        return False
        
    logger.info(f"[METRIC] op=warmup expert=diagnostics duration_ms={elapsed:.2f}")
    return True

async def test_vram_guard_eviction():
    logger.info("📡 [CoreTest] Phase 4: Validating VRAM Stewardship (Eviction)...")
    
    runner = StupidRunner()
    
    # 1. Manually populate active experts with mocks
    expert_a = MagicMock()
    expert_b = MagicMock()
    
    runner.active_experts = {
        "expert_old": {"instance": expert_a, "last_used": time.time() - 100},
        "expert_new": {"instance": expert_b, "last_used": time.time()}
    }
    
    # 2. Mock CUDA properties to trigger OOM (95% usage)
    mock_props = MagicMock()
    mock_props.total_memory = 1000
    
    # We want memory_reserved to drop after empty_cache is called
    memory_state = {"reserved": 950}
    def mock_empty_cache():
        memory_state["reserved"] = 700 # Drops below threshold
        
    with patch('torch.cuda.is_available', return_value=True):
        with patch('torch.cuda.get_device_properties', return_value=mock_props):
            with patch('torch.cuda.memory_reserved', side_effect=lambda device=0: memory_state["reserved"]):
                with patch('asyncio.to_thread', side_effect=lambda f, *args: f(*args) if f == torch.cuda.empty_cache else None):
                    with patch('torch.cuda.empty_cache', side_effect=mock_empty_cache):
                        logger.info("⚠️ Simulating VRAM Pressure (95%)...")
                        await runner.vram_guard.monitor(runner.active_experts)
                    
    # 3. Verify 'expert_old' was evicted
    if "expert_old" in runner.active_experts:
        logger.error("💥 VRAMGuard Failure: Cold expert was not evicted!")
        return False
        
    if "expert_new" not in runner.active_experts:
        logger.error("💥 VRAMGuard Failure: Hot expert was wrongly evicted!")
        return False
        
    logger.info(f"[METRIC] op=eviction status=success usage_before=95 usage_after=70")
    return True

async def test_recursive_splicing():
    logger.info("📡 [CoreTest] Phase 5: Validating Recursive Job Splicing...")
    
    runner = StupidRunner()
    await runner.warmup(["diagnostics"])
    
    # Create a job with a nested job as a step
    ctx = AcousticContext(trace_id="RECURSION_TEST")
    data = StupidData(content="root", context=ctx, type="signal")
    
    sub_job = StupidJob(steps=["diagnostics"], data=StupidData(content="child", context=ctx, type="signal"))
    
    root_job = StupidJob(
        steps=["diagnostics", sub_job, "diagnostics"],
        data=data
    )
    
    results = await runner.execute_job(root_job)
    
    # Audit: We expect the results from the child job to be merged!
    # Steps: 
    # 1. Diag(root) -> yields [root]
    # 2. sub_job([child]) -> yields [child]
    # 3. Diag(child) -> yields [child]
    # Final data should be 'child' (since the last step took 'child' from sub_job)
    
    if not results or results[0].content != "child":
        logger.error(f"💥 Recursion Failure: Expected 'child', got '{results[0].content if results else 'None'}'")
        return False
        
    logger.info("✅ Recursive Splicing: Data flow integrity verified across branches. 🌊")
    return True

async def test_full_pipeline_mocked():
    logger.info("📡 [CoreTest] Phase 6: Validating Full Acoustic Pipeline (Mocked)...")
    
    # Set model to diagnostics for a clean run without real weights
    os.environ["MODEL_TYPE"] = "diagnostics"
    bridge = await create_bridge(voice_preset="VARM3", text_prompt="test")
    
    if not bridge:
        logger.error("💥 Factory Failure: Bridge not created!")
        return False
        
    # 3. Trigger Eager Warmup
    # WHY: We MUST call connect() here to prove that the Factory -> Bridge -> Runner
    # chain can eagerly load and warmup neural experts without stalling.
    await bridge.connect()
        
    # Check if BridgeAdapter adopted the config-driven blueprint
    if bridge.blueprint.name != "Maintenance Mode":
        logger.error(f"💥 Integration Failure: Bridge has wrong blueprint: {bridge.blueprint.name}")
        return False

    logger.info("✅ Full Pipeline: Factory and BridgeAdapter integration verified. 🚀")
    return True

async def verify_core_integrity():
    """Run the Ultimate Core Integrity Suite."""
    logger.info("============================================================")
    logger.info("🏛️  STUPIDBOT CORE INTEGRITY AUDIT")
    logger.info("============================================================")
    
    tests = [
        test_config_and_blueprints,
        test_dynamic_loading,
        test_warmup_integrity,
        test_vram_guard_eviction,
        test_recursive_splicing,
        test_full_pipeline_mocked
    ]
    
    passed = 0
    for t in tests:
        try:
            if await t():
                passed += 1
            else:
                logger.error(f"❌ {t.__name__} FAILED")
        except Exception as e:
            logger.error(f"💥 {t.__name__} CRASHED: {e}")
            import traceback
            traceback.print_exc()
            
    logger.info("============================================================")
    logger.info(f"Audit Results: {passed}/{len(tests)} Tests Passed")
    logger.info("============================================================")
    
    return passed == len(tests)

if __name__ == "__main__":
    import logging
    # Use basic logging if setup_logger fails or for clean stdout
    logging.basicConfig(level=logging.INFO, format='%(levelname)-8s | %(message)s')
    
    success = asyncio.run(verify_core_integrity())
    sys.exit(0 if success else 1)
