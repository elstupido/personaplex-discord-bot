"""
WHY THIS FILE EXISTS:
The Testing Citadel (Master Orchestrator). 🏰🧪

WHY:
In a complex ETL system, we cannot rely on manual script execution. 
The Citadel provides a single, unified entry point for all tests, 
generating a report that proves the system's integrity across the 
Acoustic, Neural, and Structural layers.
"""

import subprocess
import sys
import time

class TestCitadel:
    def __init__(self):
        self.results = []
        self.start_time = time.time()

    def run_test(self, name, command, description):
        print(f"🚀 Running: {name}...")
        print(f"   {description}")
        
        try:
            # Inject diagnostic flags for tests
            # WHY: We force DEBUG logs and enable the STUPID_DIAGNOSTICS tracer 
            # so we can see exactly where the river stalls during a failure.
            env = {
                **subprocess.os.environ,
                "LOG_LEVEL": "DEBUG",
                "STUPID_DIAGNOSTICS": "1",
                "PYTHONASYNCIODEBUG": "1",
                "PYTHONUNBUFFERED": "1"
            }
            
            # Use Popen to stream output in real-time
            # WHY: capture_output=True hides logs until the test finishes.
            # Real-time streaming is essential for debugging 'stalls' and 
            # seeing the [METRIC] logs as they happen.
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                bufsize=1, # Line buffered
                universal_newlines=True
            )

            all_stdout = []
            all_stderr = []

            # Poll for output
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    stripped = line.strip()
                    print(f"   [LIVE] {stripped}")
                    all_stdout.append(line)

            # Capture remaining stderr
            err_output = process.stderr.read()
            if err_output:
                print(f"   [ERR] {err_output.strip()}")
                all_stderr.append(err_output)

            success = process.returncode == 0
            full_stdout = "".join(all_stdout)
            
            self.results.append({
                "name": name,
                "success": success,
                "output": full_stdout,
                "error": "".join(all_stderr)
            })
            
            if success:
                print(f"✅ {name} PASSED\n")
            else:
                print(f"💥 {name} FAILED\n")
                
        except Exception as e:
            print(f"🛑 Error executing {name}: {e}\n")
            self.results.append({
                "name": name,
                "success": False,
                "output": "",
                "error": str(e)
            })

    def print_report(self):
        total_time = time.time() - self.start_time
        passed = sum(1 for r in self.results if r["success"])
        total = len(self.results)
        
        print("\n" + "="*60)
        print("🏛️  THE TESTING CITADEL: STUPIDBOT INTEGRITY REPORT")
        print("="*60)
        print(f"Duration: {total_time:.2f}s")
        print(f"Score:    {passed}/{total} Tests Passed")
        print("-" * 60)
        
        for r in self.results:
            status = "✅ PASSED" if r["success"] else "💥 FAILED"
            print(f"{status} | {r['name']}")
        
        print("="*60)
        
        if passed == total:
            print("✨ THE RIVER IS CLEAR. ALL SIGILS VALIDATED. ✨")
        else:
            print("🛑 THE DAM IS LEAKING. REVIEW FAILURES ABOVE. 🛑")
        print("="*60 + "\n")

if __name__ == "__main__":
    citadel = TestCitadel()
    
    citadel.run_test(
        "Atomic Foundation",
        f"{sys.executable} tests/verify_foundation.py",
        "Validates StupidData, StupidRunner, and basic ETL flow."
    )
    
    citadel.run_test(
        "Sigil Registry",
        f"{sys.executable} tests/verify_config.py",
        "Validates StupidConfig, Blueprints, and dynamic Factory loading."
    )

    citadel.run_test(
        "Acoustic Integrity",
        f"{sys.executable} tests/verify_output.py",
        "Validates AudioSource buffering and Resampler upsampling."
    )
    
    citadel.run_test(
        "Heartbeat Integrity",
        f"{sys.executable} tests/verify_heartbeat.py",
        "Validates event loop responsiveness during heavy expert loading."
    )
    
    # 5. Unit Tests (Legacy/Component layer)
    # citadel.run_test(
    #     "Component Units",
    #     "python3 tests/test_audio_pipeline.py",
    #     "Validates resamplers, tokenizers, and decoders in isolation."
    # )
    
    citadel.print_report()
    
    # Exit with code 1 if any tests failed
    if any(not r["success"] for r in citadel.results):
        sys.exit(1)
    sys.exit(0)
