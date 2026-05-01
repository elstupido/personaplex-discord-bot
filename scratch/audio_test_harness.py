import sys
import os
import time
import multiprocessing
import base64
import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from utils.audio_engine import AudioEngine

def test_reconstruction():
    input_q = multiprocessing.Queue()
    output_q = multiprocessing.Queue()
    stop_event = multiprocessing.Event()
    
    engine = AudioEngine(input_q, output_q, stop_event)
    engine.start()
    
    print("--- Starting Reconstruction Test ---")
    
    ssrc = 1234
    # Simulate packets: (seq, data, arrival)
    # We'll use dummy data (all 0x01)
    def send_pkt(seq, ts, data=None):
        if data is None: data = b'\x01' * 20 # Dummy Opus
        input_q.put({
            'user_id': '1234',
            'pcm': data,
            'ssrc': ssrc,
            'timestamp': ts,
            'arrival': time.time()
        })

    # 1. First packet with SSRC 0
    print("Sending packet with SSRC 0...")
    input_q.put({
        'user_id': '1234',
        'ssrc': 0,
        'seq': 1,
        'ts': 960,
        'data': b'\x01' * 20,
        'arrival': time.time()
    })
    time.sleep(0.02)
    
    # 2. Second packet with SSRC 1234
    print("Sending packet with SSRC 1234...")
    send_pkt(2, 1920)
    time.sleep(0.02)

        
    # 2. Jitter: out of order
    print("Sending packets 8, 7, 6 (Jitter)...")
    send_pkt(8, 8 * 960)
    time.sleep(0.01)
    send_pkt(7, 7 * 960)
    time.sleep(0.01)
    send_pkt(6, 6 * 960)
    
    # 3. Micro-gap: Packet 10 missing
    print("Sending packet 9, then skipping 10, then 11...")
    send_pkt(9, 9 * 960)
    time.sleep(0.04) # Wait a bit
    send_pkt(11, 11 * 960)
    
    # 4. Macro-gap: 2 seconds of silence
    print("Waiting 2 seconds (Macro-gap)...")
    time.sleep(2.0)
    
    # 5. Resume with a "LAGGY" packet (simulating Engine delay)
    print("Sending a laggy packet (simulated 3.5s delay)...")
    now = time.time()
    input_q.put({
        'user_id': '1234',
        'pcm': b'\x01' * 20,
        'ssrc': ssrc,
        'timestamp': 200 * 960,
        'arrival': now - 5.0 # This packet arrived 5s ago!
    })
    time.sleep(0.05)
    
    # 6. Resume
    print("Resuming packets 201-205...")
    for i in range(201, 206):
        send_pkt(i, i * 960)
        time.sleep(0.02)

        
    # Wait for finalization
    print("Waiting for silence timeout...")
    time.sleep(4.0)
    
    # 6. Verify Output
    print("\n--- Verifying Results ---")
    results = []
    while not output_q.empty():
        results.append(output_q.get())
        
    print(f"Captured {len(results)} turns.")
    for i, res in enumerate(results):
        audio = base64.b64decode(res['audio'])
        print(f"Turn {i}: Duration={res['duration_s']:.2f}s, Bytes={len(audio)}")
        
    # Shutdown
    stop_event.set()
    engine.join()
    print("Test Complete.")

if __name__ == "__main__":
    test_reconstruction()
