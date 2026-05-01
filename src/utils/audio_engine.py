import multiprocessing
import time
import queue
import os
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from utils.logger import setup_logger

logger = setup_logger("Engine")

@dataclass
class StreamState:
    packets: List[dict] = field(default_factory=list)
    last_rx_arrival: float = 0.0  # Wall-clock seconds
    last_rx_now: float = 0.0      # Engine loop seconds
    last_ts: int = 0             # RTP Samples
    active: bool = False
    ssrc: Optional[int] = None
    first_ts: Optional[int] = None
    first_arrival: float = 0.0
    current_drift: float = 0.0
    max_latency: float = 0.0
    
    # Instrumentation
    jitter_filled_pkts: int = 0
    inactivity_filled_pkts: int = 0
    first_packet_delta: float = 0.0
    total_real_pkts: int = 0

class AudioEngine(multiprocessing.Process):
    """
    Process 2: Audio Turn Finalization.
    GLM-agnostic engine that receives packets from a Sink and finalizes turns.
    """
    
    # Constants for Discord audio
    SAMPLE_RATE = 48000
    CHANNELS = 2
    SAMPLE_WIDTH = 2  # 16-bit
    BYTES_PER_FRAME = 960 * CHANNELS * SAMPLE_WIDTH # 3840 bytes (20ms)
    
    SILENCE_THRESHOLD = 1.5  # Base seconds of silence before finalizing (Lowered for responsiveness)
    HEARTBEAT_MS = 20        # Interval for checking silence

    # Mode: Global Mixing (all users mixed into one turn) vs Per-User
    GLOBAL_MIXING = os.getenv("GLOBAL_MIXING", "true").lower() == "true"

    def __init__(self, input_queue: multiprocessing.Queue, output_queue: multiprocessing.Queue, stop_event: multiprocessing.Event):
        super().__init__(name="AudioEngine")
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.stop_event = stop_event
        self.streams: Dict[str, StreamState] = {}
        self.last_bot_heartbeat = time.time()

    def run(self):
        logger.info(f"[Engine] [RUN] Process started. PID={self.pid}")
        self.main_loop()

    def main_loop(self):
        logger.debug("[Engine] [LOOP] Entering main loop.")
        while not self.stop_event.is_set():
            now_wall = time.time()
            has_data = False
            
            try:
                # 1. Wait for first packet/batch (blocking with timeout)
                msg = self.input_queue.get(timeout=self.HEARTBEAT_MS / 1000)
                now = time.time()
                
                def _handle(m):
                    if m.get('type') == 'batch':
                        for p in m['packets']:
                            self.process_packet(p, now)
                    elif m.get('type') == 'flush':
                        self._flush_all_buffers()
                    else:
                        self.process_packet(m, now)

                _handle(msg)

                # 2. Drain all other pending packets immediately
                while True:
                    try:
                        next_msg = self.input_queue.get_nowait()
                        _handle(next_msg)
                    except queue.Empty:
                        break
            except queue.Empty:
                pass
            except Exception as e:
                logger.error(f"[Engine] [LOOP_ERR] {e}")
            
            # 2. Tick Silence
            self.tick(now_wall)

            if not has_data:
                time.sleep(self.HEARTBEAT_MS / 1000.0)

    def _flush_all_buffers(self):
        """Called when a wake word is detected to drop the wake word from the prompt buffer."""
        logger.debug("[Engine] [FLUSH] Dropping current audio buffers.")
        for uid, stream in self.streams.items():
            if stream.active:
                stream.packets = []
                stream.total_real_pkts = 0
                stream.jitter_filled_pkts = 0
                stream.inactivity_filled_pkts = 0
                stream.active = False # Force clean start after flush

    def _inject_silence(self, uid: str, num_frames: int, source: str = "jitter"):
        """Injects interpolated silence packets into a stream's buffer."""
        logger.debug(f"[Engine] [SILENCE_INJECT] user={uid} | frames={num_frames} | source={source}")
        if num_frames <= 0:
            return
            
        stream = self.streams[uid]
        last_arrival = stream.last_rx_arrival
        last_ts = stream.last_ts
        
        for i in range(1, num_frames + 1):
            interp_arrival = last_arrival + (i * 0.02)
            interp_ts = last_ts + (i * 960)
            
            stream.packets.append({
                'user_id': uid,
                'data': b'\x00' * self.BYTES_PER_FRAME,
                'timestamp': interp_ts,
                'arrival': interp_arrival,
                'is_silence': True
            })
            
        if source == "jitter":
            stream.jitter_filled_pkts += num_frames
        else:
            stream.inactivity_filled_pkts += num_frames
            
        stream.last_ts = last_ts + (num_frames * 960)
        stream.last_rx_arrival = last_arrival + (num_frames * 0.02)

    def process_packet(self, pkt, now: float):
        if pkt.get('type') == 'heartbeat':
            self.last_bot_heartbeat = pkt.get('arrival', now)
            return

        arrival = pkt.get('arrival', now)
        
        # 1. Identity Management
        orig_uid = str(pkt['user_id'])
        if self.GLOBAL_MIXING:
            uid = "global"
        else:
            uid = orig_uid
        # 1. Extract Metadata from Flattened Object
        ssrc = pkt.get('ssrc', 0)
        curr_ts = pkt.get('timestamp', 0)
        pcm = pkt.get('pcm', b'')

        # 2. Synthetic Timestamp Fallback
        # If we have no RTP clock, we maintain a synthetic one to allow gap detection
        is_synthetic = False
        if uid in self.streams and curr_ts == 0:
            curr_ts = self.streams[uid].last_ts + 960
            is_synthetic = True

        logger.debug(
            f"[Engine] [PROC] user={uid} | ssrc={ssrc} | ts={curr_ts} ({'SYN' if is_synthetic else 'RTP'}) | "
            f"now={now:.4f} | diff={now-arrival:.4f}"
        )
        
        # 3. Initialize Stream
        if uid not in self.streams:
            logger.info(f"[Engine] [STREAM_NEW] user={uid} | first_ssrc={ssrc}")
            self.streams[uid] = StreamState(
                last_ts=curr_ts - 960,
                last_rx_arrival=arrival - 0.02,
                last_rx_now=now,
                active=True,
                ssrc=ssrc,
                first_packet_delta=now - arrival
            )
        
        stream = self.streams[uid]
        
        # 4. Re-activation Check
        if not stream.active:
            import numpy as np
            pcm_float = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
            rms = np.sqrt(np.mean(np.square(pcm_float)))
            
            # If the packet is pure synthetic silence from Pycord FEC flush, ignore it entirely
            # Ignore background noise (VAD-lite)
            if rms < 0.005:
                return
                
            # We log exactly what packet caused this reactivation
            logger.warning(
                f"[Engine] [STREAM_REACTIVATE] Audio detected! user={uid} | "
                f"ssrc={ssrc} | pkts_in_q={self.input_queue.qsize()} | "
                f"rms={rms:.4f}"
            )
            stream.active = True
            stream.last_rx_arrival = arrival - 0.02
            stream.first_packet_delta = now - arrival
            # CRITICAL: Reset the RTP clock so we don't inject the inter-turn pause as jitter!
            stream.last_ts = curr_ts - 960
            stream.packets = []
            stream.jitter_filled_pkts = 0
            stream.inactivity_filled_pkts = 0
            stream.total_real_pkts = 0
            stream.first_packet_delta = now - arrival
            stream.first_ts = curr_ts
            stream.first_arrival = arrival
            stream.current_drift = 0.0

        # 5. SSRC Collision Check (Reset turn if SSRC shifts unexpectedly)
        if ssrc != stream.ssrc and ssrc != 0 and stream.ssrc != 0:
            logger.debug(f"[Engine] [SSRC_CHANGE] user={uid} | old={stream.ssrc} | new={ssrc}")
            if stream.packets:
                self._finalize_turn(uid, now)
                # CRITICAL FIX: Clear packets after finalizing to prevent infinitely growing duplicate turns!
                stream.packets = []
                stream.total_real_pkts = 0
                stream.jitter_filled_pkts = 0
                stream.inactivity_filled_pkts = 0
            
            stream.ssrc = ssrc
            stream.max_latency = 0.0
            stream.last_ts = curr_ts - 960
            stream.last_rx_arrival = arrival - 0.02
            # Reset drift metrics on SSRC change to prevent threshold blowout
            stream.first_ts = curr_ts
            stream.first_arrival = arrival
            stream.current_drift = 0.0

        # 6. Gap Detection
        expected_ts = stream.last_ts + 960
        gap_samples = curr_ts - expected_ts
        
        if gap_samples >= 960 and not is_synthetic:
            num_missing = gap_samples // 960
            logger.debug(f"[Engine] [GAP_DETECTED] user={uid} | missing={num_missing}")
            self._inject_silence(uid, num_missing, source="jitter")

        # 7. Append Packet
        stream.packets.append({
            'user_id': orig_uid,
            'data': pcm,
            'timestamp': curr_ts,
            'arrival': arrival
        })
        stream.total_real_pkts += 1
        
        # 6. Latency & Drift Tracking
        latency = now - arrival
        stream.max_latency = max(stream.max_latency, latency)
        
        # Calculate drift: how far behind wall-clock is our audio timeline?
        if stream.first_ts is not None:
             audio_elapsed = (curr_ts - stream.first_ts) / self.SAMPLE_RATE
             wall_elapsed = now - stream.first_arrival
             # If audio is lagging, drift will be positive
             # If audio is lagging, drift will be positive. Cap at 3.0s to prevent unreachable thresholds.
             stream.current_drift = min(3.0, max(0.0, wall_elapsed - audio_elapsed))
        
        # 8. Update References
        stream.last_ts = curr_ts
        stream.last_rx_arrival = arrival
        stream.last_rx_now = now
        stream.active = True
        
        # 9. Emit STREAM_CHUNK for real-time wake word detection
        self.output_queue.put({
            'event_type': 'STREAM_CHUNK',
            'user_id': orig_uid,
            'audio': pcm
        })

    def tick(self, now: float):
        for uid, stream in list(self.streams.items()):
            if not stream.active:
                continue

            # 1. Measurement Comparison
            sink_silence = now - stream.last_rx_arrival
            engine_silence = now - stream.last_rx_now

            # 2. Dynamic Stabilization
            # safety_buffer handles jitter.
            # current_drift handles long-term audio clock lag.
            # bot_lag handles event-loop freezing in the bot process.
            safety_buffer = min(1.0, stream.max_latency * 2)
            bot_lag = max(0, now - self.last_bot_heartbeat - 0.6) # Allow 0.6s grace
            
            dynamic_threshold = self.SILENCE_THRESHOLD + safety_buffer + stream.current_drift + bot_lag

            # if sink_silence > 1.0:
            #      logger.debug(
            #          f"[Engine] [SILENCE_WATCH] user={uid} | sink_silence={sink_silence:.4f} | "
            #          f"dynamic_threshold={dynamic_threshold:.2f} | max_lat={stream.max_latency:.3f}"
            #      )

            # 3. Finalization logic
            # Agreement between wall-clock (sink) and internal processing (engine)
            if sink_silence > dynamic_threshold and engine_silence > 0.5:
                logger.debug(
                    f"[Engine] [FINAL_TRIGGER] user={uid} | sink={sink_silence:.4f} | "
                    f"threshold={dynamic_threshold:.2f} | now_wall={now:.4f}"
                )
                self._finalize_turn(uid, now)
                
                stream.active = False
                stream.packets = []
                stream.max_latency = 0.0

    def _finalize_turn(self, uid: str, now: float):
        import torch
        import numpy as np

        stream = self.streams[uid]
        packets = stream.packets
        if not packets:
            return

        logger.debug(f"[Engine] [FINALIZE_START] user={uid} | buffer_size={len(packets)}")
        
        # 1. Group by timestamp for mixing
        ts_map = {}
        for p in packets:
            ts = p['timestamp']
            if ts not in ts_map: ts_map[ts] = []
            ts_map[ts].append(p['data'])
        
        sorted_ts = sorted(ts_map.keys())
        
        # 2. Mix using Torch on GPU (if available)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        mixed_frames = []
        
        # Process in chunks or individually? 
        # For simplicity and to avoid complex alignment, we sum per timestamp.
        for ts in sorted_ts:
            datas = ts_map[ts]
            if len(datas) == 1:
                mixed_frames.append(datas[0])
            else:
                # Summing mixer using Torch
                tensors = [torch.from_numpy(np.frombuffer(d, dtype=np.int16).astype(np.float32)).to(device) for d in datas]
                mixed = torch.stack(tensors).sum(dim=0).clamp(-32768, 32767).to(torch.int16)
                mixed_frames.append(mixed.cpu().numpy().tobytes())
        
        final_audio = b"".join(mixed_frames)
        duration = len(final_audio) / self.BYTES_PER_FRAME * 0.02 
        
        total_pkts = len(packets)
        real_pkts = stream.total_real_pkts
        
        logger.debug(
            f"[Engine] [FINALIZE_STATS] user={uid} | dur={duration:.2f}s | "
            f"total={total_pkts} | real={real_pkts} | jitter={stream.jitter_filled_pkts}"
        )
        
        self.output_queue.put({
            'user_id': uid,
            'audio': final_audio,
            'duration_s': duration,
            'ts': time.time(), 
            'event_type': 'TURN',
            'metadata': {
                'first_ts': sorted_ts[0] if sorted_ts else 0,
                'last_ts': sorted_ts[-1] if sorted_ts else 0,
                'ssrc': stream.ssrc,
                'num_packets': total_pkts,
                'real_packets': real_pkts,
                'jitter_fill': stream.jitter_filled_pkts,
                'inactivity_fill': stream.inactivity_filled_pkts,
                'queue_latency_max': stream.max_latency,
                'mixed_mode': self.GLOBAL_MIXING
            }
        })
        logger.debug(f"[Engine] [FINALIZE_SENT] user={uid}")

def spawn_engine(input_queue: multiprocessing.Queue, output_queue: multiprocessing.Queue):
    stop_event = multiprocessing.Event()
    proc = AudioEngine(input_queue, output_queue, stop_event)
    proc.start()
    return proc, stop_event
