"""
WHY THIS FILE EXISTS:
The Sensory Ingestion Port (The Data Pipeline).

WHY:
Discord sends us a stream of RTP packets. This sink acts as the 
primary 'Ingestion Port' that decodes those packets into raw PCM 
samples and routes them to the external Audio Engine process.

CONCEPT: IPC EFFICIENCY
Sending every single 20ms frame across a process boundary (IPC) is 
extremely inefficient. We 'Batch' these frames into larger segments. 
This reduces the context-switching overhead on the CPU, ensuring that 
the signaling thread stays responsive to Discord's heartbeat.
"""

import discord
import collections
import threading
import time
import multiprocessing
from core.logger import setup_logger

logger = setup_logger("voice.sink")

class PCMQueueSink(discord.sinks.Sink):
    """
    The Sensory Ingestion Port.
    
    WHY THIS SINK?
    It collects decoded PCM data from Pycord and batches it before 
    pushing to a multiprocessing Queue. This is the 'First Hop' in 
    our Recursive ETL pipeline.
    """
    def __init__(self, queue: multiprocessing.Queue):
        super().__init__()
        self.queue = queue
        self.is_listening = True
        # Required by Pycord master branch for DAVE support
        self.__sink_listeners__ = []
        self._batches = collections.defaultdict(list)
        self._batch_size = 25 # Every 500ms (25 frames)
        
        self._batch_lock = threading.Lock()
        self._last_flush = collections.defaultdict(float)
        self._flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._flush_thread.start()

    def _flush_loop(self):
        """
        Periodically flushes pending batches so frames aren't trapped 
        during periods of user silence.
        
        WHY: 100ms is the 'Planck Time' of our flush loop. It ensures 
        stale audio doesn't linger in the buffer.
        """
        while True:
            time.sleep(0.05)
            now = time.time()
            with self._batch_lock:
                for uid, batch in list(self._batches.items()):
                    if batch and (now - self._last_flush[uid] >= 0.1):
                        self.queue.put_nowait({
                            'type': 'batch',
                            'packets': list(batch)
                        })
                        batch.clear()
                        self._last_flush[uid] = now

    def walk_children(self):
        yield self

    def write(self, data, user):
        """
        Called by Pycord's PacketRouter for every decoded audio packet.
        """
        try:
            if not data or not self.is_listening:
                return
            
            now = time.time()
            if not isinstance(data, (bytes, bytearray)):
                packet = getattr(data, 'packet', None)
                pcm = getattr(data, 'pcm', b'')
                ssrc = getattr(packet, 'ssrc', 0) if packet else 0
                timestamp = getattr(packet, 'timestamp', 0) if packet else 0
            else:
                pcm = data
                ssrc = 0
                timestamp = 0
            
            # Use the user's integer ID
            user_id = getattr(user, 'id', None) or id(user)
            
            # Batching Logic
            with self._batch_lock:
                batch = self._batches[user_id]
                batch.append({
                    'user_id': user_id,
                    'pcm': pcm,
                    'ssrc': ssrc,
                    'timestamp': timestamp,
                    'arrival': time.time()
                })
                
                if len(batch) >= self._batch_size:
                    self.queue.put_nowait({
                        'type': 'batch',
                        'packets': list(batch)
                    })
                    batch.clear()
                    self._last_flush[user_id] = now
        except Exception as e:
            logger.error(f"[Sink] Error in write: {e}")
