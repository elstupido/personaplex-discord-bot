"""
WHY THIS FILE EXISTS:
The Voice Pipeline Orchestrator.

WHY:
Audio handling in Discord is a multi-process nightmare. We have:
1. The Bot Process (Signaling)
2. The Engine Process (DSP/Jitter Buffer)
3. The AI Server (Inference)

The AudioOrchestrator is the 'Glue' that binds these together. It manages 
the IPC queues, spawns the sub-processes, and listens for 'Trigger Events' 
(Wake words) to transition the bot from idle to active.
"""

import os
import asyncio
import threading
import time
import multiprocessing
from core.logger import setup_logger
from voice.engine import spawn_engine
from voice.trigger import TriggerEngine

logger = setup_logger("voice.orchestrator")

class AudioOrchestrator:
    """
    Manages the lifecycle of the Audio Engine process and the Native Sink.
    Listens to the finalized audio queue and dispatches to the bridge.
    """
    def __init__(self, vc, bridge, loop):
        self.vc = vc
        self.bridge = bridge
        self.loop = loop
        
        # IPC Queues
        self.reconstruction_q = multiprocessing.Queue()
        self.bot_q = multiprocessing.Queue()
        
        # Sub-processes
        self.collector_proc = None
        self.engine_proc = None
        self.stop_events = []
        
        self.is_running = False
        self._listener_task = None
        
        # State Management
        self.is_awake = False
        self.is_cloning = False
        
        # Trigger Engine (Enhanced for 5090)
        wake_word = os.getenv("WAKE_WORD", "hey stupid")
        clone_word = os.getenv("CLONE_WORD", "clone my voice")
        self.trigger_engine = TriggerEngine(wake_word, clone_word)
        self.trigger_engine.on_trigger = self._on_trigger
        
        # Default to SenseVoice if we have a server URL (Fallback to localhost:10000)
        server_url = os.getenv("GLM_SERVER_URL", "http://127.0.0.1:10000")
        backend = "sensevoice" if server_url else "whisper"
        self.trigger_engine.set_backend(backend)

    def _on_trigger(self, trigger_type: str, text: str):
        """Callback fired by TriggerEngine when a wake/clone word is detected."""
        logger.info(f"✨ [Orchestrator] TRIGGER FIRED: {trigger_type} (Matched: '{text}')")
        feedback_msg = None
        if trigger_type == "wake":
            self.is_awake = True
        elif trigger_type == "clone":
            self.is_cloning = True
            feedback_msg = "🎙️ **Voice Cloning Mode Active!** Please speak for 3-5 seconds to provide a reference."
            
        # Immediately flush the engine's buffer to drop the wake/clone word
        try:
            self.reconstruction_q.put_nowait({'type': 'flush'})
        except Exception:
            pass

        # Play visual/audio feedback
        if self.bridge:
            if hasattr(self.bridge, 'play_ding'):
                asyncio.run_coroutine_threadsafe(self.bridge.play_ding(), self.loop)
            
            if feedback_msg and hasattr(self.bridge, 'text_channel') and self.bridge.text_channel:
                asyncio.run_coroutine_threadsafe(self.bridge.text_channel.send(feedback_msg), self.loop)

    async def start(self):
        """Initialize and start the 3-process pipeline."""
        logger.info("[Orchestrator] Starting audio pipeline...")
        
        # 1. Wait for connection to be fully established
        max_retries = 100 # 20 seconds
        logger.debug("[Orchestrator] Waiting for VoiceClient connection...")
        while not self.vc.is_connected() and max_retries > 0:
            await asyncio.sleep(0.2)
            max_retries -= 1
            
        if not self.vc.is_connected():
            raise RuntimeError("VoiceClient failed to connect within timeout")

        # 2. Spawn Engine (Process 2)
        engine_proc, engine_stop = spawn_engine(self.reconstruction_q, self.bot_q)
        self.engine_proc = engine_proc
        self.stop_events.append(engine_stop)
        
        # 3. Start Native Sink
        from voice.sink import PCMQueueSink
        self.vc.start_recording(
            PCMQueueSink(self.reconstruction_q),
            self._on_recording_finished
        )
        logger.info("[Orchestrator] Native Sink started. Pipeline active.")
        
        # 4. Start Bot-side listener, Heartbeat and Loop Monitor
        self.is_running = True
        self.trigger_engine.start()
        self._listener_thread = threading.Thread(target=self._listen_to_engine, daemon=True)
        self._listener_thread.start()
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self._monitor_task = asyncio.create_task(self._loop_monitor())
        
        logger.info(f"[Orchestrator] Pipeline active. Engine={engine_proc.pid}")

    async def _loop_monitor(self):
        """Monitor for event loop blocking / GIL starvation."""
        while self.is_running:
            start = time.time()
            await asyncio.sleep(0.1)
            elapsed = time.time() - start
            if elapsed > 0.15: # 50ms threshold
                logger.warning(f"[LoopMonitor] Event loop blocked for {elapsed-0.1:.4f}s")

    async def _heartbeat_loop(self):
        """Send a heartbeat to the engine to prove the Bot event loop is alive."""
        while self.is_running:
            try:
                self.reconstruction_q.put_nowait({
                    'type': 'heartbeat',
                    'arrival': time.time()
                })
            except Exception:
                pass
            await asyncio.sleep(0.5)

    async def stop(self):
        """Cleanly shut down the pipeline."""
        self.is_running = False
            
        for event in self.stop_events:
            event.set()
            
        # 1. Stop recording
        if hasattr(self.vc, 'stop_recording'):
            try:
                self.vc.stop_recording()
            except Exception as e:
                logger.error(f"[Orchestrator] Error stopping recording: {e}")
                
        self.trigger_engine.stop()
            
        # 2. Terminate engine process
        if self.engine_proc:
            self.engine_proc.terminate()
            self.engine_proc.join()
            
        logger.info("[Orchestrator] Pipeline stopped.")

    def _listen_to_engine(self):
        """Pull finalized segments from the Engine and feed the Bridge."""
        import queue
        while self.is_running:
            try:
                try:
                    payload = self.bot_q.get(timeout=0.5)
                except queue.Empty:
                    continue

                if not payload:
                    continue
                event_type = payload.get('event_type')
                
                if event_type == 'STREAM_CHUNK':
                    self.trigger_engine.feed(payload.get('audio', b''))
                    continue
                
                if event_type != 'TURN':
                    continue
                
                # Resolve UserID
                user_id = payload.get('user_id')
                user = self.vc.guild.get_member(user_id) if user_id else None
                username = user.display_name if user else f"User {user_id}"
                
                # Only dispatch if we are explicitly awake or cloning
                if self.is_awake or self.is_cloning:
                    duration = payload.get('duration_s', 0)
                    
                    if duration < 0.5:
                        logger.debug(f"[Orchestrator] Ignoring micro-turn ({duration:.2f}s).")
                        continue
                        
                    logger.debug(f"[Orchestrator] [DISPATCH] Handing {duration:.2f}s turn to Bridge for {username}")
                    payload['is_clone_reference'] = self.is_cloning
                    
                    self.is_awake = False
                    self.is_cloning = False
                    
                    asyncio.run_coroutine_threadsafe(self.bridge.send_audio_packet(payload), self.loop)
                else:
                    logger.debug(f"[Orchestrator] [IDLE] Dropping {payload.get('duration_s', 0):.2f}s turn from {username}")
                
            except Exception as e:
                logger.error(f"[Orchestrator] Listener error: {e}")
                time.sleep(0.1)

    async def _on_recording_finished(self, sink, *args):
        """Callback for when recording stops."""
        logger.info("[Orchestrator] Recording session finished.")
