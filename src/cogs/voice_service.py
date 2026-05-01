"""
PersonaPlex Voice Service Cog

Orchestrates the 2-process audio capture pipeline:
1. Native Sink   (Bot Process)   - Decryption and decoding via Pycord.
2. AudioEngine   (External Process) - Jitter buffering and silence detection.
3. VoiceServiceCog (Bot Process)   - Signaling and inference dispatch.
"""

import discord
from discord.ext import commands
from discord.commands import slash_command
from utils.logger import setup_logger
import os
import asyncio
import multiprocessing
import threading
import time
import base64
import collections
import numpy as np
import torch
import torchaudio.functional as F
import wave
import sys
from bridge.factory import create_bridge
from bridge.orchestrator import orchestrator, Colors
from utils.audio_engine import spawn_engine
from utils.trigger_engine import TriggerEngine

logger = setup_logger("cogs.voice_service")

# ---------------------------------------------------------------------------
# Discord audio constants
# ---------------------------------------------------------------------------
DISCORD_FRAME_BYTES = 3840 # 20ms stereo int16 at 48kHz

# ---------------------------------------------------------------------------
# AudioOrchestrator
# ---------------------------------------------------------------------------

class PCMQueueSink(discord.sinks.Sink):
    """
    Native Pycord Sink that pipes decoded PCM to the Audio Engine.
    Handles DAVE (E2EE) audio packets.
    """
    def __init__(self, queue: multiprocessing.Queue):
        super().__init__()
        self.queue = queue
        self.is_listening = True
        # Required by Pycord master branch for DAVE support
        self.__sink_listeners__ = []
        self._batches = collections.defaultdict(list)
        self._batch_size = 25 # Send every 500ms (25 frames) to minimize IPC overhead
        
        self._batch_lock = threading.Lock()
        self._last_flush = collections.defaultdict(float)
        self._flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._flush_thread.start()

    def _flush_loop(self):
        """Periodically flushes pending batches so frames aren't trapped during silence."""
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
            self._last_arrival = now
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
        
        # Trigger Engine
        wake_word = os.getenv("WAKE_WORD", "hey stupid")
        clone_word = os.getenv("CLONE_WORD", "clone my voice")
        self.trigger_engine = TriggerEngine(wake_word, clone_word, self._on_trigger)

    def _on_trigger(self, trigger_type: str):
        """Callback fired by TriggerEngine when a wake/clone word is detected."""
        logger.info(f"[Orchestrator] Trigger fired: {trigger_type}")
        feedback_msg = None
        if trigger_type == "WAKE":
            self.is_awake = True
        elif trigger_type == "CLONE":
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
        # vc.start_recording() handles decryption and Opus decoding internally.
        self.vc.start_recording(
            PCMQueueSink(self.reconstruction_q),
            self._on_recording_finished
        )
        logger.info("[Orchestrator] Native Sink started. Pycord is handling decryption/decoding via start_recording.")
        
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
        # The daemon thread will exit on the next loop iteration due to is_running.
            
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
                # Run synchronously in our dedicated thread
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
                    
                    # Ignore empty/micro turns (e.g. from the buffer flush or a quick breath)
                    # so we don't prematurely reset the awake state before the real prompt arrives.
                    if duration < 0.5:
                        logger.debug(f"[Orchestrator] Ignoring micro-turn ({duration:.2f}s). Staying awake.")
                        continue
                        
                    logger.debug(f"[Orchestrator] [DISPATCH] Handing {duration:.2f}s turn to Bridge for {username}")
                    payload['is_clone_reference'] = self.is_cloning
                    
                    # Reset state instantly
                    self.is_awake = False
                    self.is_cloning = False
                    
                    # Dispatch to bridge inside the asyncio event loop
                    asyncio.run_coroutine_threadsafe(self.bridge.send_audio_packet(payload), self.loop)
                else:
                    logger.debug(f"[Orchestrator] [IDLE] Dropping {payload.get('duration_s', 0):.2f}s turn from {username}")
                
                
            except Exception as e:
                logger.error(f"[Orchestrator] Listener error: {e}")
                time.sleep(0.1)
    async def _on_recording_finished(self, sink, *args):
        """Callback for when recording stops."""
        logger.info("[Orchestrator] Recording session finished.")


# ---------------------------------------------------------------------------
# PersonaPlexAudioSource
# ---------------------------------------------------------------------------

class PersonaPlexAudioSource(discord.AudioSource):
    """
    Receives 22050 Hz mono int16 frames from the GLM-4 server,
    resamples to 48000 Hz stereo, and feeds Discord's audio loop.
    """
    def __init__(self):
        self.frame_buffer = collections.deque(maxlen=5000) # 100 seconds of 20ms frames
        self.lock = threading.Lock()
        self.pre_roll_count = 80 # Buffer 1.6s (80 frames) before starting playback to prevent stutter
        self.is_playing = False
        self._last_feed_time = 0
        self._first_frame_read = False

    def clear(self):
        """Flush the playback buffer."""
        with self.lock:
            self.frame_buffer.clear()
            self.is_playing = False
            self._first_frame_read = False

    def prepare_for_model(self):
        """Called when a new model response starts to ensure the pre-roll buffer is applied."""
        with self.lock:
            # Setting is_playing to False forces feed() to wait for pre_roll_count
            self.is_playing = False
            self._first_frame_read = False

    def feed(self, pcm_frame: bytes):
        """Feed PCM from the AI model (uses preroll buffer to prevent stutter)."""
        with self.lock:
            if not self.frame_buffer:
                self._last_feed_time = time.time()
                self._first_frame_read = False
            
            frames_before = len(self.frame_buffer)
            self._enqueue_frames(pcm_frame)
            frames_after = len(self.frame_buffer)
            
            logger.debug(f"[AudioSource] [FEED_MODEL] Added {frames_after - frames_before} frames. Total={frames_after}")
            
            if not self.is_playing and len(self.frame_buffer) >= self.pre_roll_count:
                self.is_playing = True

    def feed_raw(self, pcm_48k_stereo: bytes):
        """Feed 48000 Hz stereo PCM directly (instant playback for dings, bypasses preroll)."""
        with self.lock:
            if not self.frame_buffer:
                self._last_feed_time = time.time()
                self._first_frame_read = False
            
            frames_before = len(self.frame_buffer)
            self._enqueue_frames(pcm_48k_stereo)
            frames_after = len(self.frame_buffer)
            
            logger.debug(f"[AudioSource] [FEED_RAW] Added {frames_after - frames_before} frames. Total={frames_after}")
            
            if not self.is_playing:
                self.is_playing = True

    # def _resample_to_discord(self, pcm_22k: bytes) -> bytes:
    #     mono_f32 = np.frombuffer(pcm_22k, dtype=np.int16).astype(np.float32) / 32768.0
    #     tensor = torch.from_numpy(mono_f32).unsqueeze(0)
    #     resampled = F.resample(tensor, 22050, 48000)
    #     stereo = torch.cat([resampled, resampled], dim=0)
    #     return (stereo.clamp(-1, 1).numpy() * 32767).astype(np.int16).T.copy(order='C').tobytes()

    def _enqueue_frames(self, pcm: bytes):
        """Append PCM frames to the deque. Deque is thread-safe for appends."""
        mv = memoryview(pcm)
        for i in range(0, len(pcm), DISCORD_FRAME_BYTES):
            frame = mv[i:i + DISCORD_FRAME_BYTES]
            if len(frame) == DISCORD_FRAME_BYTES:
                self.frame_buffer.append(frame.tobytes())

    def read(self) -> bytes:
        if not self.is_playing:
            return b'\x00' * DISCORD_FRAME_BYTES

        try:
            # collections.deque.popleft() is thread-safe.
            # We avoid the lock on the hot-path to prevent voice-thread starvation.
            frame = self.frame_buffer.popleft()
            if not self._first_frame_read:
                self._first_frame_read = True
                logger.debug(f"[AudioSource] [PLAY_START] Buffer size: {len(self.frame_buffer)}")
            return frame
        except IndexError:
            with self.lock:
                self.is_playing = False
            return b'\x00' * DISCORD_FRAME_BYTES

    def is_opus(self):
        return False

    def cleanup(self):
        with self.lock:
            self.frame_buffer.clear()
            self.is_playing = False


# ---------------------------------------------------------------------------
# VoiceServiceCog
# ---------------------------------------------------------------------------

class VoiceServiceCog(commands.Cog):
    def __init__(self, bot: discord.Bot):
        self.bot = bot
        self.active_session = None
        self.voice_preset = os.getenv("VOICE_PRESET", "NATF2")
        self.text_prompt = os.getenv(
            "TEXT_PROMPT",
            "You enjoy having a good conversation. You are curious and ask "
            "thoughtful questions. You find everything funny.",
        )
        self.bridge = create_bridge(self.voice_preset, self.text_prompt)
        
        # Pre-build the TriggerEngine so warmup() can be called in on_ready
        wake_word  = os.getenv("WAKE_WORD",  "hey stupid")
        clone_word = os.getenv("CLONE_WORD", "clone my voice")
        self._trigger_engine_template = TriggerEngine(wake_word, clone_word, lambda _: None)
        
        # Suppress Pycord's benign packet loss warnings from VAD
        import logging
        logging.getLogger("discord.opus").setLevel(logging.ERROR)

    @commands.Cog.listener()
    async def on_ready(self):
        """Warm up all models immediately when bot is ready."""
        if self.bridge:
            logger.info("[VoiceService] Pre-loading and warming up models...")
            asyncio.create_task(self._warmup_all())

    async def _warmup_all(self):
        """Run all warmup tasks concurrently so startup is as fast as possible."""
        loop = asyncio.get_event_loop()
        # Bridge warmup (GLM models, Whisper VQ encoder, audio decoder)
        bridge_task = asyncio.create_task(self.bridge.connect())
        # TriggerEngine warmup (faster-whisper tiny.en + dummy inference)
        trigger_task = loop.run_in_executor(None, self._trigger_engine_template.warmup)
        await asyncio.gather(bridge_task, trigger_task, return_exceptions=True)
        logger.info("[VoiceService] All models warm and ready.")

    @slash_command(name="ping", description="Check bot status.")
    async def ping(self, ctx: discord.ApplicationContext):
        ms = round(self.bot.latency * 1000)
        await ctx.respond(f"✅ Online. Latency: {ms}ms")

    @slash_command(name="join", description="Join a voice channel.")
    async def join(self, ctx: discord.ApplicationContext, channel: discord.VoiceChannel = None):
        await ctx.defer()
        if not channel:
            if ctx.author.voice:
                channel = ctx.author.voice.channel
            else:
                return await ctx.followup.send("You are not in a voice channel!")

        if self.active_session:
            await self._teardown_session()

        try:
            # 1. Prepare session
            audio_source = PersonaPlexAudioSource()
            # Update the bridge with the new audio source and text channel
            self.bridge.audio_source = audio_source
            self.bridge.text_channel = ctx.channel
            
            # 2. Join the channel
            vc = await self._connect_vc(ctx.guild, channel)
            await ctx.followup.send(f"👋 Joined {channel.mention}!")
            
            # 3. Start the session — hand off the pre-warmed TriggerEngine so the
            #    first trigger fires instantly without a cold-start delay.
            self.active_session = await self._start_session(
                vc, channel, self.bridge, audio_source, self._trigger_engine_template
            )
        except Exception as e:
            logger.error(f"Join failed: {e}", exc_info=True)
            await ctx.followup.send(f"❌ Error: {e}")

    @slash_command(name="leave", description="Leave voice.")
    async def leave(self, ctx: discord.ApplicationContext):
        if not self.active_session:
            return await ctx.respond("Not in a session.", ephemeral=True)
        await ctx.defer()
        await self._teardown_session()
        await ctx.followup.send("🖕 Session ended.")

    @slash_command(name="clone", description="Clone your voice. The bot will listen to your next 5 seconds of speech.")
    async def clone(self, ctx: discord.ApplicationContext):
        if not self.active_session:
            return await ctx.respond("Join a voice channel first!", ephemeral=True)
        
        # Tell the orchestrator to mark the next turn as a clone reference
        self.active_session['orchestrator'].is_cloning = True
        
        await ctx.respond("🎙️ **Voice Cloning Mode Active!** Please speak for 3-5 seconds to provide a reference.")

    async def _connect_vc(self, guild, channel) -> discord.VoiceClient:
        vc = guild.voice_client
        if vc:
            if vc.channel != channel:
                await vc.move_to(channel)
        else:
            vc = await channel.connect(timeout=30.0)
        await vc.guild.change_voice_state(channel=channel, self_deaf=False, self_mute=False)
        return vc

    async def _start_session(self, vc, channel, bridge, audio_source, trigger_engine=None) -> dict:
        for member in channel.members:
            orchestrator.register_user(str(member.id), member.display_name)

        # Start the audio orchestrator
        orch = AudioOrchestrator(vc, bridge, asyncio.get_running_loop())
        
        # Inject the pre-warmed TriggerEngine if provided, rebinding its callback
        # to the live orchestrator. This eliminates cold-start lag on the first trigger.
        if trigger_engine is not None:
            trigger_engine.on_trigger = orch._on_trigger
            orch.trigger_engine = trigger_engine
        
        await orch.start()
        
        # Start bridge processing and voice playback
        bridge.vc = vc
        await bridge.start_streaming()
        vc.play(audio_source)

        return {'vc': vc, 'orchestrator': orch, 'bridge': bridge, 'source': audio_source}

    async def _teardown_session(self):
        s = self.active_session
        self.active_session = None
        if not s: return
        
        try:
            await s['bridge'].close()
        except Exception: pass
        
        try:
            await s['orchestrator'].stop()
        except Exception: pass
        
        try:
            await s['vc'].disconnect()
        except Exception: pass

def setup(bot: discord.Bot):
    bot.add_cog(VoiceServiceCog(bot))
