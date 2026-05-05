"""
The Executive Branch & Signaling Hub (The Voice Service Cog).

WHY THIS FILE EXISTS:
This is the 'Signaling layer'. It handles Discord UI (Slash Commands), 
voice state tracking (joining/leaving), and high-level session orchestration.

THE SIGNALING CONTRACT:
Because audio DSP and neural inference happen in external processes 
and servers, the Cog's primary job is to maintain the 'Shared Social 
Reality' between the users and the AI. It acts as the switchboard 
operator, ensuring that the right audio streams are routed to the 
right neural kernels at the right time.
"""

import discord
import os
import asyncio
from discord.ext import commands
from discord.commands import slash_command, Option

from core.logger import setup_logger
from ai.stupid_factory import create_bridge
from ai.orchestrator import orchestrator

# Functional Experts
from voice.patches import apply_reader_patches
from voice.trigger import TriggerEngine
from voice.orchestrator import AudioOrchestrator
from voice.source import PersonaPlexAudioSource

# Apply surgical interventions on import
apply_reader_patches()

logger = setup_logger("cogs.voice_service")

class VoiceServiceCog(commands.Cog):
    """
    The Executive Hub for Voice Interactions.
    """
    def __init__(self, bot: discord.Bot):
        self.bot = bot
        self.active_session = None
        
        # Load identity preferences
        self.voice_preset = os.getenv("VOICE_PRESET", "NATF2")
        self.text_prompt = os.getenv("TEXT_PROMPT", "You are a helpful RPG narrator.")
        self.vocoder = os.getenv("VOCODER_BACKEND", "glm")
        
        # Instantiate AI Bridge
        self.bridge = None
        
        # Template for pre-warming (used in on_ready)
        wake_word  = os.getenv("WAKE_WORD",  "hey stupid")
        clone_word = os.getenv("CLONE_WORD", "clone my voice")
        self._trigger_template = TriggerEngine(wake_word, clone_word)
        
        # Suppress benign Opus warnings
        import logging
        logging.getLogger("discord.opus").setLevel(logging.ERROR)

    @commands.Cog.listener()
    async def on_ready(self):
        """Warm up models immediately."""
        if not self.bridge: return
        
        if os.getenv("LAZY_LOAD_MODELS", "true").lower() == "true":
            logger.info("💤 [VoiceService] Lazy Mode: Skipping eager warmup.")
        else:
            logger.info("🔥 [VoiceService] Eager Mode: Igniting kernels...")
            asyncio.create_task(self._ensure_bridge())

    async def _warmup_all(self):
        loop = asyncio.get_event_loop()
        await asyncio.gather(
            self.bridge.connect(),
            loop.run_in_executor(None, self._trigger_template.warmup),
            return_exceptions=True
        )
        logger.info("✅ [VoiceService] All systems warm.")

    async def _ensure_bridge(self):
        """Lazy initializer for the AI bridge."""
        if self.bridge is None:
            self.bridge = await create_bridge(self.voice_preset, self.text_prompt, vocoder=self.vocoder)
            await self.bridge.connect()
        return self.bridge

    # --- UI Commands ---

    @slash_command(name="ping", description="Check bot status.")
    async def ping(self, ctx: discord.ApplicationContext):
        ms = round(self.bot.latency * 1000)
        await ctx.respond(f"✅ Online. Latency: {ms}ms")

    @slash_command(name="join", description="Join a voice channel.")
    async def join(self, ctx: discord.ApplicationContext, channel: discord.VoiceChannel = None):
        if not channel:
            if not ctx.author.voice:
                return await ctx.respond("You are not in a voice channel.")
            channel = ctx.author.voice.channel

        if not ctx.interaction.response.is_done():
            await ctx.defer()
        try:
            # 1. Ensure Bridge is initialized and warmed up
            await self._ensure_bridge()
            
            audio_source = PersonaPlexAudioSource()
            self.bridge.audio_source = audio_source
            self.bridge.text_channel = ctx.channel
            
            # 2. Establish Voice Link
            vc = await self._connect_vc(ctx.guild, channel)
            
            # 3. Start Session
            self.active_session = await self._start_session(
                vc, channel, self.bridge, audio_source, self._trigger_template
            )
            await ctx.followup.send(f"🎙️ Identity Link Established in {channel.mention}!")
        except Exception as e:
            logger.error(f"💥 Join failed: {e}", exc_info=True)
            await ctx.followup.send(f"🛑 Connection Terminated: {e}")

    @slash_command(name="leave", description="Leave voice.")
    async def leave(self, ctx: discord.ApplicationContext):
        if not self.active_session:
            return await ctx.respond("Not in a session.", ephemeral=True)
        if not ctx.interaction.response.is_done():
            await ctx.defer()
        await self._teardown_session()
        await ctx.followup.send("🖕 Session ended.")

    # --- RPG Character Management ---

    voice = discord.SlashCommandGroup("voice", "Manage RPG character voices.")

    @voice.command(name="clone", description="Clone a voice profile.")
    async def voice_clone(self, ctx: discord.ApplicationContext, name: Option(str, "Character name", default=None)):
        if not self.active_session:
            return await ctx.respond("Join voice first!", ephemeral=True)
            
        orch_ptr = self.active_session['orchestrator']
        orch_ptr.is_cloning = True
        
        if name:
            name = "".join([c for c in name if c.isalnum() or c in ('_', '-')]).lower()
            self.bridge.active_voice = name
            await ctx.respond(f"🎙️ **Cloning Mode Active!** Profile: **{name}**")
        else:
            await ctx.respond("🎙️ **Cloning Mode Active!** Speak for 3-5 seconds.")

    @voice.command(name="switch", description="Switch character.")
    async def voice_switch(self, ctx: discord.ApplicationContext, name: str):
        if not self.active_session:
            return await ctx.respond("Not in a session.", ephemeral=True)
        if await self.bridge.switch_voice(name):
            await ctx.respond(f"✨ Now speaking as: **{name}**")
        else:
            await ctx.respond(f"🛑 Character **{name}** not found.")

    # --- Internal Lifecycle ---

    async def _connect_vc(self, guild, channel) -> discord.VoiceClient:
        vc = guild.voice_client
        if vc:
            if vc.channel != channel: await vc.move_to(channel)
        else:
            vc = await channel.connect(timeout=30.0)
        await vc.guild.change_voice_state(channel=channel, self_deaf=False, self_mute=False)
        return vc

    async def _start_session(self, vc, channel, bridge, source, trigger_engine) -> dict:
        for member in channel.members:
            orchestrator.register_user(str(member.id), member.display_name)

        orch = AudioOrchestrator(vc, bridge, asyncio.get_running_loop())
        
        # Rebind pre-warmed engine
        if trigger_engine:
            trigger_engine.on_trigger = orch._on_trigger
            orch.trigger_engine = trigger_engine
        
        await orch.start()
        bridge.vc = vc
        await bridge.start_streaming()
        
        # Start playback source
        vc.play(source)
        vc.pause()
        source.attach_vc(vc, asyncio.get_running_loop())
        
        return {'vc': vc, 'orchestrator': orch, 'bridge': bridge, 'source': source}

    async def _teardown_session(self):
        s = self.active_session
        self.active_session = None
        if not s: return
        
        for key in ['bridge', 'orchestrator', 'vc']:
            try:
                task = s[key].close() if key == 'bridge' else s[key].stop() if key == 'orchestrator' else s[key].disconnect()
                await task
            except Exception: pass

def setup(bot: discord.Bot):
    bot.add_cog(VoiceServiceCog(bot))
