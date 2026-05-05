"""
WHY THIS FILE EXISTS:
This is the primary entry point and lifecycle manager for the PersonaPlex Discord Bot.
It orchestrates the Discord connection, loads voice service extensions, and ensures
essential native libraries (like Opus) are loaded for audio processing.

WHY: We separate this from the AI logic so the Discord connection remains stable 
even if the AI bridges crash or need to be reloaded.
"""
import sys
import os
import warnings
import asyncio

# WHY: Suppress noisy library-internal deprecation warnings that we cannot fix 
# without refactoring third-party dependencies.
warnings.filterwarnings("ignore", category=FutureWarning, module="diffusers")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*weight_norm.*")

# Add src to sys.path to resolve absolute imports correctly
# WHY: Because running scripts from random directories shouldn't break import resolution.
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import discord
from dotenv import load_dotenv
from core.logger import setup_logger
import torch
import aiohttp
import logging

# WHY: Muzzle noisy library-internal logs that pollute the real-time stream.
logging.getLogger("faster_whisper").setLevel(logging.WARNING)
logging.getLogger("funasr").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.ERROR)

# Load environment variables
load_dotenv()

# Setup root bot logger
logger = setup_logger("bot")

# Force load Opus (Required for Voice in Docker and Windows)
try:
    if sys.platform == "win32":
        discord.opus.load_opus('./libopus-0.dll')
    else:
        discord.opus.load_opus('libopus.so.0')
    logger.info("Opus library explicitly loaded.")
except Exception as e:
    logger.error(f"Failed to load Opus: {e}")

class PersonaPlexBot(discord.Bot):

    def __init__(self):
        intents = discord.Intents.all()
        
        # Load guild ID for instant command registration
        guild_id = os.getenv("GUILD_ID")
        debug_guilds = [int(guild_id)] if guild_id else None
        
        super().__init__(intents=intents, debug_guilds=debug_guilds)
        
        # Load extensions (Cogs)
        # WHY: We dynamically load the voice cog rather than hardcoding its logic here
        # so we can potentially hot-reload it in the future without dropping the bot connection.
        self.load_extension("voice.cog")

    async def on_ready(self):
        # Confirm GPU availability (informational — the model runs on the Moshi server)
        if torch.cuda.is_available():
            logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("No GPU detected on bot process (OK — model runs on Moshi server)")

        logger.info(f"Logged in as {self.user} (ID: {self.user.id})")
        logger.info("------")
        
        # 🛰️ EAGER WARMUP HANDSHAKE
        # WHY: The bot must be 'Ready to Talk' before the user issues a command.
        # We fire this in the background so it doesn't block the Discord heartrate.
        asyncio.create_task(self._eager_warmup())

    async def _eager_warmup(self):
        """Primes the inference server's neural kernels."""
        url = "http://localhost:10000/warmup"
        logger.info(f"🛰️  Triggering Eager Warmup at {url}...")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, timeout=300) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        logger.info("✅ Eager Warmup Complete. Neural Experts are standing by. 🧠🔥")
                    else:
                        logger.warning(f"⚠️ Eager Warmup Handshake failed (HTTP {resp.status})")
        except Exception as e:
            logger.warning(f"⚠️ Could not reach Inference Gateway for warmup: {e}")

    async def on_voice_state_update(self, member, before, after):
        if member.id == self.user.id:
            logger.debug(f"[Voice-State] Bot moved: {before.channel} -> {after.channel} (Deaf: {after.self_deaf})")

    async def on_voice_server_update(self, data):
        logger.debug(f"[Voice-Server] Server Update! Endpoint: {data.get('endpoint')} Token: {data.get('token')[:10]}...")

def main():
    print("\n" + "="*50)
    print("PERSONAPLEX BOT v2.1.0 - SYNC VERIFIED")
    print("="*50 + "\n")
    token = os.getenv("DISCORD_TOKEN")
    if not token or token == "your_token_here":
        logger.error("Please set a valid DISCORD_TOKEN in the .env file.")
        return

    # Bot starts here
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    bot = PersonaPlexBot()

    logger.info("Starting PersonaPlex Discord bot...")
    bot.run(token)

if __name__ == "__main__":
    main()
