import sys
import os
import discord
from dotenv import load_dotenv
from utils.logger import setup_logger
import torch

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
        self.load_extension("cogs.voice_service")

    async def on_ready(self):
        # Confirm GPU availability (informational — the model runs on the Moshi server)
        if torch.cuda.is_available():
            logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("No GPU detected on bot process (OK — model runs on Moshi server)")

        logger.info(f"Logged in as {self.user} (ID: {self.user.id})")
        logger.info("------")

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

    import asyncio
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
