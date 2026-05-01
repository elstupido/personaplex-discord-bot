import discord
import discord.voice_client
import asyncio

async def test():
    print(f"VoiceClient has: {dir(discord.voice_client)}")
    try:
        from discord.voice_client import VoiceProtocol
        print("Found VoiceProtocol")
    except ImportError:
        print("VoiceProtocol NOT found")

asyncio.run(test())
