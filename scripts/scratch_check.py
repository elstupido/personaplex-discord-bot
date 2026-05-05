import discord
import asyncio

async def check_vc():
    print(f"VoiceClient methods: {[m for m in dir(discord.VoiceClient) if 'listen' in m or 'record' in m]}")

asyncio.run(check_vc())
