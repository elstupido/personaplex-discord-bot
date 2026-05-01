import discord.voice_client
import asyncio

def find_protocol():
    for name in dir(discord.voice_client):
        obj = getattr(discord.voice_client, name)
        if isinstance(obj, type) and hasattr(obj, 'datagram_received'):
            print(f"FOUND_PROTOCOL: {name}")

find_protocol()
