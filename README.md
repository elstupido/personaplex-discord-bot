# PersonaPlex Discord Bot

A Discord voice bot powered by [NVIDIA PersonaPlex](https://github.com/NVIDIA/personaplex) — a real-time, full-duplex speech-to-speech conversational AI with persona control.

Join a voice channel, talk to the AI, and hear it respond in real-time with configurable voice presets and system prompts.

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                 Docker Container (GPU)                │
│                                                      │
│  ┌──────────────────┐    WebSocket    ┌───────────┐  │
│  │  Discord Bot     │ ◄════════════► │  Moshi    │  │
│  │  (Python/Pycord) │  localhost:8998 │  Server   │  │
│  │                  │  Opus frames    │  (GPU)    │  │
│  └────────┬─────────┘                └───────────┘  │
│           │                                          │
└───────────┼──────────────────────────────────────────┘
            │ Discord Gateway + Voice
            ▼
       Discord Voice
```

Both the Moshi inference server and the Discord bot run inside a single Docker container. A lightweight supervisor process manages both.

## Prerequisites

- **NVIDIA GPU** with ≥16GB VRAM (tested on RTX 5090)
- **Docker** with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- **Discord Bot Token** from the [Discord Developer Portal](https://discord.com/developers/applications)
- **HuggingFace Token** — accept the [PersonaPlex model license](https://huggingface.co/nvidia/personaplex-7b-v1) first

## Quick Start

1. **Clone and configure:**
   ```bash
   git clone https://github.com/yourusername/personaplex-discord-bot.git
   cd personaplex-discord-bot
   cp .env.example .env
   # Edit .env with your tokens
   ```

2. **Build and run:**
   ```bash
   docker-compose up --build
   ```

3. **Use in Discord:**
   - `/join` — Bot joins your voice channel, starts listening
   - `/leave` — Bot disconnects
   - `/voice NATM1` — Change voice preset
   - `/prompt You are a pirate captain.` — Change system prompt

## Voice Presets

| Category | Female | Male |
|----------|--------|------|
| Natural  | NATF0, NATF1, NATF2, NATF3 | NATM0, NATM1, NATM2, NATM3 |
| Variety  | VARF0, VARF1, VARF2, VARF3, VARF4 | VARM0, VARM1, VARM2, VARM3, VARM4 |

## Prompting Guide

**Assistant:**
```
You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way.
```

**Customer Service:**
```
You work for [Company] which is a [type] and your name is [Name]. Information: [details].
```

**Casual:**
```
You enjoy having a good conversation.
```

## Limitations

- **One conversation at a time** — The Moshi server uses a global lock. Multi-guild support requires multiple server instances.
- **Single audio stream** — All speakers in the voice channel are heard by the AI (audio is mixed sequentially, not summed).

## License

MIT (bot code) — PersonaPlex model weights are under the [NVIDIA Open Model License](https://huggingface.co/nvidia/personaplex-7b-v1).
