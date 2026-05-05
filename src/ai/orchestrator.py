"""
WHY THIS FILE EXISTS:
The central brain of the PersonaPlex AI logic. This module manages 
conversation history, user registration, and the assembly of complex 
message payloads for the inference server.

WHY THE HISTORY LIMIT?
Large language models have finite context windows. By enforcing a strict 
history limit here, we prevent VRAM overflow and ensure that the most 
recent (and relevant) parts of the conversation are prioritized.
"""
from __future__ import annotations
import asyncio
import time
from typing import List, Dict, Any

from core.logger import setup_logger

logger = setup_logger("ai.orchestrator")

class Colors:
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    GRAY = "\033[90m"
    RESET = "\033[0m"
    BOLD = "\033[1m"

class Orchestrator:
    """
    The central brain of the PersonaPlex bot.
    Manages user sessions and conversation history.
    Outputs a structured message list for the GLM server.
    """
    def __init__(self, history_limit: int = 10):
        self.history_limit = history_limit
        self.history: List[Dict[str, Any]] = []  # Unified history list
        self.user_map: Dict[str, str] = {}       # user_id -> display_name mapping
        
        # System instructions
        # CRITICAL: This exact prompt is required to trigger GLM-4-Voice's audio generation mode.
        self.system_message = (
            "User will provide you with a speech instruction. Do it step by step. "
            "First, think about the instruction and provide a brief plan, "
            "then follow the instruction and respond directly to the user."
        )
        logger.info(f"Initialized with history_limit={history_limit}")

    def register_user(self, user_id: str, display_name: str):
        self.user_map[user_id] = display_name

    def update_user(self, user_id: str, display_name: str):
        """Alias for register_user used by the voice sink."""
        self.register_user(user_id, display_name)

    def assemble_payload(self, incoming_audio_segments: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Takes raw audio segments and formats them into a GLM-4-Voice message list.
        If incoming_audio_segments is None, returns just the system prompt for initialization.
        """
        messages = []
        
        # 1. System Message
        messages.append({
            "role": "system",
            "content": self.system_message
        })

        if incoming_audio_segments is None:
            return messages
        
        # 2. History (limited)
        messages.extend(self.history[-self.history_limit:])
        
        # 3. Current User Turn
        current_audios = []
        active_users = set()
        
        for segment in incoming_audio_segments:
            user_id = segment.get('user_id', 'unknown')
            user_name = self.user_map.get(user_id, f"User {user_id}")
            active_users.add(user_name)
            current_audios.append(segment['audio'])
        
        if current_audios:
            user_list = ", ".join(active_users) if active_users else "Someone"
            logger.info(f"Forming Prompt | Users: {user_list} | Segments: {len(current_audios)}")

            payload = {
                "role": "user",
                "content": "",
                "audio_tokens": None,
                "audio_segments": incoming_audio_segments,
            }
            messages.append(payload)

        return messages

    def add_user_turn(self, audio_tokens: list = None):
        """Explicitly record a user turn in history. Call this after sending to the model."""
        self.history.append({"role": "user", "content": "", "audio_tokens": audio_tokens or []})
        if len(self.history) > self.history_limit * 2:
            self.history = self.history[-self.history_limit * 2:]

    def add_assistant_response(self, text: str, audio_tokens=None):
        self.history.append({"role": "assistant", "content": text, "audio_tokens": audio_tokens or []})
        # Keep history clean
        if len(self.history) > self.history_limit * 2:
            self.history = self.history[-self.history_limit * 2:]

# Global instance
orchestrator = Orchestrator()
