from abc import ABC, abstractmethod
import asyncio
import logging
from typing import Optional, Any

logger = logging.getLogger("bridge.base")

class BridgeBase(ABC):
    """
    Base class for AI model bridges.
    Handles the common logic for connecting to a model server and 
    exchanging audio/text data.
    """
    def __init__(self, model_type: str, sample_rate: int):
        self.model_type = model_type
        self._sample_rate = sample_rate
        self.running = False
        self.ws: Optional[Any] = None

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, value: int):
        self._sample_rate = value
        
    @abstractmethod
    async def connect(self):
        """Establish connection to the model server."""
        pass

    @abstractmethod
    async def send_audio_packet(self, pcm_tensor: bytearray):
        """Send a single packet of audio to the model."""
        pass

    @abstractmethod
    async def start_streaming(self):
        """Start the async loops for sending/receiving."""
        pass

    @abstractmethod
    async def close(self):
        """Clean up resources and close connection."""
        pass

    def is_healthy(self) -> bool:
        """Check if the connection is alive."""
        return self.running and self.ws is not None and not self.ws.closed
