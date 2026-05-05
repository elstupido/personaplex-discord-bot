from abc import ABC, abstractmethod
import asyncio
from dataclasses import dataclass, field
from typing import Optional, Any, Dict, List, Union, Generator, AsyncGenerator
from core.logger import setup_logger
import time

logger = setup_logger("ai.stupid_base")

# ---------------------------------------------------------------------------
# THE STUPID PRIMITIVES (Phase 1: Atomic Foundation)
# ---------------------------------------------------------------------------

@dataclass
class AcousticContext:
    """
    WHY THIS EXISTS:
    The 'Passport' for audio packets. It carries the physical metadata 
    (Sample Rate, User ID, Trace ID) through the data river.
    
    WHY NOT JUST DICT?
    Because types are love, and attribute access is faster than hash lookups. 
    Also, we need a standard way to track the 50Hz timing across experts.
    """
    sample_rate: int = 48000
    user_id: Optional[int] = None
    trace_id: str = field(default_factory=lambda: str(time.time_ns()))
    arrival_time: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StupidData:
    """
    WHY THIS EXISTS:
    The universal container for the 'Atomic Particles' on the conveyor belt. 
    It can carry raw PCM, text, or even 'Ghost Tokens' for speculation.
    """
    content: Any
    context: AcousticContext
    type: str = "pcm" # pcm, text, tokens, signal

class StupidStep(ABC):
    """
    WHY THIS EXISTS:
    The 'Expert' interface. Every model, resampler, or filter is a StupidStep.
    It takes an input, does a 'Transformation', and yields an output.
    
    WHY YIELD?
    Because a single input (Voice) might yield multiple outputs (Text + Sentiment + Audio).
    Generator patterns prevent memory bloat and keep the river flowing.
    """
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    async def process(self, data: StupidData) -> AsyncGenerator[StupidData, None]:
        """Transform the data. THE GIL IS YOUR ENEMY. Be fast or get smitten."""
        pass

@dataclass
class StupidJob:
    """
    WHY THIS EXISTS:
    A declarative 'Recipe' for the data river. It tells the StupidRunner 
    which steps to execute and in what order.
    
    WHY RECURSIVE?
    Because a StupidJob can contain other StupidJobs. This is the 
    'Recursive Power of the Sigil' that allows DAGs to branch at runtime.
    """
    steps: List[Union[str, 'StupidJob']]
    data: StupidData
    sigils: Dict[str, Any] = field(default_factory=dict)

class StupidRegistry:
    """
    WHY THIS EXISTS:
    The 'Yellow Pages' of Experts. Instead of hard-coding imports, 
    we register experts here by a string ID.
    
    WHY? 
    Because 'No Magic Strings' means we explicitly map strings to classes.
    It allows for lazy-loading and easy swapping of ASR/TTS backends.
    """
    _experts: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str):
        def wrapper(expert_cls: type):
            cls._experts[name] = expert_cls
            return expert_cls
        return wrapper

    @classmethod
    def get_expert(cls, name: str) -> type:
        if name not in cls._experts:
            raise ValueError(f"Expert '{name}' not found in the registry. DID YOU UPDATE THE COMMENT CLAUDE??")
        return cls._experts[name]

# ---------------------------------------------------------------------------
# LEGACY BRIDGE BASE (To be wrapped/deprecated)
# ---------------------------------------------------------------------------

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
