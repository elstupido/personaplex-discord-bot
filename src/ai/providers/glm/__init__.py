"""
The PersonaPlex GLM Bridge Package.

WHY THIS PACKAGE EXISTS:
To keep the code clean, modular, and maintainable. The old monolithic 
glm.py has been deconstructed into functional sub-modules, while this 
__init__.py ensures that existing imports continue to work without modification.
"""

from .core import GLMBridge
from .stubs import Transcriber

__all__ = ["GLMBridge", "Transcriber"]
