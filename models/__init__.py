import numpy as np
if not hasattr(np, "float_"):
    np.float_ = np.float64

from .schemas import *
from .emotional_intelligence import EmotionalIntelligence
from .memory_system import MemorySystem
from .ai_engine import AIEngine
