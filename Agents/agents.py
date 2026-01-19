from __future__ import annotations
"""
agents.py
==========
Reference implementation of agents for Inference-Aware Prompt Optimization (IAPO).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple, Callable
import math
import random
import itertools
import numpy as np

# ---------------------------------------------------------------------------
# Basic data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Arm:
    """Unique identifier of a (prompt, N) pair."""
    prompt: str
    N: int

@dataclass
class QStatistics:
    """Stores estimates and counts for a single arm within a context."""
    pulls: int = 0
    reward_sum: float = 0.0
     
    @property
    def mean(self) -> float:
        return self.reward_sum / max(1, self.pulls)

    

# ---------------------------------------------------------------------------
# Abstract Agent base class
# ---------------------------------------------------------------------------

class Agent:
    def __init__(self, name: str):
        self.name = name

    def train(self, env: Any, budget: int, **kwargs):
        raise NotImplementedError

    def predict(self, context: Any) -> Tuple[str, int]:
        raise NotImplementedError


