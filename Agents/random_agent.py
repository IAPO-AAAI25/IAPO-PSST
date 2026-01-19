# one_round_uniform_agent.py  –  Budget on P×Nmax, info shared to sub-arms
# -----------------------------------------------------------------------
from __future__ import annotations
from typing import Dict, List, Tuple, Any
import numpy as np
import math

try:
    from agents import Agent, Arm, QStatistics
except Exception:
    from Agents.agents import Agent, Arm, QStatistics   # fallback path




# ─────────────────────────────── Agent ─────────────────────────────────
class RandomAgent(Agent):

    def __init__(self, strategy: str = "mv") -> None:
        super().__init__("OneRoundUniform")
        self.strategy = strategy
        self._all_arms:      List[Arm] = []   # all (p, k)
        self.T_used: int = 0                  # total tokens actually spent

    def _init_structures(self, env: Any) -> None:
        """Create *all* sub-arms plus primary (p, N_max) arms."""
        self._all_arms     = [Arm(p, k)
                              for p in env.prompts
                              for k in range(1, env.N_max + 1)]
        
    def train(self, env: Any, budget: int, **kwargs) -> None:
        self._init_structures(env)
        
    def predict(self, context: Any) -> Tuple[int, int]:
        """Return the surviving (prompt, k) for `context` (or (0,1) fallback)."""
        idx = int(np.random.choice(np.arange(len(self._all_arms))))
        arm_star = self._all_arms[idx]
        return (arm_star.prompt, arm_star.N) 
