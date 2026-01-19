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


# ───── Helper: split a raw N×D matrix into sub-scale blocks ────────────
def split_blocks(raw: np.ndarray) -> Dict[int, List[np.ndarray]]:
    """Return {k: [non-overlapping blocks of length k]} for k = 1…N."""
    N = raw.shape[0]
    out: Dict[int, List[np.ndarray]] = {}
    for k in range(1, N + 1):
        m = N // k
        if m:
            out[k] = [raw[j * k : (j + 1) * k] for j in range(m)]
    return out


# ─────────────────────────────── Agent ─────────────────────────────────
class UniformAgent(Agent):
    """
    One-round uniform allocation:
      • budget spent only on (prompt, N_max) arms;
      • rewards shared to every (prompt, k) sub-arm and every context;
      • after the round, keep the single best arm per context.
    """

    def __init__(self, strategy: str = "mv") -> None:
        super().__init__("OneRoundUniform")
        self.strategy = strategy
        self._stats:  Dict[Tuple[Any, Arm], QStatistics] = {}
        self._active: Dict[Any, List[Arm]] = {}
        self._all_arms:      List[Arm] = []   # all (p, k)
        self._primary_arms:  List[Arm] = []   # only (p, N_max)
        self.T_used: int = 0                  # total tokens actually spent

    # ------------------------------------------------------------------ #
    # Initialise per-context structures                                  #
    # ------------------------------------------------------------------ #
    def _init_structures(self, env: Any) -> None:
        """Create *all* sub-arms plus primary (p, N_max) arms."""
        self._all_arms     = [Arm(p, k)
                              for p in env.prompts
                              for k in range(1, env.N_max + 1)]
        self._primary_arms = [Arm(p, env.N_max) for p in env.prompts]

        for ctx in env._contexts:
            self._active[ctx] = self._all_arms.copy()
            for arm in self._all_arms:
                self._stats[(ctx, arm)] = QStatistics()

    # ------------------------------------------------------------------ #
    # Training – exactly one uniform round                               #
    # ------------------------------------------------------------------ #
    def train(self, env: Any, budget: int, **kwargs) -> None:
        """
        Each primary arm (p, N_max) is pulled
            pulls_each = floor(T / (P · N_max))   times.
        """
        self._init_structures(env)
        P          = max(1, len(self._primary_arms))
        pulls_each = max(1, budget // (P * env.N_max))

        for arm in self._primary_arms:
            for _ in range(pulls_each):
                raw = env.pull(arm.prompt, arm.N, split="train")  # cost N_max
                self.T_used += arm.N
                block_dict = split_blocks(raw)

                # share with *all* contexts and *all* sub-arms
                for ctx in env._contexts:
                    for k, blocks in block_dict.items():
                        sub_arm = Arm(arm.prompt, k)
                        st = self._stats[(ctx, sub_arm)]
                        for raw_k in blocks:
                            r = env.get_utility(raw_k, ctx, self.strategy)
                            st.pulls      += 1
                            st.reward_sum += r

        # keep only the empirical-best arm per context
        for ctx in env._contexts:
            best = max(self._active[ctx],
                       key=lambda a: self._stats[(ctx, a)].mean)
            self._active[ctx] = [best]

        print(f"Uniform agent tokens used = {self.T_used} / budget {budget}")

    # ------------------------------------------------------------------ #
    # Prediction                                                         #
    # ------------------------------------------------------------------ #
    def predict(self, context: Any) -> Tuple[int, int]:
        """Return the surviving (prompt, k) for `context` (or (0,1) fallback)."""
        active = self._active.get(context, [])
        #print(active)
        return (active[0].prompt, active[0].N) if active else (0, 1)
