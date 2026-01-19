# psst_agent.py  –  Prompt-Scaling via Sequential Trimming (plain PSST)
# --------------------------------------------------------------------
# Compatible with SyntheticBernoulliEnv and the Agent / Arm / QStatistics
# definitions you already use for SequentialHalving.
#
#  • No delayed-UCB, just classic sequential trimming.
#  • Uses *block-based* prefix reuse: a raw pull of size N yields
#       ⌊N / k⌋ i.i.d. virtual pulls for every sub-scale k (1…N).
#  • Allocation per round follows Eq. 7 of the paper (cost-aware uniform).
#
# Drop this file next to your other agents and:
#     from psst_agent import PSSTAgent
# --------------------------------------------------------------------
from __future__ import annotations
import math
from typing import Dict, List, Tuple, Any

import numpy as np

try:
    from agents import Agent, Arm, QStatistics          # your existing classes
except Exception:
    from Agents.agents import Agent, Arm, QStatistics   # fallback path


# --------------------------------------------------------------------------- #
# Helper: full block-based sample reuse                                        #
# --------------------------------------------------------------------------- #
def split_blocks(raw: np.ndarray) -> Dict[int, List[np.ndarray]]:
    """
    For an (N × D) raw outcome matrix, return a dictionary mapping
        k  →  list of non-overlapping blocks of length k
    so that each block is an i.i.d. sample for sub-scale k.

    Example:  N = 7
        k = 3  →  blocks of rows [0:3], [3:6]   (⌊7/3⌋ = 2 blocks)
        k = 2  →  blocks [0:2], [2:4], [4:6]    (⌊7/2⌋ = 3 blocks)
        k = 1  →  7 single-row blocks
    """
    N = raw.shape[0]
    out: Dict[int, List[np.ndarray]] = {}
    for k in range(1, N + 1):
        n_blocks = N // k
        if n_blocks == 0:
            continue
        out[k] = [raw[j * k : (j + 1) * k] for j in range(n_blocks)]
    return out


# --------------------------------------------------------------------------- #
# PSST Agent                                                                  #
# --------------------------------------------------------------------------- #
class TRAgent(Agent):
    """Plain Prompt-Scaling via Sequential Trimming for SyntheticBernoulliEnv."""

    def __init__(self, strategy: str = "mv", h:str  = "one") -> None:
        """
        Parameters
        ----------
        strategy : {"mv", "bon", "ia"}
            Which aggregator should env.get_utility() use when turning
            raw outcome matrices into scalar rewards.
        """
        super().__init__("PSST")
        self.strategy = strategy
        self.h = h
        self._stats:  Dict[Tuple[Any, Arm], QStatistics] = {}
        self._active: Dict[Any, List[Arm]]               = {}
        self._arms:   List[Arm]                          = []
        self._rounds: int                                = 0
        self.env_ref = None  # only for fallback in predict()
        self.T = 0

    # ------------------------------------------------------------------ #
    # Initialisation                                                     #
    # ------------------------------------------------------------------ #
    def _init_structures(self, env: Any) -> None:
        self.env_ref = env
        self._arms = [Arm(p,1) for p in env.prompts ]

        for ctx in env._contexts:
            self._active[ctx] = self._arms.copy()
            for arm in self._arms:
                self._stats[(ctx, arm)] = QStatistics()

        self._rounds = max(1, math.ceil(math.log2(len(self._arms))))


    # ------------------------------------------------------------------ #
    # Allocation plan (Eq. 7)                                            #
    # ------------------------------------------------------------------ #
    def _allocation_plan(self, round_budget: int) -> List[Tuple[Arm, int]]:
        """
        One global plan for the round:
            * For each prompt, take its **highest** active scale across contexts.
            * Allocate pulls ∝ arm.N, with ≥1 pull each, so that the sum
              of pulls×N ≈ round_budget.
        Returns
        -------
        List of (Arm, pulls) tuples.
        """
        # Highest active scale per prompt (union of contexts)
        highest: Dict[int, Arm] = {}
        for ctx_active in self._active.values():
            for arm in ctx_active:
                if arm.prompt not in highest or arm.N > highest[arm.prompt].N:
                    highest[arm.prompt] = arm

        total_cost = sum(a.N for a in highest.values()) or 1
        plan: List[Tuple[Arm, int]] = []
        pulls = max(1, int(round_budget / total_cost))
        for arm in highest.values():
            plan.append((arm, pulls))
        return plan

    # ------------------------------------------------------------------ #
    # Training                                                           #
    # ------------------------------------------------------------------ #
    def train(self, env: Any, budget: int) -> None:
        """
        Parameters
        ----------
        env     : SyntheticBernoulliEnv
        budget  : int
            Total token budget.  One pull of scale N costs N tokens.
        """
        self._init_structures(env)
        budget_per_round = max(1, budget // self._rounds)

        for _ in range(self._rounds):
            # ---------------- 1.  Data collection ----------------------
            plan = self._allocation_plan(budget_per_round)
            for arm, pulls in plan:
                for _ in range(pulls):
                    raw_N = env.pull(arm.prompt, arm.N, split="train")
                    self.T +=arm.N
                    block_dict = split_blocks(raw_N)

                    # Evaluate *every* context and *every* sub-scale block
                    for ctx in env._contexts:
                        for k, block_list in block_dict.items():
                            sub_arm = Arm(arm.prompt, k)
                            if sub_arm not in self._active[ctx]:
                                continue  # already trimmed
                            for raw_k in block_list:
                                r = env.get_utility(raw_k, ctx, self.strategy)
                                st = self._stats[(ctx, sub_arm)]
                                st.pulls      += 1
                                st.reward_sum += r

            # ---------------- 2.  Sequential trimming ------------------
            for ctx in env._contexts:
                active = self._active[ctx]
                if len(active) <= 1:
                    continue
                ranked = sorted(
                    active, key=lambda a: self._stats[(ctx, a)].mean, reverse=True
                )
                survivors = ranked[: math.ceil(len(ranked) / 2)]
                self._active[ctx] = survivors

        #print (f"Final active arms: {self._active}")
        print(f"TRIPLE agent Total T: ", self.T)

    # ------------------------------------------------------------------ #
    # Prediction                                                         #
    # ------------------------------------------------------------------ #
    def predict(self, context: Any) -> Tuple[int, int]:
        """
        Return
        ------
        (prompt_id, N)
            Best surviving arm under empirical mean.  Falls back to
            (0,1) if the context was never seen in training.
        """
        active = self._active.get(context, [])
        if not active:
            return (0, 1)  # safe default
        best = max(active, key=lambda a: self._stats[(context, a)].mean)
        if self.h == "random":
            return (best.prompt, np.random.randint(1, self.env_ref.N_max + 1))
        else:
            return (best.prompt, 1)
