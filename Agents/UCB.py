# ucb_random_online.py  –  online ctx/arm sampling with UCB (random ctx; no rounds/batching)
# ------------------------------------------------------------------------------------------------
from __future__ import annotations
from typing import Dict, List, Tuple, Any
import numpy as np, random, math

try:
    from agents import Agent, Arm, QStatistics
except Exception:
    from Agents.agents import Agent, Arm, QStatistics        # fallback


# ------------------------------ helpers -----------------------------------------
def split_blocks(raw: np.ndarray) -> Dict[int, List[np.ndarray]]:
    """Split raw completions into every possible block size (1 … N)."""
    N = raw.shape[0]
    return {k: [raw[j*k:(j+1)*k] for j in range(N // k)]
            for k in range(1, N + 1) if N // k}


class UCBAgent(Agent):
    """Online UCB agent with random context sampling (no rounds, no batching)."""

    # ---------------------------------------------------------------------
    def __init__(self,
                 ucb_c: float   = 2.0,      # exploration coefficient
                 strategy: str  = "mv") -> None:

        super().__init__("UCBRandomCtxOnline")
        self.ucb_c    = float(ucb_c)
        self.strategy = strategy

        # state (filled in by _init_structures)
        self._stats:  Dict[Tuple[Any, Arm], QStatistics] = {}
        self._active: Dict[Any, List[Arm]]               = {}
        self._all_arms:  List[Arm] = []
        self._arm_sizes: np.ndarray  = np.array([])      # N for every arm
        self._ctx_total: Dict[Any, int] = {}             # total virtual pulls per context
        self.T_used: int = 0

    # ---------------------------- utilities ---------------------------------
    def _emp_mean(self, ctx: Any, arm: Arm) -> float:
        st = self._stats[(ctx, arm)]
        return st.mean if st.pulls else 0.0

    def _ucb_index(self, ctx: Any, arm: Arm) -> float:
        st = self._stats[(ctx, arm)]
        n  = st.pulls
        t  = max(1, self._ctx_total.get(ctx, 0))
        if n == 0:
            return float("inf")
        bonus = self.ucb_c * math.sqrt(max(0.0, math.log(t) / n))
        return st.mean + bonus

    def _single_pull(self, env: Any, arm: Arm) -> None:
        """Pull `arm`, then share its completions across *all* contexts."""
        raw = env.pull(arm.prompt, arm.N, split="train")
        self.T_used += arm.N
        blocks_by_k = split_blocks(raw)
        for ctx in env._contexts:
            for k, blocks in blocks_by_k.items():
                sub   = Arm(arm.prompt, k)
                stats = self._stats[(ctx, sub)]
                for blk in blocks:
                    r = env.get_utility(blk, ctx, self.strategy)
                    stats.pulls      += 1
                    stats.reward_sum += r
                    # keep per-context total pulls up-to-date for UCB's log(t)
                    self._ctx_total[ctx] = self._ctx_total.get(ctx, 0) + 1

    # --------------------------- initialisation -----------------------------
    def _init_structures(self, env: Any) -> None:
        self._all_arms  = [Arm(p, k)
                           for p in env.prompts
                           for k in range(1, env.N_max + 1)]
        self._arm_sizes = np.array([a.N for a in self._all_arms])

        for ctx in env._contexts:
            self._active[ctx] = self._all_arms.copy()
            self._ctx_total[ctx] = 0
            for arm in self._all_arms:
                self._stats[(ctx, arm)] = QStatistics()

    # ------------------------------ train -----------------------------------
    def train(self, env: Any, budget: int, **kwargs) -> None:
        """Online loop: randomly pick a context, select an arm via UCB,
        pull immediately, update Q; repeat until budget is exhausted."""
        self._init_structures(env)

        ctxs = list(env._contexts)
        if not ctxs:
            return

        while True:
            remaining = budget - self.T_used
            feasible_idx = np.nonzero(self._arm_sizes <= remaining)[0]
            if feasible_idx.size == 0:
                break  # no arm fits the remaining budget

            ctx_star = random.choice(ctxs)

            # UCB over feasible arms for the sampled context
            best_idx = None
            best_val = -float("inf")
            for i in feasible_idx:
                a = self._all_arms[i]
                val = self._ucb_index(ctx_star, a)
                # random tie-break
                if val > best_val or (val == best_val and random.random() < 0.5):
                    best_val = val
                    best_idx = i

            arm_star = self._all_arms[int(best_idx)]
            self._single_pull(env, arm_star)   # immediate execution & updates

        # -------- keep the single best arm per context --------------------
        for ctx in env._contexts:
            best = max(self._active[ctx], key=lambda a: self._emp_mean(ctx, a))
            self._active[ctx] = [best]

        print(f"UCB agent sample used: {self.T_used} / {budget}")

    # ----------------------------- predict ----------------------------------
    def predict(self, context: Any) -> Tuple[int, int]:
        active = self._active.get(context, [])
        return (active[0].prompt, active[0].N) if active else (0, 1)
