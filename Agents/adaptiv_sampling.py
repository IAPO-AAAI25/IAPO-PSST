# epsilon_greedy_random_online.py  –  online ctx/arm sampling (random ctx; no rounds/batching)
# ------------------------------------------------------------------------------------------------
from __future__ import annotations
from typing import Dict, List, Tuple, Any
import numpy as np, random

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


class AdaptiveAgent(Agent):
    """Online ε-greedy / softmax agent with random context sampling (no rounds, no batching)."""

    # ---------------------------------------------------------------------
    def __init__(self,
                 epsilon: float   = 0.1,
                 action_mode: str = "epsilon",   # "epsilon" | "softmax"
                 tau: float       = 0.05,        # soft-max temperature (arms only)
                 strategy: str    = "mv") -> None:

        super().__init__("EpsGreedyRandomCtxOnline")
        if action_mode not in {"epsilon", "softmax"}:
            raise ValueError("action_mode must be 'epsilon' or 'softmax'")

        # hyper-parameters
        self.epsilon  = float(epsilon)
        self.mode     = action_mode
        self.tau      = float(tau)     # used only for arm softmax
        self.strategy = strategy

        # state (filled in by _init_structures)
        self._stats:  Dict[Tuple[Any, Arm], QStatistics] = {}
        self._active: Dict[Any, List[Arm]]               = {}
        self._all_arms:  List[Arm] = []
        self._arm_sizes: np.ndarray  = np.array([])      # N for every arm
        self.T_used: int = 0

    # ---------------------------- utilities ---------------------------------
    def _emp_mean(self, ctx: Any, arm: Arm) -> float:
        st = self._stats[(ctx, arm)]
        return st.mean if st.pulls else 0.0

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

    # --------------------------- initialisation -----------------------------
    def _init_structures(self, env: Any) -> None:
        self._all_arms  = [Arm(p, k)
                           for p in env.prompts
                           for k in range(1, env.N_max + 1)]
        self._arm_sizes = np.array([a.N for a in self._all_arms])

        for ctx in env._contexts:
            self._active[ctx] = self._all_arms.copy()
            for arm in self._all_arms:
                self._stats[(ctx, arm)] = QStatistics()

    # ------------------------------ train -----------------------------------
    def train(self, env: Any, budget: int, **kwargs) -> None:
        """Online loop: randomly pick a context, select an arm (ε-greedy/softmax),
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
            # compute means for feasible arms in the chosen context
            means = np.array([self._emp_mean(ctx_star, self._all_arms[i]) for i in feasible_idx])

            if self.mode == "epsilon":
                explore = (random.random() < self.epsilon)
                if explore:
                    idx = int(random.choice(feasible_idx))
                else:
                    idx = int(feasible_idx[int(means.argmax())])
            else:  # soft-max over feasible arms
                logits = means / (self.tau + 1e-12)
                logits -= logits.max()          # numerical stability
                probs = np.exp(logits)
                probs /= probs.sum()
                choice_local = int(np.random.choice(len(feasible_idx), p=probs))
                idx = int(feasible_idx[choice_local])

            arm_star = self._all_arms[idx]
            self._single_pull(env, arm_star)   # immediate execution & updates

        # -------- keep the single best arm per context --------------------
        for ctx in env._contexts:
            best = max(self._active[ctx], key=lambda a: self._emp_mean(ctx, a))
            self._active[ctx] = [best]

        print(f"Adaptive sampling tokens used: {self.T_used} / {budget}")

    # ----------------------------- predict ----------------------------------
    def predict(self, context: Any) -> Tuple[int, int]:
        active = self._active.get(context, [])
        return (active[0].prompt, active[0].N) if active else (0, 1)
