# psst_agent.py  –  Prompt-Scaling via Sequential Trimming (plain PSST)
# --------------------------------------------------------------------
# Compatible with SyntheticBernoulliEnv and the Agent / Arm / QStatistics
# definitions you already use for SequentialHalving.
#
#  • Two-phase training:
#      (1) burn-in: uniform sampling at N=1 over all prompts and contexts
#      (2) pruning: keep top-K prompts per context, then run trimming up to N_max
#  • Block-based prefix reuse remains for Phase 2.
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
class PSSTHAgent(Agent):
    """Plain Prompt-Scaling via Sequential Trimming for SyntheticBernoulliEnv."""

    def __init__(self, strategy: str = "mv", burn: float = 0.2, K: int = 1) -> None:
        """
        Parameters
        ----------
        strategy : {"mv", "bon", "ia"}
            Which aggregator should env.get_utility() use when turning
            raw outcome matrices into scalar rewards.
        burn : float in [0,1]
            Fraction of total token budget to use for Phase 1 (uniform N=1).
        """
        super().__init__("PSST")
        self.strategy = strategy
        self.burn = burn
        self.K = K  # number of prompts to keep per context after burn-in
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
        self._arms = [Arm(p, N) for p in env.prompts for N in range(1, env.N_max + 1)]

        for ctx in env._contexts:
            self._active[ctx] = self._arms.copy()
            for arm in self._arms:
                self._stats[(ctx, arm)] = QStatistics()

        # _rounds will be set after Top-K pruning (Phase 2 size depends on pruning)
        self._rounds = 1

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
    # Phase 1: Uniform N=1 sampling with cross‑context sharing           #
    # ------------------------------------------------------------------ #
    def _burn_in(self, env: Any, burn_tokens: int) -> int:
        """
        Uniformly sample prompts with N=1 and reuse each raw sample across
        all contexts (same sharing pattern as Phase 2). Each pull costs 1 token.

        Returns the number of tokens actually spent.
        """
        if burn_tokens <= 0:
            return 0

        prompts: List[int] = list(env.prompts)
        if not prompts:
            return 0

        spent = 0
        idx = 0
        L = len(prompts)

        while spent < burn_tokens:
            p = prompts[idx]
            idx = (idx + 1) % L

            # Pull once at N=1 for this prompt
            raw = env.pull(p, 1, split="train")  # shape (1, D)

            # Share this single draw across *all* contexts
            for ctx in env._contexts:
                r = env.get_utility(raw, ctx, self.strategy)
                st = self._stats[(ctx, Arm(p, 1))]
                st.pulls      += 1
                st.reward_sum += r

            spent += 1      # 1 token spent (N=1)
            self.T += 1     # track total token usage

        return spent


    # ------------------------------------------------------------------ #
    # Phase 1→2 transition: Top-K prompts per context                    #
    # ------------------------------------------------------------------ #
    def _prune_topK(self, env: Any, K: int) -> None:
        """
        For each context, compute empirical means at N=1 for each prompt and
        keep only the top-K prompts. After pruning, all scales 1..N_max for
        those prompts remain active in that context.
        """
        # Build helper: per-context means at N=1
        for ctx in env._contexts:
            means: List[Tuple[int, float]] = []
            for p in env.prompts:
                st = self._stats[(ctx, Arm(p, 1))]
                # Prefer seen prompts; unseen get very low mean so they are dropped
                m = (st.mean if getattr(st, "pulls", 0) > 0 else -1e30)
                means.append((p, m))
                self._stats[(ctx, Arm(p, 1))] = QStatistics()  # reset for Phase 2
            means.sort(key=lambda t: t[1], reverse=True)
            keep_prompts = set([p for p, _ in means[: max(1, min(K, len(means)))]])
            # Filter active arms in this context to those prompts (all scales)
            self._active[ctx] = [a for a in self._active[ctx] if a.prompt in keep_prompts]

        # Recompute number of rounds for Phase 2 based on max active set size
        max_active = K*self.env_ref.N_max
        self._rounds = max(1, math.ceil(math.log2(max_active)))

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
        K       : int
            Number of prompts to keep per context after burn-in.
        """
        self._init_structures(env)
        #print("HHHHHHHHHHHHHHHHHHHHHHH")
        # -------- Phase 1: uniform N=1 sampling --------
        burn_tokens = int(max(0, min(1, self.burn)) * budget)
        spent_burn = self._burn_in(env, burn_tokens)

        # -------- Top-K pruning (per context) ----------
        self._prune_topK(env, self.K)

        # -------- Phase 2: sequential trimming ---------
        remaining = max(0, budget - spent_burn)
        if remaining <= 0:
            # No budget left for Phase 2; just report current bests
            for ctx in env._contexts:
                active = self._active.get(ctx, [])
                if not active:
                    continue
                best = max(active, key=lambda a: self._stats[(ctx, a)].mean)
                #print(ctx, best)
            #print("Total T: ", self.T)
            return

        budget_per_round = max(1, remaining // self._rounds)

        for _ in range(self._rounds):
            # ---------------- 1.  Data collection ----------------------
            plan = self._allocation_plan(budget_per_round)
            for arm, pulls in plan:
                for _ in range(pulls):
                    raw_N = env.pull(arm.prompt, arm.N, split="train")
                    self.T += arm.N
                    block_dict = split_blocks(raw_N)

                    # Evaluate *every* context and *every* sub-scale block
                    for ctx in env._contexts:
                        for subN, block_list in block_dict.items():
                            sub_arm = Arm(arm.prompt, subN)
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

        for ctx in env._contexts:
            active = self._active.get(ctx, [])
            if not active:
                continue
            best = max(active, key=lambda a: self._stats[(ctx, a)].mean)
            #print(ctx, best)
        print(f"PSST+K{self.K} agent Total T: ", self.T)

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
        return (best.prompt, best.N)
