from __future__ import annotations
"""
Synthetic Bernulii Environment
"""
from dataclasses import dataclass
from typing import Dict, List, Tuple, Sequence, Literal, Optional
import numpy as np

# ---------------------------------------------------------------------------
# Data structures ------------------------------------------------------------
# ---------------------------------------------------------------------------

ObjectiveVector = Tuple[int, float]  # (correctness ∈ {0,1}, negative_cost)


@dataclass(frozen=True)
class CostContext:
    """Context is *only* the cost coefficient level (low, mid, high)."""

    level: Literal["low", "mid", "high"]
    # Objective weights – keep constant across contexts; cost already carries
    # the coefficient so the weight on cost is simply 1.
    weights: Tuple[float, float] = (1.0, -1.0)  # (w_correct, w_cost)

    def __iter__(self):
        return iter((self.level, self.weights))


# ---------------------------------------------------------------------------
# Environment ----------------------------------------------------------------
# ---------------------------------------------------------------------------

class SyntheticBernoulliEnv:
    """Bernoulli reward environment with *difficulty tiers* & *cost contexts*.

    Parameters
    ----------
    success_probs : Dict[str, np.ndarray]
        Mapping of tier → probability table with shape `(P, |X_t|)`, where *P*
        is the number of prompts and `|X_t|` is the number of examples in that
        tier.
    tier_distribution : Sequence[float], length = 3
        Probability of sampling a *difficulty tier* (easy/medium/hard) when a
        query is drawn.
    cost_per_prompt : Sequence[float], length = P
        Base cost per completion for each prompt (e.g. average token length).
    cost_coeffs : Dict[str, float]
        Multiplicative factors for contexts *low*, *mid*, *high*.
    context_distribution : Sequence[float], length = 3
        Sampling distribution over cost contexts (low/mid/high).
    N_max : int, default = 16
        Maximum allowed `N` in Best‑of‑N or Majority Voting.
    rng : int | np.random.Generator | None, optional
        RNG seed or generator.
    """

    TIERS = ("easy", "medium", "hard")
    CONTEXTS = ("low", "mid", "high")

    # ---------------------------------------------------------------------
    def __init__(
        self,
        success_probs: Dict[str, np.ndarray],
        success_probs_test: Dict[str, np.ndarray],
        tier_distribution: Sequence[float] = (1 / 3, 1 / 3, 1 / 3),
        cost_per_prompt: Sequence[float] | np.ndarray = (1.0,),
        cost_coeffs: Dict[str, float] | None = None,
        context_distribution: Sequence[float] = (1 / 3, 1 / 3, 1 / 3),
        N_max: int = 16,
        rng: Optional[int | np.random.Generator] = None,
    ) -> None:
        # ------------------ validate difficulty tables --------------------
        missing = [t for t in self.TIERS if t not in success_probs]
        if missing:
            raise ValueError(f"success_probs missing tiers: {missing}")

        num_prompts = None
        for tier, table in success_probs.items():
            if table.ndim != 2:
                raise ValueError(f"success_probs['{tier}'] must be 2-D (Px|X_t|)")
            if num_prompts is None:
                num_prompts = table.shape[0]
            elif table.shape[0] != num_prompts:
                raise ValueError("All tier tables must have the same #prompts")
            if not ((table >= 0) & (table <= 1)).all():
                raise ValueError("Success probabilities must be in [0,1]")
        

        self.num_prompts = num_prompts  # type: ignore[arg-type]
        self.success_probs = {t: np.asarray(tbl, dtype=float) for t, tbl in success_probs.items()}
        self.success_probs_test = {t: np.asarray(tbl, dtype=float) for t, tbl in success_probs_test.items()}
        
        self.tier_distribution = np.asarray(tier_distribution, dtype=float)
        if len(self.tier_distribution) != 3 or not np.isclose(self.tier_distribution.sum(), 1.0):
            raise ValueError("tier_distribution must have length 3 and sum to 1")

        # ------------------------ cost parameters -------------------------
        self.cost_per_prompt = np.asarray(cost_per_prompt, dtype=float)
        if self.cost_per_prompt.shape[0] != self.num_prompts:
            raise ValueError("cost_per_prompt length must equal #prompts")

        self.cost_coeffs = cost_coeffs or {"low": 0.0, "mid": 0.5, "high": 1.0}
        if set(self.CONTEXTS) != set(self.cost_coeffs):
            raise ValueError("cost_coeffs must provide 'low', 'mid', and 'high'")

        self.context_distribution = np.asarray(context_distribution, dtype=float)
        if len(self.context_distribution) != 3 or not np.isclose(self.context_distribution.sum(), 1.0):
            raise ValueError("context_distribution must have length 3 and sum to 1")

        # --------------- miscellaneous -----------------------------------
        self.prompts = [i for i in range(self.num_prompts)]
        if N_max < 1:
            raise ValueError("N_max must be ≥ 1")
        self.N_max = N_max
        self.rng = np.random.default_rng(rng)

        # Pre‑build context objects
        self._contexts: List[CostContext] = [CostContext(level,(1.0,-1.0*self.cost_coeffs[level])) for level in self.CONTEXTS]
        self._level2ctx = {ctx.level: ctx for ctx in self._contexts}
        self.prompt_text = []
        for p in self.prompts:
            text = f"Prompt {p}:\n"
            e = self.success_probs["easy"][p].mean()
            m = self.success_probs["medium"][p].mean()
            h = self.success_probs["hard"][p].mean()
            text = text + f"Easy Performance: {e:.2f}\nMedium Performance: {m:.2f}\nHard Performance: {h:.2f}\n"
            text = text+f"Generation Cost: {self.cost_per_prompt[p]:.2f}"
            self.prompt_text.append(text)
            #print(text)
    
    
    # ---------------------------------------------------------------------
    # Sampling helpers -----------------------------------------------------
    # ---------------------------------------------------------------------

    def sample_context(self) -> CostContext:
        """Draw a cost context (low/mid/high) according to `context_distribution`."""
        idx = self.rng.choice(3, p=self.context_distribution)
        return self._contexts[idx]

    def _sample_query(self, split:str = "train") -> Tuple[str, int]:
        """Sample a (tier, example_index) pair - the *query* x."""
        tier_idx = self.rng.choice(3, p=self.tier_distribution)
        tier = self.TIERS[tier_idx]
        if split == "train":
            num_examples = self.success_probs[tier].shape[1]
        else:
            num_examples = self.success_probs_test[tier].shape[1]
        ex_idx = self.rng.integers(num_examples)
        return tier, ex_idx

    # ------------------------------------------------------------------
    def pull(
        self,
        prompt_id: int,
        N: int,
        split: str = "train"
    ) -> Tuple[np.ndarray, float] | float:
        """Simulate `N` completions for *prompt_id* under *context*.

        The difficulty tier and specific example are *randomised per pull* to
        emulate the hidden query distribution.
        """
        if not (0 <= prompt_id < self.num_prompts):
            raise IndexError("prompt_id out of range")
        if not (1 <= N <= self.N_max):
            raise ValueError("N must be in [1, N_max]")

        # Draw query (difficulty tier + example) -------------------------
        tier, ex_idx = self._sample_query(split)
        if split == "train":
            p_success = self.success_probs[tier][prompt_id, ex_idx]
        else:
            p_success = self.success_probs_test[tier][prompt_id, ex_idx]
        # Vectorised Bernoulli -----------------------------------------
        successes = self.rng.random(N) < p_success  # shape = (N,)
        correctness = successes.astype(int)

        # Cost vector (negative) ---------------------------------------
        per_completion_cost = self.cost_per_prompt[prompt_id]
        cost = np.full(N, per_completion_cost, dtype=float)

        raw = np.vstack([correctness, cost]).T  # shape = (N, 2)

        
        return raw

    # ------------------------------------------------------------------
    # Aggregators (MV / BoN / IA) ----------------------------------------
    # ------------------------------------------------------------------

    def get_utility(self, raw: np.ndarray, context: CostContext, strategy: str) -> float:
        if strategy == "mv":
            return self.majority_vote(raw, context)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")


    @staticmethod
    def majority_vote(
        raw: np.ndarray,
        context: CostContext,
        rng: Optional[np.random.Generator] = None,
    ) -> float:
        """
        Majority-vote utility with *random* tie-break:

            • strict majority ⇒ 1
            • tie (N even, wins == N/2) ⇒ Bernoulli(0.5)  ∈ {0,1}
            • strict minority ⇒ 0
        """
        rng = rng or np.random.default_rng()

        correctness = raw[:, 0]
        cost        = raw[:, 1].sum()
        wins        = correctness.sum()
        N           = len(correctness)

        if wins > N / 2:                             # clear majority
            majority = 1
        elif N % 2 == 0 and wins == N / 2:           # tie → coin-flip
            majority = int(rng.random() < 0.5)       # 0 or 1 with equal prob.
        else:                                        # minority
            majority = 0

        return context.weights[0] * majority + context.weights[1] * cost 

   

    # ------------------------------------------------------------------
    # Diagnostics --------------------------------------------------------
    # ------------------------------------------------------------------

    def summary(self) -> None:
        print("Synthetic Bernoulli Environment (v2)")
        print("-----------------------------------")
        for tier in self.TIERS:
            tbl = self.success_probs[tier]
            print(f"Tier \u2013 {tier}:  |X| = {tbl.shape[1]}")
        print("#Prompts:", self.num_prompts)
        print("Cost per prompt:", self.cost_per_prompt)
        print("Cost coefficients:", self.cost_coeffs)
        print("Tier distribution:", self.tier_distribution)
        print("Context distribution:", self.context_distribution)
        print("N_max:", self.N_max)

