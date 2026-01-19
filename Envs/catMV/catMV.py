from __future__ import annotations
"""
Synthetic *Categorical* Environment (single tier)
-------------------------------------------------
Each (prompt, query) has a categorical distribution over C answer choices.
Majority‑vote utility is computed against the ground‑truth answer.

Usage
-----
env = SyntheticCategoricalEnv(cat_probs, answers, cat_probs_test, answers_test,
                              cost_per_prompt=..., ...)
raw = env.pull(prompt_id=0, N=8)         # raw shape (N, 3)
ctx  = env.sample_context()
u    = env.get_utility(raw, ctx, strategy="mv")
"""
from dataclasses import dataclass
from typing import Tuple, Sequence, Literal, Optional
import numpy as np

@dataclass(frozen=True)
class CostContext:
    level: Literal["low", "mid", "high"]
    weights: Tuple[float, float] = (1.0, -1.0)  # (w_correct, w_cost)
    def __iter__(self):
        return iter((self.level, self.weights))


class CategoricalMVEnv:
    CONTEXTS = ("low", "mid", "high")

    def __init__(
        self,
        cat_probs: np.ndarray,        # shape (P, X_train, C)
        answers: np.ndarray,          # shape (X_train,)
        cat_probs_test: np.ndarray,   # shape (P, X_test, C)
        answers_test: np.ndarray,     # shape (X_test,)
        cost_per_prompt: Sequence[float] | np.ndarray = (1.0,),
        cost_coeffs: dict[str, float] | None = None,
        context_distribution: Sequence[float] = (1/3, 1/3, 1/3),
        N_max: int = 16,
        rng: Optional[int | np.random.Generator] = None,
    ) -> None:
        # ---- validate tensors ----
        if cat_probs.ndim != 3:
            raise ValueError("cat_probs must have shape (P, X_train, C)")
        if cat_probs_test.ndim != 3:
            raise ValueError("cat_probs_test must have shape (P, X_test, C)")
        if answers.ndim != 1 or answers_test.ndim != 1:
            raise ValueError("answers / answers_test must be 1‑D arrays")

        P, X_train, C = cat_probs.shape
        P2, X_test, C2 = cat_probs_test.shape
        if P2 != P or C2 != C:
            raise ValueError("Train/test tensors must share #prompts and #choices")
        if answers.shape[0] != X_train or answers_test.shape[0] != X_test:
            raise ValueError("answers length must match #queries in probs arrays")

        # probability validity
        if not np.allclose(cat_probs.sum(axis=2), 1.0, atol=1e-6):
            raise ValueError("cat_probs rows must sum to 1")
        if not np.allclose(cat_probs_test.sum(axis=2), 1.0, atol=1e-6):
            raise ValueError("cat_probs_test rows must sum to 1")
        if not ((0 <= cat_probs) & (cat_probs <= 1)).all():
            raise ValueError("cat_probs must lie in [0,1]")
        if not ((0 <= cat_probs_test) & (cat_probs_test <= 1)).all():
            raise ValueError("cat_probs_test must lie in [0,1]")

        self.num_prompts  = P
        self.num_choices  = C
        self.cat_probs    = np.asarray(cat_probs, dtype=float)
        self.answers      = np.asarray(answers, dtype=int)
        self.cat_probs_te = np.asarray(cat_probs_test, dtype=float)
        self.answers_te   = np.asarray(answers_test, dtype=int)

        # ---- context distribution ----
        self.context_distribution = np.asarray(context_distribution, dtype=float)
        if len(self.context_distribution) != 3 or not np.isclose(self.context_distribution.sum(), 1.0):
            raise ValueError("context_distribution must have length 3 and sum to 1")

        # ---- costs ----
        self.cost_per_prompt = np.asarray(cost_per_prompt, dtype=float)
        if self.cost_per_prompt.shape[0] != self.num_prompts:
            raise ValueError("cost_per_prompt length must equal #prompts")
        self.cost_coeffs = cost_coeffs or {"low": 0.1, "mid": 0.5, "high": 1.0}
        if set(self.cost_coeffs) != set(self.CONTEXTS):
            raise ValueError("cost_coeffs must provide 'low','mid','high'")

        # ---- misc ----
        if N_max < 1:
            raise ValueError("N_max must be ≥ 1")
        self.N_max = N_max
        self.rng = np.random.default_rng(rng)

        # pre‑build contexts
        self._contexts = [CostContext(level, (1.0, -1.0 * self.cost_coeffs[level]))
                          for level in self.CONTEXTS]
        self.prompts = list(range(self.num_prompts))


    def combine_and_resplit(self, train_frac: float = 0.8, seed: int | None = None, shuffle: bool = True) -> None:
        """
        Merge current train/test tensors and re-split them in-place using *train_frac*.

        Parameters
        ----------
        train_frac : float
            Fraction of queries to assign to the new training split (0 < train_frac < 1).
        seed : int | None
            Optional random seed for reproducibility (overrides the env RNG for this op).
        shuffle : bool
            Whether to shuffle queries before splitting.
        """
        if not (0.0 < train_frac < 1.0):
            raise ValueError("train_frac must lie in (0,1)")

        # Concatenate across query axis
        all_probs = np.concatenate([self.cat_probs, self.cat_probs_te], axis=1)      # (P, X_all, C)
        all_answers = np.concatenate([self.answers, self.answers_te], axis=0)        # (X_all,)

        X_all = all_answers.shape[0]
        rng = np.random.default_rng(seed if seed is not None else self.rng)

        indices = np.arange(X_all)
        if shuffle:
            rng.shuffle(indices)

        n_train = int(np.floor(X_all * train_frac))
        n_train = max(1, min(n_train, X_all - 1))  # ensure both splits non-empty

        train_idx = indices[:n_train]
        test_idx = indices[n_train:]

        # Reassign tensors
        self.cat_probs = all_probs[:, train_idx, :]
        self.answers = all_answers[train_idx]
        self.cat_probs_te = all_probs[:, test_idx, :]
        self.answers_te = all_answers[test_idx]

    # ------------------ sampling helpers ------------------
    def sample_context(self) -> CostContext:
        idx = self.rng.choice(3, p=self.context_distribution)
        return self._contexts[idx]

    def _sample_query_index(self, split: str) -> int:
        if split == "train":
            return self.rng.integers(self.cat_probs.shape[1])
        else:
            return self.rng.integers(self.cat_probs_te.shape[1])


    # -------------------- interaction ---------------------
    def pull(self, prompt_id: int, N: int, split: str = "train") -> np.ndarray:
        """
        Return raw completions for a sampled query:
            columns = [predicted_category, ground_truth_category, per_completion_cost]
        """
        if not (0 <= prompt_id < self.num_prompts):
            raise IndexError("prompt_id out of range")
        if not (1 <= N <= self.N_max):
            raise ValueError("N must be in [1, N_max]")

        ex_idx = self._sample_query_index(split)
        if split == "train":
            probs  = self.cat_probs[prompt_id, ex_idx]   # shape (C,)
            answer = self.answers[ex_idx]
        else:
            probs  = self.cat_probs_te[prompt_id, ex_idx]
            answer = self.answers_te[ex_idx]

        preds = self.rng.choice(self.num_choices, size=N, p=probs)
        costs = np.full(N, self.cost_per_prompt[prompt_id], dtype=float)
        answer_col = np.full(N, answer, dtype=int)
        raw = np.vstack([preds, answer_col, costs]).T  # (N,3)
        return raw

    # -------------------- aggregators ---------------------
    def get_utility(self, raw: np.ndarray, context: CostContext, strategy: str) -> float:
        if strategy == "mv":
            return self.majority_vote(raw, context)
        raise ValueError(f"Unknown strategy '{strategy}'")

    @staticmethod
    def majority_vote(
        raw: np.ndarray,
        context: CostContext,
        rng: Optional[np.random.Generator] = None,
    ) -> float:
        rng = rng or np.random.default_rng()
        preds = raw[:, 0].astype(int)
        answer = int(raw[0, 1])
        total_cost = raw[:, 2].sum()

        # FIX: remove float minlength
        counts = np.bincount(preds)  # or: np.bincount(preds, minlength=4)

        max_count = counts.max()
        tied = np.flatnonzero(counts == max_count)

        if tied.size == 1:
            majority_correct = 1.0 if tied[0] == answer else 0.0
        else:
            majority_correct = 0.5 if answer in tied else 0.0

        return context.weights[0] * majority_correct + context.weights[1] * total_cost

    # -------------------- diagnostics ---------------------
    def summary(self) -> None:
        print("Synthetic Categorical Environment (single tier)")
        print("---------------------------------------------")
        print(f"#Prompts: {self.num_prompts}")
        print(f"#Train queries: {self.cat_probs.shape[1]}")
        print(f"#Test  queries: {self.cat_probs_te.shape[1]}")
        print(f"#Choices per query: {self.num_choices}")
        print("Cost per prompt:", self.cost_per_prompt)
        print("Cost coefficients:", self.cost_coeffs)
        print("Context distribution:", self.context_distribution)
        print("N_max:", self.N_max)
