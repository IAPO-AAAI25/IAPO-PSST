#!/usr/bin/env python3
"""
Synthetic Multi-Objective Categorical Environment 
───────────────────────────────────────────────────────────────────────
• A (prompt, query) pair has M possible outcome vectors o_j ∈ R^(K+1).
  - The last coordinate can be a (possibly negative-weighted) cost term,
    or simply a (K+1)-th objective - you decide via the context weights.
• Probabilities π_j (Σπ_j = 1) define a categorical distribution.
• A *pull* draws N i.i.d. outcomes from that distribution and returns them
  as an (N, K+1) NumPy array.

All complication around “difficulty tiers” and “cost contexts” is gone.`
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, Tuple, Optional, List
import numpy as np

# ─── Context ────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Context:
    """A simple immutable (K+1,) weight vector."""
    weights: Tuple[float, ...]          # len = K+1

    def __iter__(self):
        return iter(self.weights)

# ─── Environment ────────────────────────────────────────────────────────────
class CategoricalBoNEnv:
    """
    Multi-objective categorical bandit environment *without* tiers.

    Shapes
    ------
    outcome_vectors_* : (P, Q, M, K+1)  float
        P  - #prompts
        Q  - #queries per prompt
        M  - #categorical outcomes for that (prompt, query)
        K+1 - #objectives (last may be cost)
    outcome_probs_*   : (P, Q, M) float   • categorical probabilities
    """

    # ------------------------------------------------------------------ #
    def __init__(self,
                 outcome_vectors_train: np.ndarray,
                 outcome_probs_train:  np.ndarray,
                 *,
                 outcome_vectors_test: Optional[np.ndarray] = None,
                 outcome_probs_test:   Optional[np.ndarray] = None,
                 cost_per_prompt:      Sequence[float],
                 weights: Sequence[Tuple[float, ...]],
                 context_distribution: Sequence[float],
                 N_max: int = 32,
                 rng: Optional[int | np.random.Generator] = None,
                 ) -> None:

        # – RNG & basic sanity -----------------------------------------
        self.rng = np.random.default_rng(rng)
        self.N_max = int(N_max)
        if self.N_max < 1:
            raise ValueError("N_max must be >=1")

        # – Contexts ----------------------------------------------------
        self._contexts: List[Context] = [Context(tuple(w)) for w in weights]
        self.context_distribution = np.asarray(context_distribution, dtype=float)
        if len(self.context_distribution) != len(self._contexts):
            raise ValueError("context_distribution length ≠ #contexts")
        if not np.isclose(self.context_distribution.sum(), 1.0):
            raise ValueError("context_distribution must sum to 1")

        # – Outcome tensors (train) ------------------------------------

        self.outcome_vectors_train = np.asarray(outcome_vectors_train, dtype=float)
        self.outcome_probs_train   = np.asarray(outcome_probs_train,   dtype=float)
        if self.outcome_vectors_train.ndim != 4:
            raise ValueError("outcome_vectors_train must be 4‑D (P,Q,M,K)")
        if self.outcome_probs_train.shape != self.outcome_vectors_train.shape[:3]:
            raise ValueError("outcome_probs_train shape mismatch with vectors")
        if not np.isclose(self.outcome_probs_train.sum(axis=2), 1.0).all():
            raise ValueError("Training probabilities must sum to 1")

        self.P, self.Q, self.M, self.K = self.outcome_vectors_train.shape 
        # ---------- per‑prompt cost --------------------------------------------
        self.cost_per_prompt = np.asarray(cost_per_prompt, dtype=float)
        if self.cost_per_prompt.shape != (self.P,):
            raise ValueError("cost_per_prompt length must equal #prompts (P)")

        # – Outcome tensors (test) -------------------------------------
        self.outcome_vectors_test = (
            np.asarray(outcome_vectors_test, dtype=float)
            if outcome_vectors_test is not None
            else self.outcome_vectors_train.copy()
        )
        self.outcome_probs_test = (
            np.asarray(outcome_probs_test, dtype=float)
            if outcome_probs_test is not None
            else self.outcome_probs_train.copy()
        )
        print(self.outcome_vectors_test.shape,self.outcome_vectors_train.shape )
        if (self.outcome_vectors_test.shape[2:] != self.outcome_vectors_train.shape[2:] or
                self.outcome_probs_test.shape[2:]  != self.outcome_probs_train.shape[2:]):
            raise ValueError("Train/test shapes must match")

        self._, self.Q_test, self._, self._ = self.outcome_vectors_test.shape 
        
        # Convenience alias for callers
        self.prompts = list(range(self.P))

    # ─── Sampling helpers ────────────────────────────────────────────
    def _sample_query_index(self, split = "train") -> int:
        """Uniformly sample a query index 0 … Q‑1."""
        if split == "train":
            return int(self.rng.integers(self.Q))
        else:
            return int(self.rng.integers(self.Q_test))
        
    def sample_context(self) -> Context:
        """Draw a context according to context_distribution."""
        idx = int(self.rng.choice(len(self._contexts), p=self.context_distribution))
        return self._contexts[idx]

    def combine_and_resplit(self, train_frac: float = 0.8,
                        seed: int | None = None,
                        shuffle: bool = True) -> None:
        """
        Merge current train/test outcome tensors and re‑split them in‑place.

        Parameters
        ----------
        train_frac : float
            Fraction of queries to assign to the *new* training split
            (must satisfy 0 < train_frac < 1).
        seed : int | None
            Optional random seed for reproducibility (overrides `self.rng`
            for this operation only).
        shuffle : bool
            Whether to shuffle query indices before splitting.
        """
        if not (0.0 < train_frac < 1.0):
            raise ValueError("train_frac must lie strictly between 0 and 1")

        # ── Concatenate along the query axis ───────────────────────────────
        all_vecs  = np.concatenate([self.outcome_vectors_train,
                                    self.outcome_vectors_test], axis=1)   # (P, Q_all, M, K+1)
        all_probs = np.concatenate([self.outcome_probs_train,
                                    self.outcome_probs_test],  axis=1)    # (P, Q_all, M)

        Q_all = all_vecs.shape[1]
        rng = np.random.default_rng(seed if seed is not None else self.rng)

        indices = np.arange(Q_all)
        if shuffle:
            rng.shuffle(indices)

        n_train = int(np.floor(Q_all * train_frac))
        n_train = max(1, min(n_train, Q_all - 1))      # ensure both splits non‑empty

        train_idx = indices[:n_train]
        test_idx  = indices[n_train:]

        # ── Reassign tensors in‑place ───────────────────────────────────────
        self.outcome_vectors_train = all_vecs[:, train_idx, :, :]
        self.outcome_probs_train   = all_probs[:, train_idx, :]

        self.outcome_vectors_test  = all_vecs[:, test_idx, :, :]
        self.outcome_probs_test    = all_probs[:, test_idx, :]

        # ── Update cached dimensions ───────────────────────────────────────
        self.Q      = self.outcome_vectors_train.shape[1]
        self.Q_test = self.outcome_vectors_test.shape[1]

    # ─── Pull / rollout ──────────────────────────────────────────────
    def pull(self, prompt_id: int, N: int, *, split: str = "train") -> np.ndarray:
        """
        Draw *N* categorical completions.  Return shape (N, K+1):
        • first K columns  – sampled positive objectives
        • last column      – constant cost_per_prompt[prompt_id]
        """
        # --- sanity checks unchanged ---
        q_idx = self._sample_query_index(split)
        if split == "train":
            vecs, probs = (self.outcome_vectors_train[prompt_id, q_idx],
                           self.outcome_probs_train[prompt_id, q_idx])
        else:
            vecs, probs = (self.outcome_vectors_test[prompt_id, q_idx],
                           self.outcome_probs_test[prompt_id, q_idx])
        #print(probs, vecs)
        choices   = self.rng.choice(self.M, size=N, p=probs)
        positives = vecs[choices]                      # (N, K)
        cost_col  = np.full((N, 1), self.cost_per_prompt[prompt_id], dtype=float)
        return np.hstack([positives, cost_col])        # (N, K+1)
                            # (N, K+1)

    # ─── Best‑of‑N utility  (unchanged mathematics) ──────────────────
    @staticmethod
    def best_of_n(raw: np.ndarray, context: Context) -> float:
        """
        u = max_i  Σ_{k=1..K} w_k·o_k(i)   +   w_{K+1}·N·cost_per_prompt
        (because every completion carries the same cost term)
        """
        positives = raw[:, :-1]              # (N, K)
        cost      = raw[0,  -1]              # scalar = cost_per_prompt[p]

        w_pos  = np.asarray(context.weights[:-1])
        w_cost = context.weights[-1]
        assert w_cost <= 0.0
        best_positive = (positives * w_pos).sum(axis=1).max()
        total_cost    = w_cost * raw.shape[0] * cost         # N * cost_per_prompt
        #print(best_positive)
        return float(best_positive + total_cost)

    def get_utility(self, raw: np.ndarray, context: Context,
                    strategy: str = "bon") -> float:
        if strategy != "bon":
            raise ValueError("Only 'bon' implemented.")
        return self.best_of_n(raw, context)