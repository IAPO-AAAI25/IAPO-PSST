#!/usr/bin/env python3
"""
example_runner.py
-----------------
Small, self-contained demo that shows how to instantiate each environment in
`Envs/` and train/evaluate each agent in `Agents/`.

Examples
--------
  python example_runner.py --list
  python example_runner.py --env synth_bernoulli --agent psst --budget 500 --test 200
  python example_runner.py --run-all --budget 300 --test 100
  python example_runner.py --pkl data/HH_final.pkl --agent psst --budget 3000 --test 1000 --first-k 5
"""

from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Tuple

import numpy as np

# Agents
from Agents.PSST import PSSTAgent
from Agents.PSST_HB import PSSTHAgent
from Agents.TRIPLE import TRAgent
from Agents.UCB import UCBAgent
from Agents.Uniform import UniformAgent
from Agents.adaptiv_sampling import AdaptiveAgent
from Agents.random_agent import RandomAgent

# Environments
from Envs.catBoN.catBoN import CategoricalBoNEnv
from Envs.catMV.catMV import CategoricalMVEnv
from Envs.synthBernoulli.synthBernoulli import SyntheticBernoulliEnv


PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class EnvSpec:
    name: str
    strategy: str
    factory: Callable[[int], Any]


def _build_synth_bernoulli(seed: int) -> SyntheticBernoulliEnv:
    rng = np.random.default_rng(seed)
    num_prompts = 3
    x_train = 40
    x_test = 20

    base = rng.uniform(0.3, 0.8, size=(num_prompts, 1))
    tier_mod = {"easy": 0.10, "medium": 0.00, "hard": -0.10}

    success_probs: Dict[str, np.ndarray] = {}
    success_probs_test: Dict[str, np.ndarray] = {}
    for tier, mod in tier_mod.items():
        success_probs[tier] = np.clip(
            base + mod + rng.normal(0.0, 0.05, size=(num_prompts, x_train)),
            0.05,
            0.95,
        )
        success_probs_test[tier] = np.clip(
            base + mod + rng.normal(0.0, 0.05, size=(num_prompts, x_test)),
            0.05,
            0.95,
        )

    cost_per_prompt = rng.uniform(0.5, 1.5, size=num_prompts)
    return SyntheticBernoulliEnv(
        success_probs=success_probs,
        success_probs_test=success_probs_test,
        cost_per_prompt=cost_per_prompt,
        N_max=8,
        rng=seed,
    )


def _build_cat_mv(seed: int) -> CategoricalMVEnv:
    rng = np.random.default_rng(seed)
    num_prompts = 3
    num_choices = 4
    x_train = 50
    x_test = 25

    answers = rng.integers(num_choices, size=x_train)
    answers_test = rng.integers(num_choices, size=x_test)

    # Prompt quality controls how much probability mass goes to the true answer.
    prompt_acc = rng.uniform(0.35, 0.75, size=num_prompts)

    def _make_probs(y: np.ndarray) -> np.ndarray:
        probs = np.full((num_prompts, y.shape[0], num_choices), 0.0, dtype=float)
        for p in range(num_prompts):
            for i, a in enumerate(y):
                acc = float(prompt_acc[p])
                row = np.full(num_choices, (1.0 - acc) / (num_choices - 1), dtype=float)
                row[int(a)] = acc
                row += rng.normal(0.0, 0.01, size=num_choices)
                row = np.clip(row, 1e-6, None)
                row /= row.sum()
                probs[p, i] = row
        return probs

    cat_probs = _make_probs(answers)
    cat_probs_test = _make_probs(answers_test)
    cost_per_prompt = rng.uniform(0.5, 1.5, size=num_prompts)

    return CategoricalMVEnv(
        cat_probs=cat_probs,
        answers=answers,
        cat_probs_test=cat_probs_test,
        answers_test=answers_test,
        cost_per_prompt=cost_per_prompt,
        N_max=8,
        rng=seed,
    )


def _build_cat_bon(seed: int) -> CategoricalBoNEnv:
    rng = np.random.default_rng(seed)
    num_prompts = 3
    q_train = 12
    q_test = 6
    num_outcomes = 6
    num_pos_objectives = 2

    # Outcome vectors contain ONLY positive objectives; env appends a cost column.
    outcome_vectors_train = rng.uniform(
        0.0,
        1.0,
        size=(num_prompts, q_train, num_outcomes, num_pos_objectives),
    )
    outcome_vectors_test = rng.uniform(
        0.0,
        1.0,
        size=(num_prompts, q_test, num_outcomes, num_pos_objectives),
    )

    # Random categorical probabilities over outcomes for each (prompt, query).
    outcome_probs_train = rng.uniform(0.0, 1.0, size=(num_prompts, q_train, num_outcomes))
    outcome_probs_train /= outcome_probs_train.sum(axis=2, keepdims=True)
    outcome_probs_test = rng.uniform(0.0, 1.0, size=(num_prompts, q_test, num_outcomes))
    outcome_probs_test /= outcome_probs_test.sum(axis=2, keepdims=True)

    cost_per_prompt = rng.uniform(0.5, 1.5, size=num_prompts)

    # Context weights are (w_obj1, w_obj2, w_cost). Cost weight should be <= 0.
    weights = [(1.0, 1.0, -0.25), (1.0, 0.5, -0.5), (0.5, 1.0, -1.0)]
    context_distribution = [1 / 3, 1 / 3, 1 / 3]

    return CategoricalBoNEnv(
        outcome_vectors_train=outcome_vectors_train,
        outcome_probs_train=outcome_probs_train,
        outcome_vectors_test=outcome_vectors_test,
        outcome_probs_test=outcome_probs_test,
        cost_per_prompt=cost_per_prompt,
        weights=weights,
        context_distribution=context_distribution,
        N_max=8,
        rng=seed,
    )


ENVS: Dict[str, EnvSpec] = {
    "synth_bernoulli": EnvSpec("synth_bernoulli", "mv", _build_synth_bernoulli),
    "cat_mv": EnvSpec("cat_mv", "mv", _build_cat_mv),
    "cat_bon": EnvSpec("cat_bon", "bon", _build_cat_bon),
}


def agent_factories(strategy: str) -> Dict[str, Callable[[], Any]]:
    return {
        "psst": lambda: PSSTAgent(strategy=strategy),
        "psst_h1": lambda: PSSTHAgent(strategy=strategy, burn=0.2, K=1),
        "psst_h4": lambda: PSSTHAgent(strategy=strategy, burn=0.2, K=4),
        "psst_h8": lambda: PSSTHAgent(strategy=strategy, burn=0.2, K=8),
        "ucb": lambda: UCBAgent(strategy=strategy),
        "uniform": lambda: UniformAgent(strategy=strategy),
        "adaptive_epsilon": lambda: AdaptiveAgent(strategy=strategy, action_mode="epsilon", epsilon=0.1),
        "adaptive_softmax": lambda: AdaptiveAgent(strategy=strategy, action_mode="softmax", tau=0.05),
        "random": lambda: RandomAgent(strategy=strategy),
        "triple_n1": lambda: TRAgent(strategy=strategy, h="one"),
        "triple_randomN": lambda: TRAgent(strategy=strategy, h="random"),
    }


def _maybe_resplit(env: Any, seed: int) -> None:
    if hasattr(env, "combine_and_resplit"):
        try:
            env.combine_and_resplit(train_frac=0.8, seed=seed)
        except TypeError:
            # Some envs may not accept a seed keyword; ignore for the demo.
            env.combine_and_resplit(train_frac=0.8)


def evaluate(agent: Any, env: Any, test_steps: int) -> float:
    total = 0.0
    for _ in range(int(test_steps)):
        ctx = env.sample_context()
        prompt_id, N = agent.predict(ctx)
        raw = env.pull(prompt_id, int(N), split="test")
        total += float(env.get_utility(raw, ctx, agent.strategy))
    return total / max(1, int(test_steps))


def run_one(env_key: str, agent_key: str, budget: int, test_steps: int, seed: int) -> float:
    env_spec = ENVS[env_key]
    factories = agent_factories(env_spec.strategy)
    if agent_key not in factories:
        raise KeyError(f"Unknown agent: {agent_key}")

    env = env_spec.factory(seed)
    _maybe_resplit(env, seed=seed)

    agent = factories[agent_key]()
    agent.train(env=env, budget=int(budget))
    return evaluate(agent, env, test_steps=int(test_steps))


def _load_envs_from_pickle(pkl_path: str) -> list[Any]:
    path = Path(pkl_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"No such pickle file: {path}")
    with path.open("rb") as fh:
        obj = pickle.load(fh)
    if isinstance(obj, list):
        return obj
    return [obj]


def _infer_env_strategy(env: Any) -> str:
    """
    Infer which utility strategy the environment supports.
    Currently supports Majority Voting ("mv") or Best-of-N ("bon").
    """
    ctx = env.sample_context()
    prompt_id = int(env.prompts[0]) if hasattr(env, "prompts") else 0
    raw = env.pull(prompt_id, 1, split="test")

    for strategy in ("mv", "bon"):
        try:
            env.get_utility(raw, ctx, strategy)
            return strategy
        except Exception:
            continue

    raise ValueError(
        f"Could not infer strategy for env type={type(env)}; expected mv/bon-compatible env."
    )


def run_from_pickle(
    pkl_path: str,
    agent_key: str,
    budget: int,
    test_steps: int,
    seed: int,
    first_k: int | None = None,
) -> float:
    resolved_path = Path(pkl_path)
    if not resolved_path.is_absolute():
        resolved_path = PROJECT_ROOT / resolved_path
    envs = _load_envs_from_pickle(resolved_path.as_posix())
    if not envs:
        raise ValueError(f"Pickle file contains no envs: {resolved_path}")
    if first_k is not None and first_k > 0:
        envs = envs[: int(first_k)]

    strategy = _infer_env_strategy(envs[0])
    factories = agent_factories(strategy)
    if agent_key not in factories:
        raise KeyError(
            f"Unknown agent '{agent_key}' for strategy='{strategy}'. Use --list to see valid agent keys."
        )

    scores: list[float] = []
    for idx, env in enumerate(envs, start=1):
        _maybe_resplit(env, seed=seed + idx)
        agent = factories[agent_key]()
        agent.train(env=env, budget=int(budget))
        score = evaluate(agent, env, test_steps=int(test_steps))
        scores.append(float(score))
        print(f"env#{idx} avg_utility={score:.4f}")

    mean = float(np.mean(scores))
    sem = float(np.std(scores, ddof=1) / np.sqrt(len(scores))) if len(scores) > 1 else 0.0
    print(
        f"dataset={resolved_path.name} strategy={strategy} agent={agent_key} "
        f"mean={mean:.4f} sem={sem:.4f} n_envs={len(scores)}"
    )
    return mean


def _print_list() -> None:
    print("Environments:")
    for k, spec in ENVS.items():
        print(f"  - {k} (strategy='{spec.strategy}')")
    print("\nAgents:")
    # Show the superset across strategies.
    all_agents = sorted(set(agent_factories("mv")) | set(agent_factories("bon")))
    for a in all_agents:
        print(f"  - {a}")
    print(
        "\nTip: MV envs require agents with strategy='mv'; BoN envs require strategy='bon'."
    )
    data_dir = PROJECT_ROOT / "data"
    if data_dir.is_dir():
        pkls = sorted(data_dir.glob("*.pkl"))
        if pkls:
            print("\nPickle datasets (optional):")
            for p in pkls:
                print(f"  - {p.relative_to(PROJECT_ROOT).as_posix()}")
            print("  (run with: --pkl <path> --agent <agent> [--first-k K])")


def _iter_all_runs(envs: Iterable[str], budget: int, test_steps: int, seed: int) -> None:
    for env_key in envs:
        env_spec = ENVS[env_key]
        factories = agent_factories(env_spec.strategy)
        print(f"\n== env={env_key} (strategy={env_spec.strategy}) ==")
        for agent_key in sorted(factories.keys()):
            score = run_one(env_key, agent_key, budget=budget, test_steps=test_steps, seed=seed)
            print(f"{agent_key:>16}  avg_utility={score:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal IAPO env/agent demo runner.")
    parser.add_argument("--list", action="store_true", help="List available envs and agents.")
    parser.add_argument("--env", default="synth_bernoulli", choices=sorted(ENVS.keys()))
    parser.add_argument(
        "--pkl",
        default="",
        help="Load env(s) from a pickle file (overrides --env). Example: data/HH_final.pkl",
    )
    parser.add_argument("--agent", default="psst", help="Agent key (use --list).")
    parser.add_argument("--budget", type=int, default=500, help="Training token budget.")
    parser.add_argument("--test", type=int, default=200, help="Number of test rollouts.")
    parser.add_argument(
        "--first-k",
        type=int,
        default=0,
        help="When using --pkl, use only the first K envs from the pickle list (0 = all).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-all", action="store_true", help="Run all agents on all envs.")
    args = parser.parse_args()

    if args.list:
        _print_list()
        return

    if args.run_all:
        _iter_all_runs(ENVS.keys(), budget=args.budget, test_steps=args.test, seed=args.seed)
        return

    if args.pkl:
        first_k = int(args.first_k) if int(args.first_k) > 0 else None
        run_from_pickle(
            args.pkl,
            args.agent,
            budget=args.budget,
            test_steps=args.test,
            seed=args.seed,
            first_k=first_k,
        )
        return

    env_key = args.env
    env_spec = ENVS[env_key]
    factories = agent_factories(env_spec.strategy)
    if args.agent not in factories:
        raise SystemExit(
            f"Unknown agent '{args.agent}'. Use --list to see valid agent keys."
        )

    score = run_one(env_key, args.agent, budget=args.budget, test_steps=args.test, seed=args.seed)
    print(f"env={env_key} agent={args.agent} avg_utility={score:.4f}")


if __name__ == "__main__":
    main()
