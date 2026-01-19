# IAPO ( Inference-Aware Prompt Optimization for Aligning Black-Box Large Language Model, AAAI 2025)

This folder is a lightweight/public copy of the IAPO codebase containing:

- `Agents/`: reference agent implementations (PSST, UCB, Uniform, TRIPLE, etc.)
- `Envs/`: synthetic environments used by the agents
- `data/`: optional pickled env datasets (`*.pkl`)
- `example_runner.py`: small demo showing how to run each env + agent

## Quickstart

```bash
python3 -m pip install numpy
python3 example_runner.py --list
python3 example_runner.py --env synth_bernoulli --agent psst --budget 500 --test 200
python3 example_runner.py --run-all --budget 300 --test 100
```

## Run with pickled datasets

If you have `data/*.pkl` files (e.g., `data/HH_final.pkl`), you can load them and run an agent on the stored env list:

```bash
python3 example_runner.py --pkl data/HH_final.pkl --agent psst --budget 3000 --test 1000 --first-k 5
```

Notes:
- The demo runner uses small randomly generated environments by default; it does **not** require the `data/*.pkl` files.
- `--first-k` limits how many envs are loaded from the pickle (useful for a quick test).
- `--env cat_bon` uses the Best-of-N (`bon`) utility; `--env synth_bernoulli` and `--env cat_mv` use Majority Voting (`mv`).
