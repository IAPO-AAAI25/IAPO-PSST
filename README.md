# IAPO (public code)

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

Notes:
- The demo runner uses small randomly generated environments; it does **not** require the `data/*.pkl` files.
- `--env cat_bon` uses the Best-of-N (`bon`) utility; `--env synth_bernoulli` and `--env cat_mv` use Majority Voting (`mv`).

