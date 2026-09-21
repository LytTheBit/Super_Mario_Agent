# Super Mario Agent

A reinforcement learning project that teaches an agent to play Super Mario Bros. using a neural network and a DQN-style training pipeline.

![Super Mario Bros.](https://upload.wikimedia.org/wikipedia/en/5/50/NES_Super_Mario_Bros.png)

This project is inspired by the excellent tutorial by H. Huang:
https://h-huang.github.io/tutorials/intermediate/mario_rl_tutorial.html

## Overview

This repository implements an agent that:

- observes the game state from grayscale, stacked frames;
- selects actions through a neural policy;
- stores experiences in a replay buffer;
- updates the model using Q-learning;
- saves checkpoints and training metrics during execution.

The core idea is simple but effective: let the model learn to optimize game reward over time without handcrafted rules.

## Project structure

- `main.py`: training and evaluation entry point;
- `agent.py`: agent definition and learning loop;
- `neural_net.py`: neural network architecture;
- `wrappers.py`: observation preprocessing and frame transformations;
- `metric_logger.py`: progress and reward logging.

## Requirements

Before running the project, make sure you have the following installed:

- Python 3.10+
- PyTorch
- Gym
- gym-super-mario-bros
- nes-py
- torchrl
- tensordict

Recommended install:

```bash
pip install torch gym gym-super-mario-bros nes-py torchrl tensordict
```

## Quick start

### Full training run

```bash
python main.py --run-name configA --episodes 5000 --log-every 10
```

### Resume an interrupted training run

```bash
python main.py --run-name configA --episodes 5000 --load-checkpoint checkpoints\configA\mario_net_3.chkpt
```

### Demo in evaluation mode

```bash
python main.py --load-checkpoint checkpoints\configA\mario_net_9.chkpt --eval --render human --speed normal --episodes 3
```

### Fast smoke test

```bash
python main.py --episodes 40
```

## Available command-line options

| Flag | Values | Default | Description |
|---|---|---:|---|
| `--run-name` | free text | automatic timestamp | Name of the `checkpoints/` folder and label for the run. |
| `--episodes` | integer | `40` | Number of episodes to train or play. |
| `--render` | `rgb` / `human` | `rgb` | `rgb` = fast headless mode; `human` = visual game window. |
| `--speed` | `fast` / `normal` | `fast` | Only relevant with `--render human`; `normal` matches real NES timing. |
| `--load-checkpoint` | `.chkpt` path | none | Resume from a saved checkpoint. |
| `--eval` | flag | disabled | Evaluation mode: no learning, no exploration. |
| `--save-every` | integer (steps) | `500000` | Save checkpoint frequency. |
| `--log-every` | integer (episodes) | `20` | Log summary interval. |

## Output

During execution, the project saves:

- checkpoints in `checkpoints/<run-name>/`;
- run configuration in `config.json`;
- metrics and episode summaries via `MetricLogger`.

## Notes

- `rgb` mode is much faster and is the preferred option for serious training runs.
- `human` mode is useful for visual demos and manual inspection.
- `--eval` is useful to test a trained model without continuing training.

## Image note

The image above is used for illustrative purposes and was sourced from a public Wikimedia page. If you want to add more visuals to the README, it is best to use royalty-free or properly licensed assets, or local files stored in the repository.

## License

This project is intended for educational and personal experimentation. Please check the repository files for any specific license details or constraints.
