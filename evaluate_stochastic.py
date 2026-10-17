# evaluate_stochastic.py
"""Robustness check: evaluate selected checkpoints under epsilon-noise.

For each (checkpoint, epsilon) pair, play N episodes where, at every step,
a random action replaces the greedy one with probability epsilon.
Writes <run_dir>/evaluation_stochastic.csv (appending, one row per pair).
"""
import argparse
import csv
from pathlib import Path

import gym
import gym_super_mario_bros
import numpy as np
from gym.wrappers import FrameStack
from nes_py.wrappers import JoypadSpace

from agent import Mario
from wrappers import SkipFrame, GrayScaleObservation, ResizeObservation


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate checkpoints under epsilon-noise")
    p.add_argument("--run", required=True, help="Run folder name inside checkpoints/")
    p.add_argument("--checkpoints", nargs="+", required=True,
                   help="Checkpoint names, e.g. mario_net_21 best")
    p.add_argument("--epsilons", nargs="+", type=float, default=[0.0, 0.02, 0.05, 0.1])
    p.add_argument("--episodes", type=int, default=20, help="Episodes per (checkpoint, epsilon)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def make_env():
    if gym.__version__ < "0.26":
        env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", new_step_api=True)
    else:
        env = gym_super_mario_bros.make(
            "SuperMarioBros-1-1-v0", render_mode="rgb", apply_api_compatibility=True
        )
    env = JoypadSpace(env, [["right"], ["right", "A"]])
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    return FrameStack(env, num_stack=4)


def play_episode(env, mario, epsilon, rng):
    """One episode; returns (success, max_x_pos, length)."""
    state = env.reset()
    max_x, length = 0, 0
    while True:
        greedy_action = mario.act(state)  # eval_mode: always argmax, no step counting
        if epsilon > 0 and rng.random() < epsilon:
            action = int(rng.integers(mario.action_dim))
        else:
            action = greedy_action
        state, _, done, _, info = env.step(action)
        max_x = max(max_x, info["x_pos"])
        length += 1
        if done or info["flag_get"]:
            return bool(info["flag_get"]), max_x, length


def main():
    args = parse_args()
    run_dir = Path("checkpoints") / args.run
    env = make_env()
    mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=run_dir)
    mario.eval_mode = True
    rng = np.random.default_rng(args.seed)

    out_path = run_dir / "evaluation_stochastic.csv"
    new_file = not out_path.exists()
    header = ["run", "checkpoint", "step", "epsilon", "episodes",
              "success_rate", "mean_max_x_pos", "std_max_x_pos", "mean_length"]

    print(f"{'checkpoint':<16}{'eps':>6}{'success':>10}{'mean_x':>10}{'std_x':>9}{'len':>8}")
    with open(out_path, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(header)
        for name in args.checkpoints:
            ckpt = run_dir / (name if name.endswith(".chkpt") else f"{name}.chkpt")
            mario.load(str(ckpt))
            for eps in args.epsilons:
                results = [play_episode(env, mario, eps, rng) for _ in range(args.episodes)]
                succ = np.mean([r[0] for r in results])
                xs = np.array([r[1] for r in results])
                lens = np.array([r[2] for r in results])
                print(f"{ckpt.stem:<16}{eps:>6.2f}{succ:>10.0%}{xs.mean():>10.0f}"
                      f"{xs.std():>9.0f}{lens.mean():>8.0f}")
                writer.writerow([args.run, ckpt.stem, mario.curr_step, eps, args.episodes,
                                 f"{succ:.4f}", f"{xs.mean():.1f}", f"{xs.std():.1f}",
                                 f"{lens.mean():.1f}"])
                f.flush()
    env.close()


if __name__ == "__main__":
    main()