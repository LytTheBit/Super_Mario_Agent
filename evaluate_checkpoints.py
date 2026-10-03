# evaluate_checkpoints.py
import argparse
import re
import statistics as stats
from pathlib import Path

import gym
import gym_super_mario_bros
from gym.wrappers import FrameStack
from nes_py.wrappers import JoypadSpace

from wrappers import SkipFrame, GrayScaleObservation, ResizeObservation
from agent import Mario


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate all checkpoints in a run folder")
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--episodes-per-checkpoint", type=int, default=10)
    parser.add_argument(
        "--timeout-length",
        type=int,
        default=2000,
        help="Episode length (in agent steps, after SkipFrame) above which a non-flag_get "
             "episode is classified as 'timeout' rather than 'death'.",
    )
    return parser.parse_args()


def run_episode(mario, env, timeout_length):
    """Play one greedy episode and return a dict of raw per-episode metrics."""
    state = env.reset()
    done = False
    total_reward = 0.0
    steps = 0
    max_x_pos = 0
    final_info = {}

    while not done:
        action = mario.act(state)
        state, reward, done, trunc, info = env.step(action)
        total_reward += reward
        steps += 1
        max_x_pos = max(max_x_pos, info.get("x_pos", 0))
        final_info = info
        if info.get("flag_get", False):
            break

    if final_info.get("flag_get", False):
        outcome = "success"
    elif steps >= timeout_length:
        outcome = "timeout"
    else:
        outcome = "death"

    return {
        "reward": total_reward,
        "length": steps,
        "max_x_pos": max_x_pos,
        "coins": final_info.get("coins", 0),
        "score": final_info.get("score", 0),
        "outcome": outcome,
    }


def evaluate_checkpoint(mario, env, n_episodes, timeout_length):
    """Run n_episodes greedy episodes and return (summary dict, list of raw episode dicts)."""
    episodes = [run_episode(mario, env, timeout_length) for _ in range(n_episodes)]

    rewards = [e["reward"] for e in episodes]
    lengths = [e["length"] for e in episodes]
    x_positions = [e["max_x_pos"] for e in episodes]
    coins = [e["coins"] for e in episodes]
    scores = [e["score"] for e in episodes]
    outcomes = [e["outcome"] for e in episodes]

    summary = {
        "mean_reward": stats.mean(rewards),
        "std_reward": stats.pstdev(rewards) if n_episodes > 1 else 0.0,
        "mean_length": stats.mean(lengths),
        "std_length": stats.pstdev(lengths) if n_episodes > 1 else 0.0,
        "mean_max_x_pos": stats.mean(x_positions),
        "std_max_x_pos": stats.pstdev(x_positions) if n_episodes > 1 else 0.0,
        "mean_coins": stats.mean(coins),
        "mean_score": stats.mean(scores),
        "success_rate": outcomes.count("success") / n_episodes,
        "timeout_rate": outcomes.count("timeout") / n_episodes,
        "death_rate": outcomes.count("death") / n_episodes,
    }
    return summary, episodes


def main():
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint_dir)

    # Sort checkpoints numerically (mario_net_1, mario_net_2, ..., mario_net_31), not alphabetically
    checkpoints = sorted(
        checkpoint_dir.glob("mario_net_*.chkpt"),
        key=lambda p: int(re.search(r"mario_net_(\d+)\.chkpt", p.name).group(1)),
    )

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
    env = FrameStack(env, num_stack=4)

    mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=checkpoint_dir)
    mario.eval_mode = True

    summary_rows = []
    raw_rows = []

    header = (
        f"{'Checkpoint':<20}{'MeanReward':>12}{'StdReward':>11}{'MeanLen':>10}"
        f"{'MeanXPos':>10}{'MeanScore':>11}{'Success':>10}{'Timeout':>10}{'Death':>9}"
    )
    print(header)

    for ckpt in checkpoints:
        mario.load(str(ckpt))
        summary, episodes = evaluate_checkpoint(
            mario, env, args.episodes_per_checkpoint, args.timeout_length
        )
        summary_rows.append((ckpt.name, summary))
        for i, ep in enumerate(episodes):
            raw_rows.append((ckpt.name, i, ep))

        print(
            f"{ckpt.name:<20}"
            f"{summary['mean_reward']:>12.1f}"
            f"{summary['std_reward']:>11.1f}"
            f"{summary['mean_length']:>10.1f}"
            f"{summary['mean_max_x_pos']:>10.1f}"
            f"{summary['mean_score']:>11.1f}"
            f"{summary['success_rate']:>9.1%}"
            f"{summary['timeout_rate']:>10.1%}"
            f"{summary['death_rate']:>9.1%}"
        )

    env.close()

    # --- evaluation_curve.csv: one row per checkpoint, aggregated metrics ---
    curve_path = checkpoint_dir / "evaluation_curve.csv"
    with open(curve_path, "w") as f:
        f.write(
            "checkpoint,mean_reward,std_reward,mean_length,std_length,"
            "mean_max_x_pos,std_max_x_pos,mean_coins,mean_score,"
            "success_rate,timeout_rate,death_rate\n"
        )
        for name, s in summary_rows:
            f.write(
                f"{name},{s['mean_reward']:.2f},{s['std_reward']:.2f},"
                f"{s['mean_length']:.2f},{s['std_length']:.2f},"
                f"{s['mean_max_x_pos']:.2f},{s['std_max_x_pos']:.2f},"
                f"{s['mean_coins']:.2f},{s['mean_score']:.2f},"
                f"{s['success_rate']:.4f},{s['timeout_rate']:.4f},{s['death_rate']:.4f}\n"
            )

    # --- evaluation_raw.csv: one row per episode, for distribution-level analysis ---
    raw_path = checkpoint_dir / "evaluation_raw.csv"
    with open(raw_path, "w") as f:
        f.write("checkpoint,episode,reward,length,max_x_pos,coins,score,outcome\n")
        for name, i, ep in raw_rows:
            f.write(
                f"{name},{i},{ep['reward']:.2f},{ep['length']},"
                f"{ep['max_x_pos']},{ep['coins']},{ep['score']},{ep['outcome']}\n"
            )

    print(f"\nSalvato in {curve_path}")
    print(f"Salvato in {raw_path}")


if __name__ == "__main__":
    main()