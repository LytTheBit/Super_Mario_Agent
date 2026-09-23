# evaluate_checkpoints.py
import argparse
import re
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
    return parser.parse_args()


args = parse_args()
checkpoint_dir = Path(args.checkpoint_dir)

# Sort checkpoints numerically (mario_net_1, mario_net_2, ..., mario_net_31), not alphabetically
checkpoints = sorted(
    checkpoint_dir.glob("mario_net_*.chkpt"),
    key=lambda p: int(re.search(r"mario_net_(\d+)\.chkpt", p.name).group(1)),
)

if gym.__version__ < '0.26':
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

results = []
print(f"{'Checkpoint':<20}{'MeanReward':>12}{'MeanLength':>12}{'SuccessRate':>14}")

for ckpt in checkpoints:
    mario.load(str(ckpt))
    rewards, lengths, successes = [], [], 0

    for _ in range(args.episodes_per_checkpoint):
        state = env.reset()
        ep_reward, ep_length = 0.0, 0
        while True:
            action = mario.act(state)
            next_state, reward, done, trunc, info = env.step(action)
            ep_reward += reward
            ep_length += 1
            state = next_state
            if done or info["flag_get"]:
                if info["flag_get"]:
                    successes += 1
                break
        rewards.append(ep_reward)
        lengths.append(ep_length)

    mean_reward = sum(rewards) / len(rewards)
    mean_length = sum(lengths) / len(lengths)
    success_rate = successes / args.episodes_per_checkpoint
    results.append((ckpt.name, mean_reward, mean_length, success_rate))
    print(f"{ckpt.name:<20}{mean_reward:>12.1f}{mean_length:>12.1f}{success_rate:>13.1%}")

env.close()

with open(checkpoint_dir / "evaluation_curve.csv", "w") as f:
    f.write("checkpoint,mean_reward,mean_length,success_rate\n")
    for name, r, l, s in results:
        f.write(f"{name},{r:.2f},{l:.2f},{s:.4f}\n")

print(f"\nSalvato in {checkpoint_dir / 'evaluation_curve.csv'}")