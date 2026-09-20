# main.py
import argparse
import datetime
import json
import time
from pathlib import Path

import gym
import gym_super_mario_bros
from gym.wrappers import FrameStack
from nes_py.wrappers import JoypadSpace

from wrappers import SkipFrame, GrayScaleObservation, ResizeObservation
from agent import Mario
from metric_logger import MetricLogger


def parse_args():
    parser = argparse.ArgumentParser(description="Train or evaluate a Mario RL agent")
    parser.add_argument("--save-every", type=int, default=int(5e5))
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Label for this run (e.g. 'configA-lr2.5e-4'). Used for the checkpoint folder name "
             "and printed to console, so you can tell runs apart when comparing configurations.",
    )
    parser.add_argument("--render", choices=["rgb", "human"], default="rgb")
    parser.add_argument(
        "--speed",
        choices=["fast", "normal"],
        default="fast",
        help="'normal' paces playback to real NES speed (~60 fps) for recording demos. "
             "Only meaningful with --render human; ignored otherwise.",
    )
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument(
        "--load-checkpoint",
        type=str,
        default=None,
        help="Path to a .chkpt file to resume training from, or to play with a trained agent.",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Evaluation mode: no learning, no exploration (pure exploit). "
             "Use together with --load-checkpoint and --render human --speed normal for demos.",
    )



    return parser.parse_args()



args = parse_args()

if gym.__version__ < '0.26':
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", new_step_api=True)
else:
    env = gym_super_mario_bros.make(
        "SuperMarioBros-1-1-v0", render_mode=args.render, apply_api_compatibility=True
    )

env = JoypadSpace(env, [["right"], ["right", "A"]])
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=84)
env = FrameStack(env, num_stack=4)

# Name the run: explicit --run-name if given, otherwise a timestamp
run_label = args.run_name or datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir = Path("checkpoints") / run_label
save_dir.mkdir(parents=True, exist_ok=True)

mario = Mario(
    state_dim=(4, 84, 84),
    action_dim=env.action_space.n,
    save_dir=save_dir,
    save_every=args.save_every,
)

if args.load_checkpoint:
    mario.load(args.load_checkpoint)

if args.eval:
    mario.exploration_rate = 0.0  # pure exploit, no random actions

# Snapshot of the hyperparameters used in this run, for later comparison across configurations
config = {
    "run_name": run_label,
    "episodes": args.episodes,
    "eval_mode": args.eval,
    "loaded_checkpoint": args.load_checkpoint,
    "gamma": mario.gamma,
    "learning_rate": mario.optimizer.param_groups[0]["lr"],
    "batch_size": mario.batch_size,
    "sync_every": mario.sync_every,
    "learn_every": mario.learn_every,
    "burnin": mario.burnin,
    "exploration_rate_decay": mario.exploration_rate_decay,
    "exploration_rate_min": mario.exploration_rate_min,
}
with open(save_dir / "config.json", "w") as f:
    json.dump(config, f, indent=2)

logger = MetricLogger(save_dir)

print(f"=== Esecuzione: {run_label} ===")
print(f"episodes={args.episodes}  device={mario.device}  render={args.render}  eval={args.eval}")

# NES runs at ~60.0988 fps; SkipFrame repeats each action for 4 real frames per env.step() call
FRAME_DELAY = 4 / 60.0988

for e in range(args.episodes):
    state = env.reset()

    while True:
        action = mario.act(state)
        next_state, reward, done, trunc, info = env.step(action)

        if not args.eval:
            mario.cache(state, next_state, action, reward, done)
            q, loss = mario.learn()
        else:
            q, loss = None, None

        logger.log_step(reward, loss, q)
        state = next_state

        if args.render == "human" and args.speed == "normal":
            time.sleep(FRAME_DELAY)

        if done or info["flag_get"]:
            break

    logger.log_episode()

    if (e % 20 == 0) or (e == args.episodes - 1):
        logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.curr_step)

env.close()