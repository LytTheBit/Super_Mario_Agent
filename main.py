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

from wrappers import SkipFrame, GrayScaleObservation, ResizeObservation, RewardShaping
from agent import Mario
from metric_logger import MetricLogger


def parse_args():
    parser = argparse.ArgumentParser(description="Train or evaluate a Mario RL agent")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--render", choices=["rgb", "human"], default="rgb")
    parser.add_argument("--speed", choices=["fast", "normal"], default="fast")
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--load-checkpoint", type=str, default=None)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--save-every", type=int, default=int(5e5))
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument(
        "--reward-mode",
        choices=["none", "flag-bonus", "coin-shaping"],
        default="none",
        help="'none': default env reward. 'flag-bonus': +500 on level completion. "
             "'coin-shaping': + coin delta * 10, plus the flag bonus.",
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
env = RewardShaping(env, mode=args.reward_mode)
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=84)
env = FrameStack(env, num_stack=4)

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
    mario.eval_mode = True

config = {
    "run_name": run_label,
    "episodes": args.episodes,
    "eval_mode": args.eval,
    "loaded_checkpoint": args.load_checkpoint,
    "reward_mode": args.reward_mode,
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
print(f"episodes={args.episodes}  device={mario.device}  render={args.render}  eval={args.eval}  reward_mode={args.reward_mode}")

FRAME_DELAY = 4 / 60.0988

training_start = time.time()

try:
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

        if (e % args.log_every == 0) or (e == args.episodes - 1):
            logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.curr_step)

            mean_r, succ_rate = mario.quick_eval(env, n_episodes=1)
            mario.save_if_best(mean_r)
            print(f"Eval rapida: reward={mean_r:.1f} success={succ_rate:.0%}")

            elapsed = time.time() - training_start
            frac_done = (e + 1) / args.episodes
            eta_sec = elapsed / frac_done - elapsed if frac_done > 0 else float("nan")
            print(
                f"Progresso: {frac_done * 100:.1f}% ({e + 1}/{args.episodes} episodi) - "
                f"Trascorso {elapsed / 60:.1f} min - Stimato rimanente {eta_sec / 60:.1f} min"
            )
except KeyboardInterrupt:
    print("\nInterrotto manualmente — salvo un checkpoint finale prima di uscire...")
    if not args.eval:
        mario.save()
finally:
    env.close()
    print(f"Sessione terminata dopo {(time.time() - training_start) / 60:.1f} minuti.")