"""Record one greedy episode of a trained agent as an animated GIF.

Example:
    python record_gif.py --run Overnight-v2 --checkpoint mario_net_21 --out assets/mario_demo.gif

The episode is played with the purely greedy policy (no exploration), so the
recording is reproducible. One GIF frame is captured per agent step (every 4
emulator frames, because of SkipFrame), i.e. about 15 frames per second.
"""
import argparse
from pathlib import Path

import gym
import gym_super_mario_bros
from gym.wrappers import FrameStack
from nes_py.wrappers import JoypadSpace
from PIL import Image

from agent import Mario
from wrappers import SkipFrame, GrayScaleObservation, ResizeObservation

FRAME_MS = 4 / 60.0988 * 1000  # duration of one agent step at real NES speed


def parse_args():
    p = argparse.ArgumentParser(description="Record a greedy episode as a GIF")
    p.add_argument("--run", required=True, help="Run folder inside checkpoints/")
    p.add_argument("--checkpoint", required=True, help="e.g. mario_net_21 or best")
    p.add_argument("--out", default="assets/mario_demo.gif", help="Output GIF path")
    p.add_argument("--stride", type=int, default=2,
                   help="Keep one frame every N agent steps (larger = smaller file)")
    p.add_argument("--scale", type=float, default=1.0, help="Resize factor for the GIF")
    p.add_argument("--max-steps", type=int, default=3000, help="Safety cap on episode length")
    return p.parse_args()


def make_env():
    # Same wrappers as in training (no reward shaping: it does not affect actions).
    if gym.__version__ < "0.26":
        env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", new_step_api=True)
    else:
        env = gym_super_mario_bros.make(
            "SuperMarioBros-1-1-v0", render_mode="rgb_array", apply_api_compatibility=True
        )
    env = JoypadSpace(env, [["right"], ["right", "A"]])
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    return FrameStack(env, num_stack=4)


def grab(base_env):
    """Return the current emulator screen (240x256x3 RGB) as a PIL image."""
    frame = getattr(base_env, "screen", None)
    if frame is None:
        if gym.__version__ < "0.26":
            frame = base_env.render("rgb_array")
        else:
            frame = base_env.render()
    return Image.fromarray(frame.copy())


def main():
    args = parse_args()
    run_dir = Path("checkpoints") / args.run
    ckpt = run_dir / (args.checkpoint if args.checkpoint.endswith(".chkpt")
                      else f"{args.checkpoint}.chkpt")

    env = make_env()
    base = env.unwrapped
    mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=run_dir)
    mario.eval_mode = True
    mario.load(str(ckpt))

    state = env.reset()
    frames = [grab(base)]
    steps = 0
    reached_flag = False
    while True:
        action = mario.act(state)
        state, _, done, _, info = env.step(action)
        steps += 1
        if steps % args.stride == 0:
            frames.append(grab(base))
        if info["flag_get"]:
            reached_flag = True
        if done or info["flag_get"] or steps >= args.max_steps:
            break
    frames.append(grab(base))
    env.close()

    if args.scale != 1.0:
        size = (int(frames[0].width * args.scale), int(frames[0].height * args.scale))
        frames = [f.resize(size, Image.NEAREST) for f in frames]

    durations = [int(FRAME_MS * args.stride)] * len(frames)
    durations[-1] = 1500  # hold the last frame
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=durations,
                   loop=0, optimize=True)
    print(f"Episode: {steps} agent steps, flag reached: {reached_flag}")
    print(f"Saved {len(frames)} frames to {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
