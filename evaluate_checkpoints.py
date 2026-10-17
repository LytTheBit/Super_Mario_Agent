# evaluate_checkpoints.py
"""Evaluate every numbered checkpoint of every run folder under `checkpoints/`.

Results are printed to the terminal and saved to:
  - <root>/evaluation_all.csv : one row per (run, checkpoint), for plotting/analysis
  - <root>/evaluation_all.md  : per-run summary, equal-step-budget comparison, details

Evaluation is incremental: a checkpoint is re-evaluated only if its file
modification time changed (or with --force).
"""
import argparse
import contextlib
import csv
import fnmatch
import io
import re
import statistics as stats
from datetime import datetime
from pathlib import Path

import gym
import gym_super_mario_bros
from gym.wrappers import FrameStack
from nes_py.wrappers import JoypadSpace

from agent import Mario
from wrappers import SkipFrame, GrayScaleObservation, ResizeObservation

CSV_COLUMNS = [
    "run", "checkpoint", "index", "step", "epsilon",
    "mean_reward", "std_reward", "mean_length", "mean_max_x_pos", "std_max_x_pos",
    "mean_coins", "mean_score", "success_rate", "timeout_rate", "death_rate", "modified",
]
STR_FIELDS = {"run", "checkpoint", "modified"}
INT_FIELDS = {"index", "step"}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate all checkpoints of all runs.")
    parser.add_argument("--checkpoints-root", type=str, default="checkpoints")
    parser.add_argument("--runs", nargs="*", default=None,
                        help="Only evaluate these run folders (overrides --exclude).")
    parser.add_argument("--exclude", nargs="*", default=[],
                        help="Optional fnmatch patterns of run folders to skip "
                             "(e.g. --exclude 'smoke-*'). Nothing is excluded by default.")
    parser.add_argument("--episodes-per-checkpoint", type=int, default=1,
                        help="Greedy policies are deterministic, so 1 is enough.")
    parser.add_argument("--timeout-length", type=int, default=2000,
                        help="Non-success episodes at least this long are 'timeout'.")
    parser.add_argument("--budget", type=int, default=18_000_000,
                        help="Step budget for the equal-step comparison table.")
    parser.add_argument("--output-prefix", type=str, default="evaluation_all")
    parser.add_argument("--force", action="store_true",
                        help="Re-evaluate checkpoints even if cached.")
    return parser.parse_args()


# --------------------------------------------------------------------------- env
def make_env():
    if gym.__version__ < "0.26":
        env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", new_step_api=True)
    else:
        env = gym_super_mario_bros.make(
            "SuperMarioBros-1-1-v0", render_mode="rgb", apply_api_compatibility=True
        )
    env = JoypadSpace(env, [["right"], ["right", "A"]])
    # No RewardShaping here on purpose: evaluation always uses the default reward,
    # so MeanReward is comparable across runs trained with different reward modes.
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    env = FrameStack(env, num_stack=4)
    return env


# ------------------------------------------------------------------- evaluation
def run_episode(mario, env, timeout_length):
    """Play one greedy episode and return raw per-episode metrics."""
    state = env.reset()
    done = False
    total_reward, steps, max_x_pos = 0.0, 0, 0
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
        "reward": total_reward, "length": steps, "max_x_pos": max_x_pos,
        "coins": final_info.get("coins", 0), "score": final_info.get("score", 0),
        "outcome": outcome,
    }


def evaluate_checkpoint(mario, env, n_episodes, timeout_length):
    eps = [run_episode(mario, env, timeout_length) for _ in range(n_episodes)]
    sd = (lambda xs: stats.pstdev(xs) if len(xs) > 1 else 0.0)
    outcomes = [e["outcome"] for e in eps]
    return {
        "mean_reward": stats.mean(e["reward"] for e in eps),
        "std_reward": sd([e["reward"] for e in eps]),
        "mean_length": stats.mean(e["length"] for e in eps),
        "mean_max_x_pos": stats.mean(e["max_x_pos"] for e in eps),
        "std_max_x_pos": sd([e["max_x_pos"] for e in eps]),
        "mean_coins": stats.mean(e["coins"] for e in eps),
        "mean_score": stats.mean(e["score"] for e in eps),
        "success_rate": outcomes.count("success") / n_episodes,
        "timeout_rate": outcomes.count("timeout") / n_episodes,
        "death_rate": outcomes.count("death") / n_episodes,
    }


# -------------------------------------------------------------------- file utils
def list_checkpoints(run_dir):
    """Numbered checkpoints only (mario_net_N.chkpt), sorted numerically."""
    found = []
    for p in run_dir.glob("mario_net_*.chkpt"):
        m = re.fullmatch(r"mario_net_(\d+)\.chkpt", p.name)
        if m:
            found.append((int(m.group(1)), p))
    return sorted(found)


def select_runs(root, args):
    all_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    if args.runs:
        selected = [root / r for r in args.runs if (root / r).is_dir()]
        missing = [r for r in args.runs if not (root / r).is_dir()]
        if missing:
            print(f"Cartelle non trovate: {missing}")
        return selected
    selected, skipped = [], []
    for d in all_dirs:
        if any(fnmatch.fnmatch(d.name, pat) for pat in args.exclude):
            skipped.append(d.name)
        else:
            selected.append(d)
    if skipped:
        print(f"Cartelle escluse ({args.exclude}): {skipped}")
    return selected


def load_existing(csv_path, root):
    """Load cached rows, pruning rows whose checkpoint file no longer exists."""
    rows = {}
    if not csv_path.exists():
        return rows
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if not (root / r["run"] / r["checkpoint"]).exists():
                continue
            for k in CSV_COLUMNS:
                if k in STR_FIELDS:
                    continue
                r[k] = int(float(r[k])) if k in INT_FIELDS else float(r[k])
            rows[(r["run"], r["checkpoint"])] = r
    return rows


def write_csv(rows, csv_path):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLUMNS)
        for r in sorted(rows.values(), key=lambda r: (r["run"], r["index"])):
            w.writerow([
                r[k] if (k in STR_FIELDS or k in INT_FIELDS) else f"{r[k]:.4f}"
                for k in CSV_COLUMNS
            ])


# ----------------------------------------------------------------------- tables
def fmt_step(s):
    return f"{s / 1e6:.2f}M"


def group_by_run(rows):
    runs = {}
    for r in rows.values():
        runs.setdefault(r["run"], []).append(r)
    for rs in runs.values():
        rs.sort(key=lambda r: r["index"])
    return dict(sorted(runs.items()))


SUMMARY_HEADERS = ["Run", "Ckpt", "Ultimo step", "Ckpt con successo",
                   "Primo successo", "Best x_pos (step)", "x_pos medio ultimi 10"]


def summary_row(run, rs):
    n = len(rs)
    succ = [r for r in rs if r["success_rate"] > 0]
    best = max(rs, key=lambda r: r["mean_max_x_pos"])
    last10 = rs[-10:]
    return [
        run, str(n), fmt_step(rs[-1]["step"]), f"{len(succ)}/{n}",
        fmt_step(succ[0]["step"]) if succ else "-",
        f"{best['mean_max_x_pos']:.0f} ({fmt_step(best['step'])})",
        f"{stats.mean(r['mean_max_x_pos'] for r in last10):.0f}",
    ]


def summary_tables(runs, budget):
    full = [summary_row(run, rs) for run, rs in runs.items()]
    capped = []
    for run, rs in runs.items():
        within = [r for r in rs if r["step"] <= budget]
        if within:
            capped.append(summary_row(run, within))
    return full, capped


DETAIL_HEADERS = ["Checkpoint", "Step", "MeanReward", "MeanLen", "MeanXPos",
                  "MeanScore", "Success", "Timeout", "Death"]


def detail_row(r):
    return [r["checkpoint"], fmt_step(r["step"]), f"{r['mean_reward']:.0f}",
            f"{r['mean_length']:.0f}", f"{r['mean_max_x_pos']:.0f}",
            f"{r['mean_score']:.0f}", f"{r['success_rate']:.0%}",
            f"{r['timeout_rate']:.0%}", f"{r['death_rate']:.0%}"]


def md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def text_table(headers, rows):
    widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h)
              for i, h in enumerate(headers)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    return "\n".join([fmt.format(*headers)] + [fmt.format(*r) for r in rows])


def write_markdown(rows, md_path, args):
    runs = group_by_run(rows)
    full, capped = summary_tables(runs, args.budget)
    out = [
        "# Valutazione checkpoint",
        "",
        f"Generato il {datetime.now().strftime('%Y-%m-%d %H:%M')} - "
        f"{args.episodes_per_checkpoint} episodio/i greedy per checkpoint, "
        f"reward di default dell'ambiente (nessun reward shaping in valutazione).",
        "",
        "## Riepilogo (tutti gli step)", "", md_table(SUMMARY_HEADERS, full), "",
        f"## Confronto a parità di step (<= {fmt_step(args.budget)})", "",
        md_table(SUMMARY_HEADERS, capped), "",
        "## Dettaglio per run", "",
    ]
    for run, rs in runs.items():
        out += [f"<details><summary>{run} ({len(rs)} checkpoint)</summary>", "",
                md_table(DETAIL_HEADERS, [detail_row(r) for r in rs]), "",
                "</details>", ""]
    md_path.write_text("\n".join(out), encoding="utf-8")


def write_outputs(rows, csv_path, md_path, args):
    write_csv(rows, csv_path)
    write_markdown(rows, md_path, args)


# -------------------------------------------------------------------------- main
def main():
    args = parse_args()
    root = Path(args.checkpoints_root)
    csv_path = root / f"{args.output_prefix}.csv"
    md_path = root / f"{args.output_prefix}.md"

    run_dirs = select_runs(root, args)
    rows = load_existing(csv_path, root)

    env = make_env()
    mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=root)
    mario.eval_mode = True

    header = (f"{'Run':<28}{'Checkpoint':<20}{'Step':>9}{'Reward':>8}{'Len':>6}"
              f"{'XPos':>7}{'Score':>7}{'Succ':>6}{'TO':>5}{'Death':>7}")
    print(header)

    try:
        for run_dir in run_dirs:
            for idx, ckpt in list_checkpoints(run_dir):
                modified = datetime.fromtimestamp(ckpt.stat().st_mtime).strftime("%Y-%m-%dT%H:%M:%S")
                key = (run_dir.name, ckpt.name)
                cached = rows.get(key)
                tag = ""
                if cached and cached["modified"] == modified and not args.force:
                    r, tag = cached, " (cache)"
                else:
                    with contextlib.redirect_stdout(io.StringIO()):  # silence Mario.load prints
                        mario.load(str(ckpt))
                    s = evaluate_checkpoint(mario, env, args.episodes_per_checkpoint,
                                            args.timeout_length)
                    r = {"run": run_dir.name, "checkpoint": ckpt.name, "index": idx,
                         "step": int(mario.curr_step), "epsilon": float(mario.exploration_rate),
                         "modified": modified, **s}
                    rows[key] = r
                print(f"{r['run']:<28}{r['checkpoint']:<20}{r['step']:>9}"
                      f"{r['mean_reward']:>8.0f}{r['mean_length']:>6.0f}"
                      f"{r['mean_max_x_pos']:>7.0f}{r['mean_score']:>7.0f}"
                      f"{r['success_rate']:>6.0%}{r['timeout_rate']:>5.0%}"
                      f"{r['death_rate']:>7.0%}{tag}")
            write_outputs(rows, csv_path, md_path, args)  # save after every run folder
    except KeyboardInterrupt:
        print("\nInterrotto: salvo i risultati parziali.")
    finally:
        env.close()
        write_outputs(rows, csv_path, md_path, args)

    runs = group_by_run(rows)
    full, capped = summary_tables(runs, args.budget)
    print("\n=== Riepilogo (tutti gli step) ===")
    print(text_table(SUMMARY_HEADERS, full))
    print(f"\n=== Confronto a parità di step (<= {fmt_step(args.budget)}) ===")
    print(text_table(SUMMARY_HEADERS, capped))
    print(f"\nSalvato in {csv_path}\nSalvato in {md_path}")


if __name__ == "__main__":
    main()