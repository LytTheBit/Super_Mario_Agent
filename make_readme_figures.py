from pathlib import Path
"""Regenerate the figures used in README.md.

Usage:  python make_readme_figures.py
Writes: assets/pipeline.png, assets/results_overview.png

The numbers in RESULTS come from checkpoints/evaluation_all.csv (greedy evaluation,
one episode per checkpoint). Update them by hand after new runs are evaluated.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path("assets")
OUT.mkdir(exist_ok=True)

# --------------------------------------------------------------------------- #
# Figure 1: preprocessing pipeline + network
# --------------------------------------------------------------------------- #

def box(ax, x, y, w, h, text, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=color, ec="#333333", lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", lw=1.3, color="#333333"))


def make_pipeline():
    fig, ax = plt.subplots(figsize=(13, 5.2))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    w, h, gap = 1.85, 0.95, 0.38
    # Row 1: preprocessing
    ax.text(0.05, 4.55, "1. Observation preprocessing", fontsize=11, weight="bold")
    row1 = ["NES frame\n240x256x3 RGB", "SkipFrame\nrepeat action x4\nsum rewards", "Grayscale",
            "Resize\n84x84, scale [0,1]", "FrameStack\n4 frames", "State\n(4, 84, 84)"]
    for i, t in enumerate(row1):
        x = 0.05 + i * (w + gap)
        box(ax, x, 3.35, w, h, t, "#dbe9f6")
        if i:
            arrow(ax, x - gap, 3.35 + h / 2, x, 3.35 + h / 2)

    # Row 2: network
    ax.text(0.05, 2.85, "2. MarioNet (online and target copies)", fontsize=11, weight="bold")
    row2 = ["Conv 8x8, s4\n32 ch + ReLU", "Conv 4x4, s2\n64 ch + ReLU", "Conv 3x3, s1\n64 ch + ReLU",
            "Flatten\n3136", "Linear 512\n+ ReLU", "Q(s, a)\n2 actions"]
    for i, t in enumerate(row2):
        x = 0.05 + i * (w + gap)
        box(ax, x, 1.65, w, h, t, "#fde6c8" if i < 5 else "#d8f0d2")
        if i:
            arrow(ax, x - gap, 1.65 + h / 2, x, 1.65 + h / 2)

    # Row 3: training loop
    ax.text(0.05, 1.15, "3. Training loop", fontsize=11, weight="bold")
    ax.text(0.05, 0.45,
            "epsilon-greedy action  ->  env step  ->  cache in replay buffer (100k)  ->  "
            "sample batch (32)  ->  TD estimate vs. Double-DQN target  ->  Adam update\n"
            "target network synced from online network every 10k steps",
            fontsize=9.5, va="center")
    fig.savefig(OUT / "pipeline.png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 2: results overview (greedy evaluation of saved checkpoints)
# --------------------------------------------------------------------------- #
# name, checkpoints that complete the level, checkpoints evaluated, median max x_pos
RESULTS = [
    ("Longday\n(soft update)", 0, 81, 1783),
    ("Overnight v1\n(default reward)", 1, 55, 1138),
    ("Overnight v2\n(default reward)", 1, 113, 1646),
    ("CoinShaping v2", 0, 98, 1315),
    ("FlagBonus v2", 1, 100, 1655),
    ("GamePoint", 6, 105, 1126),
]
LEVEL_END_X = 3160  # approximate x position of the flag in level 1-1


def make_results():
    names = [r[0] for r in RESULTS]
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.6))

    rates = [100 * s / n for _, s, n, _ in RESULTS]
    bars = a.bar(names, rates, color="#4c78a8")
    for bar, (_, s, n, _) in zip(bars, RESULTS):
        a.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
               f"{s}/{n}", ha="center", fontsize=9)
    a.set_ylabel("% of checkpoints that complete the level")
    a.set_ylim(0, max(rates) * 1.35 + 1)
    a.set_title("Level completions (greedy policy)")
    a.tick_params(axis="x", labelsize=8)

    xs = [r[3] for r in RESULTS]
    b.bar(names, xs, color="#f58518")
    b.axhline(LEVEL_END_X, ls="--", color="#333333", lw=1)
    b.text(len(names) - 0.5, LEVEL_END_X + 40, "end of level", ha="right", fontsize=9)
    b.set_ylabel("median max x position per checkpoint")
    b.set_ylim(0, 3500)
    b.set_title("Progress through the level")
    b.tick_params(axis="x", labelsize=8)

    fig.suptitle("Greedy evaluation of saved checkpoints (one run per configuration, one episode "
                 "per checkpoint)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "results_overview.png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    make_pipeline()
    make_results()
    print("Figures written to", OUT.resolve())
