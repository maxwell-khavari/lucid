"""Headline chart from results/eval.json: error against solver calls, one panel per scenario.

python3 -m eval.plots
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e2dc"
SERIES = [  # key, label, colour, linestyle
    ("learned", "Lucid's self-check", "#2a78d6", "-"),
    ("residual", "Physics residual", "#eb6834", "-"),
    ("input", "Input-range rule", "#1baf7a", "-"),
    ("schedule", "Every k-th frame", "#8a887f", "--"),
    ("random", "Random frames", "#b3b1a8", ":"),
]
TITLES = {"normal": "Normal stirring", "kicks": "Kicks outside the training range"}


def main(path="results/eval.json", out="results/headline.png"):
    res = json.load(open(path))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=SURFACE)
    for ax, name in zip(axes, ("normal", "kicks")):
        r = res[name]
        ax.set_facecolor(SURFACE)
        for key, label, col, ls in SERIES:
            pts = sorted(r["curves"][key], key=lambda p: p["calls"])
            xs = [0.0] + [100 * p["calls"] for p in pts]
            ys = [100 * r["pure_lucid"]["error"]] + [100 * p["error"] for p in pts]
            ax.plot(xs, ys, ls, color=col, lw=2, marker="o", ms=4, label=label,
                    zorder=3 if key == "learned" else 2)
        ax.set_title(TITLES[name], loc="left", color=INK, fontsize=11)
        ax.set_xlabel("Frames handed to the real solver (%)", color=MUTED)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.grid(True, color=GRID, lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=MUTED)
    axes[0].set_ylabel("Error vs. real solver (%)", color=MUTED)
    axes[1].legend(frameon=False, labelcolor=INK, loc="upper right")
    fig.suptitle("Lower is better. Every line starts at the network alone (0% solver).",
                 x=0.01, ha="left", color=MUTED, fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    main()
