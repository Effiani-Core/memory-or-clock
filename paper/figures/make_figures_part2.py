"""Figures for Part II, drawn only from data/part2_numbers.json (exported from the lab).

Run from the paper folder:  python figures/make_figures_part2.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = json.loads((HERE.parent / "data/part2_numbers.json").read_text())
INK, MUTED, GRID = "#222222", "#666666", "#DDDDDD"

# Okabe-Ito, in a fixed order; LSTM and MLP keep their Part I colours.
ARMS = {  # label: (display name, colour, line style)
    "LSTM": ("LSTM", "#0072B2", "-"),
    "MLP-wide": ("MLP-wide + clock", "#009E73", "-"),
    "stack4": ("stack4 + clock", "#56B4E9", "--"),
    "LSTM-reset": ("LSTM-reset", "#E69F00", "--"),
    "MLP-wide-noclock": ("MLP-wide, no clock", "#CC79A7", "-."),
    "MLP": ("MLP (small)", "#D55E00", ":"),
}

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 6.5, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.5, "lines.linewidth": 1.5, "pdf.fonttype": 42,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02})


def save(fig, name):
    fig.savefig(HERE / f"{name}.pdf")
    fig.savefig(HERE / f"{name}.png", dpi=200)
    plt.close(fig)


def arms_on_sealed_panel():
    order = sorted(ARMS, key=lambda a: DATA["arms"][a]["sealed_mean"])
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    rng = np.random.default_rng(0)
    for y, arm in enumerate(order):
        name, color, _ = ARMS[arm]
        values = list(DATA["per_seed_sealed"][arm].values())
        jitter = rng.uniform(-0.18, 0.18, len(values))
        ax.scatter(values, y + jitter, s=8, color=color, alpha=0.55, edgecolor="none")
        info = DATA["arms"][arm]
        low, high = info["sealed_ci"]
        ax.plot([low, high], [y, y], color=INK, linewidth=1.2)
        ax.plot(info["sealed_mean"], y, marker="|", color=INK, markersize=9, mew=1.6)
        ax.text(1.02, y, f"{info['sealed_mean']:.0%} (n={info['n']})", va="center",
                ha="left", fontsize=6.5, color=INK, transform=ax.get_yaxis_transform(),
                clip_on=False)
    ax.set_yticks(range(len(order)), [ARMS[a][0] for a in order])
    ax.axvline(0.141, color=MUTED, linestyle=":", linewidth=0.8)
    ax.text(0.15, -0.6, "random", color=MUTED, fontsize=6)
    ax.set(xlim=(0, 1.0), ylim=(-0.8, len(order) - 0.5),
           xlabel="Success on the sealed panel (64 new maps $\\times$ 4)")
    ax.grid(axis="y", visible=False)
    save(fig, "fig_part2_arms")


def ladder():
    """From the small MLP to the LSTM, one ingredient at a time (10 shared seeds)."""
    steps = [("MLP", "small MLP"), ("MLP-wide-noclock", "+ width"),
             ("MLP-wide", "+ clock"), ("LSTM", "+ memory (LSTM)")]
    values = [DATA["ladder"][arm] for arm, _ in steps]
    total = values[-1] - values[0]
    fig, ax = plt.subplots(figsize=(3.3, 1.9))
    for i, ((arm, label), value) in enumerate(zip(steps, values)):
        color = ARMS[arm][1]
        if i == 0:
            ax.barh(i, value, color=color, height=0.55, edgecolor="white")
            ax.text(value - 0.01, i, f"{value:.0%}", ha="right", va="center",
                    color="white", fontsize=6.5)
        else:
            gain = value - values[i - 1]
            ax.barh(i, values[i - 1], color=GRID, height=0.55, edgecolor="white")
            ax.barh(i, gain, left=values[i - 1], color=color, height=0.55, edgecolor="white")
            ax.text(value + 0.01, i, f"{gain:+.1%}  ({gain / total:.0%} of the gap)",
                    va="center", fontsize=6.5, color=INK)
    ax.set_yticks(range(len(steps)), [label for _, label in steps])
    ax.invert_yaxis()
    ax.set(xlim=(0, 1.35), xlabel="Success on the sealed panel (10 seeds every arm shares)")
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(axis="y", visible=False)
    save(fig, "fig_part2_ladder")


def curves():
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    for arm, (name, color, style) in ARMS.items():
        c = DATA["curves"][arm]
        mean = np.array(c["per_seed"]).mean(axis=0)
        ax.plot(np.array(c["steps"]) / 1000, mean, color=color, linestyle=style,
                label=f"{name} ({len(c['per_seed'])})")
    ax.axhline(0.75, color=INK, linestyle="--", linewidth=0.7)
    ax.text(3, 0.765, "acquisition threshold", color=MUTED, fontsize=6)
    ax.set(xlabel="Training interactions (thousands)", ylabel="Validation success (mean)",
           xlim=(0, 300), ylim=(0, 1))
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.22), ncol=2, frameon=False)
    save(fig, "fig_part2_curves")


if __name__ == "__main__":
    arms_on_sealed_panel()
    ladder()
    curves()
    print("figures written to", HERE)
