"""Figures for Part I, drawn only from data/part1_numbers.json (exported from the lab).

Run from the paper folder:  python figures/make_figures.py
"""
import json
import math
from pathlib import Path
from statistics import NormalDist

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = json.loads((HERE.parent / "data/part1_numbers.json").read_text())

# Okabe-Ito: distinguishable under the common colour-vision deficiencies.
LSTM, MLP = "#0072B2", "#D55E00"
INK, MUTED, GRID = "#222222", "#666666", "#DDDDDD"
COMPONENT_COLORS = {"seed": "#0072B2", "seed_x_map": "#009E73", "noise": "#BBBBBB"}

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "lines.linewidth": 1.5,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})


def save(fig, name):
    fig.savefig(HERE / f"{name}.pdf")
    fig.savefig(HERE / f"{name}.png", dpi=200)
    plt.close(fig)


def learning_curves():
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    for arm, color in (("LSTM", LSTM), ("MLP", MLP)):
        curves = DATA["curves"][arm]
        steps = np.array(curves["steps"]) / 1000
        per_seed = np.array(curves["per_seed"])
        for row in per_seed:
            ax.plot(steps, row, color=color, alpha=0.12, linewidth=0.6)
        ax.plot(steps, per_seed.mean(axis=0), color=color, linewidth=2)
        ax.annotate(f"{arm} (mean of 30)", (steps[-1], per_seed.mean(axis=0)[-1]),
                    xytext=(-4, 7 if arm == "LSTM" else -11), textcoords="offset points",
                    ha="right", color=INK, fontsize=7)
    ax.axhline(0.75, color=INK, linestyle="--", linewidth=0.8)
    ax.text(2, 0.765, "acquisition threshold 0.75", color=MUTED, fontsize=6.5)
    ax.axhline(DATA["random"]["validation"], color=MUTED, linestyle=":", linewidth=0.8)
    ax.text(198, DATA["random"]["validation"] + 0.015, "random policy", color=MUTED,
            fontsize=6.5, ha="right")
    ax.set(xlabel="Training interactions (thousands)", ylabel="Validation success (32 maps)",
           xlim=(0, 200), ylim=(0, 1))
    save(fig, "fig_learning_curves")


def paired_seeds():
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    lstm, mlp = DATA["per_seed_test"]["LSTM"], DATA["per_seed_test"]["MLP"]
    seeds = sorted(lstm, key=lambda s: lstm[s] - mlp[s])
    for i, seed in enumerate(seeds):
        ax.plot([mlp[seed], lstm[seed]], [i, i], color=GRID, linewidth=1, zorder=1)
    y = np.arange(len(seeds))
    ax.scatter([mlp[s] for s in seeds], y, color=MLP, s=10, zorder=2, label="MLP (2,919 params)",
               edgecolor="white", linewidth=0.4)
    ax.scatter([lstm[s] for s in seeds], y, color=LSTM, s=10, zorder=3,
               label="LSTM (11,367 params)", edgecolor="white", linewidth=0.4)
    ax.axvline(DATA["random"]["test"], color=MUTED, linestyle=":", linewidth=0.8)
    ax.text(DATA["random"]["test"] + 0.01, len(seeds) - 1, "random", color=MUTED, fontsize=6.5,
            va="top")
    ax.set(xlabel="Test success after training (64 maps)", ylabel="Training seed (sorted by gap)",
           xlim=(0, 1), yticks=[])
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False,
              handletextpad=0.2)
    save(fig, "fig_paired_seeds")


def score_composition():
    """Between-seed variance of a run's single-evaluation score, split into its sources."""
    rows = [("final_test_success", 64, "Final test success"),
            ("validation_auc", 32, "Validation AUC"),
            ("acquired", 1, "Acquired (yes/no)")]
    labels, parts = [], []
    for metric, maps, title in rows:
        for group in ("LSTM", "MLP", "LSTM - MLP"):
            if metric == "acquired" and group != "LSTM":
                continue  # the MLP never acquires B, so its variance is zero
            c = DATA["variance"][metric][group]["components"]
            pieces = {"seed": c["seed"], "seed_x_map": c["seed_x_map"] / maps,
                      "noise": c["noise"] / maps}
            total = sum(pieces.values())
            parts.append({k: v / total for k, v in pieces.items()})
            labels.append(f"{title}: {group.replace(' - ', ' $-$ ')}")
    fig, ax = plt.subplots(figsize=(6.6, 2.3))
    y = np.arange(len(labels))[::-1]
    left = np.zeros(len(labels))
    names = {"seed": "training seed", "seed_x_map": "seed $\\times$ map",
             "noise": "evaluation sampling noise"}
    for key in ("seed", "seed_x_map", "noise"):
        values = np.array([p[key] for p in parts])
        ax.barh(y, values, left=left, color=COMPONENT_COLORS[key], height=0.62,
                edgecolor="white", linewidth=1.0, label=names[key])
        for yi, li, v in zip(y, left, values):
            if v >= 0.08:
                ax.text(li + v / 2, yi, f"{v:.0%}", ha="center", va="center", fontsize=6.5,
                        color="white" if key != "noise" else INK)
        left += values
    ax.set_yticks(y, labels)
    ax.set(xlim=(0, 1), xlabel="Share of the between-seed variance of one evaluation")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=3, frameon=False)
    save(fig, "fig_score_composition")


def seeds_needed():
    """Seeds for a paired two-sided test (alpha 0.05, power 0.8), one evaluation per run."""
    z = NormalDist().inv_cdf(0.975) + NormalDist().inv_cdf(0.8)
    gaps = np.linspace(0.03, 0.30, 200)
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    series = [("final_test_success", "LSTM - MLP", 64, "final test success", LSTM, "-", 14),
              ("validation_auc", "LSTM - MLP", 32, "validation AUC", "#009E73", "--", 8.7),
              ("acquired", "LSTM", 1, "acquired (yes/no)", MLP, "-.", 12)]
    for metric, group, maps, label, color, style, at in series:
        c = DATA["variance"][metric][group]["components"]
        per_seed = c["seed"] + c["seed_x_map"] / maps + c["noise"] / maps
        n = np.maximum(2, np.ceil(z ** 2 * per_seed / gaps ** 2))
        ax.plot(gaps * 100, n, color=color, linestyle=style)
        i = int(np.argmin(abs(gaps * 100 - at)))
        ax.annotate(label, (gaps[i] * 100, n[i]), xytext=(4, 3), textcoords="offset points",
                    color=INK, fontsize=6.5)
    for typical in (3, 5, 10):
        ax.axhline(typical, color=MUTED, linewidth=0.5, linestyle=":")
        ax.text(26, typical * 1.06, f"{typical} seeds", color=MUTED, fontsize=6, ha="center")
    ax.set_yscale("log")
    ax.set(xlabel="True gap to detect (percentage points)", ylabel="Seeds needed (log scale)",
           xlim=(3, 30), ylim=(2, 1000))
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    save(fig, "fig_seeds_needed")


if __name__ == "__main__":
    learning_curves()
    paired_seeds()
    score_composition()
    seeds_needed()
    print("figures written to", HERE)
    assert not math.isnan(DATA["paired_gap"]["mean"])
