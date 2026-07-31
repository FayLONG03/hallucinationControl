"""Regenerate the gpt5mini popQA confidence-distribution figures with larger text.

Reproduces the two-panel (First-round Answer / Best Guess) confidence histograms
that live in outputs/gpt5mini_results/plot_idk_confidence_separate, writing both
.png and .pdf to the existing file names so paper references keep working.

Usage:
    venv/bin/python main_exp/popQA/plot/replot_idk_confidence.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATA_DIR = os.path.join(REPO_ROOT, "main_exp/popQA/outputs/popqa_gpt5mini")
OUT_DIR = os.path.join(
    REPO_ROOT, "main_exp/popQA/outputs/gpt5mini_results/plot_idk_confidence_separate"
)

# (csv name, display label, reward string, output stem)
FIGURES = [
    (
        "popqa_A_results_+1_-1_20260106_100350.csv",
        "A",
        "+1, -1",
        "A_(+1_-1)_20260724_155948",
    ),
    (
        "popqa_B_results_+1_-1_+0.4_20260109_122206.csv",
        "B",
        "+1, -1, +0.4",
        "B_(+1_-1_+0.4)_20260724_163424",
    ),
    (
        "popqa_B_norm_results_+1_-1_+0.4_20260117_165433.csv",
        "I-CALM",
        "+1, -1, +0.4",
        "B_Norm_(+1_-1_+0.4)_20260724_213140",
    ),
]

# Text sizes (bumped up from the original 14/14/12/12 for readability in print)
TITLE_SIZE = 19
LABEL_SIZE = 19
TICK_SIZE = 18
LEGEND_SIZE = 20

COLOR_CORRECT = "#4CAF50"
COLOR_INCORRECT = "#EF5350"
COLOR_CORRECT_LIGHT = "#66BB6A"
COLOR_INCORRECT_LIGHT = "#FF7675"


def stacked_hist(ax, subset, total_count, bin_edges, color_correct, color_incorrect):
    correct = subset.loc[subset["correct"] == 1, "confidence"]
    incorrect = subset.loc[subset["correct"] == 0, "confidence"]

    hist_correct, _ = np.histogram(correct, bins=bin_edges)
    hist_incorrect, _ = np.histogram(incorrect, bins=bin_edges)

    hist_correct = hist_correct / total_count
    hist_incorrect = hist_incorrect / total_count

    centers = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(bin_edges) - 1)]
    ax.bar(centers, hist_correct, width=0.04, label="Correct", color=color_correct,
           alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.bar(centers, hist_incorrect, width=0.04, bottom=hist_correct, label="Incorrect",
           color=color_incorrect, alpha=0.85, edgecolor="white", linewidth=0.5)


def style_axis(ax):
    ax.set_ylim(0, 0.3)
    ax.set_xlim(-0.03, 1.03)
    ax.set_xticks(np.linspace(0, 1, 11))
    ax.set_xticklabels([f"{x:.1f}" for x in np.linspace(0, 1, 11)])
    ax.set_yticks([0.0, 0.1, 0.2, 0.3])
    ax.tick_params(labelsize=TICK_SIZE)
    ax.grid(axis="y", linestyle="-", alpha=0.2, linewidth=0.5, color="gray")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def make_figure(csv_name, display_label, reward_setting, stem):
    df = pd.read_csv(os.path.join(DATA_DIR, csv_name))

    df.loc[df["idk_flag"] == 0, "confidence"] = df.loc[df["idk_flag"] == 0, "first_confidence"]
    df.loc[df["idk_flag"] == 1, "confidence"] = df.loc[df["idk_flag"] == 1, "best_guess_confidence"]
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    df = df.dropna(subset=["confidence"])
    df["confidence"] = df["confidence"].clip(0, 1)

    bin_edges = list(np.linspace(0, 1, 21))
    total_count = len(df)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 3), sharey=True)

    stacked_hist(ax1, df[df["idk_flag"] == 0], total_count, bin_edges,
                 COLOR_CORRECT, COLOR_INCORRECT)
    ax1.set_title(f"First-round Answer ({display_label}; {reward_setting})",
                  fontsize=TITLE_SIZE, fontweight="bold", pad=10)
    ax1.set_xlabel("Confidence", fontweight="bold", fontsize=LABEL_SIZE)
    ax1.set_ylabel("Proportion", fontweight="bold", fontsize=LABEL_SIZE)
    style_axis(ax1)

    stacked_hist(ax2, df[df["idk_flag"] == 1], total_count, bin_edges,
                 COLOR_CORRECT_LIGHT, COLOR_INCORRECT_LIGHT)
    ax2.set_title(f"Best Guess ({display_label}; {reward_setting})",
                  fontsize=TITLE_SIZE, fontweight="bold", pad=10)
    ax2.set_xlabel("Confidence", fontweight="bold", fontsize=LABEL_SIZE)
    style_axis(ax2)

    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles, labels, loc="upper right", frameon=True, fancybox=False,
               shadow=False, edgecolor="black", framealpha=0.95, fontsize=LEGEND_SIZE)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        path = os.path.join(OUT_DIR, f"{stem}.{ext}")
        plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {path}")
    plt.close(fig)


def main():
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    plt.rcParams["axes.linewidth"] = 1
    plt.rcParams["xtick.major.width"] = 1
    plt.rcParams["ytick.major.width"] = 1

    os.makedirs(OUT_DIR, exist_ok=True)
    for csv_name, label, reward, stem in FIGURES:
        make_figure(csv_name, label, reward, stem)


if __name__ == "__main__":
    main()
