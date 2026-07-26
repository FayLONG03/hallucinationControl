"""Cumulative false-answer rate vs. uncertainty signal, for three PopQA runs:

  - token-prob baseline : popQA-idk-*.csv        (confidence = api_confidence_avg)
  - verbal-conf baseline: popQA-verbal-*.csv      (confidence = first_confidence
                           if idk_flag == 0 else best_guess_confidence)
  - I-CALM              : popqa_scheme_b_norm-*.csv (confidence = first_confidence
                           if idk_flag == 0 else best_guess_confidence)

For every row, "confidence" is the model's confidence in whatever answer counts
as final: first_answer when idk_flag == 0, best_guess when idk_flag == 1. This
matches the combination convention already used in main_exp/popQA/eval.py.
Uncertainty = 1 - confidence. Rows are sorted by uncertainty ascending, and the
cumulative false-answer rate is the running mean of (1 - correct) up to each
uncertainty threshold (i.e. a wrong answer counts as false regardless of the
row's idk_flag; we deliberately do not use the false_answer_flag column, since
for I-CALM it is defined to be 0 for every abstained row regardless of
whether the guessed content was actually correct).
"""
import os
import pandas as pd
import matplotlib.pyplot as plt

PLOT_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS = {
    "llama-3-8b-instruct": {
        "label": "Llama-3-8B-Instruct",
        "idk_csv": "popQA-idk-llama-3-8b-instruct-20260711_124753.csv",
        "verbal_csv": "popQA-verbal-llama-3-8b-instruct-20260711_151323.csv",
        "icalm_csv": "popqa_scheme_b_norm_results_+1_-1_+0.4_20260320_225145.csv",
        "out_png": "cumulative_false_rate.png",
        "x_as_quantile": True,
    },
    "gpt-5-mini": {
        "label": "GPT-5-mini",
        # GPT-5-mini is a reasoning model: it doesn't support logprobs, so
        # api_confidence_avg/min are empty for every row in its idk file and
        # the token-prob baseline can't be computed (see baseline/run.py:224).
        "idk_csv": None,
        "verbal_csv": "popQA-verbal-openai_gpt-5-mini-20260716_190013 21.49.52.csv",
        "icalm_csv": "popqa_gpt5mini_B_norm_results_+1_-1_+0.4_20260117_165433 21.54.35.csv",
        "out_png": "cumulative_false_rate_gpt-5-mini.png",
        "auto_ylim": True,
    },
}

COLORS = {
    "Token-prob baseline": "#2a78d6",
    "Verbal-conf baseline": "#eb6834",
    "I-CALM": "#1baf7a",
}


def load_confidence(csv_path, conf_col=None):
    """Return a DataFrame with columns [confidence, is_false].

    If conf_col is given, use it directly (already the combined per-row
    confidence). Otherwise combine first_confidence / best_guess_confidence
    by idk_flag, as done in main_exp/popQA/eval.py.

    is_false = 1 - correct (not the false_answer_flag column, see module
    docstring).
    """
    df = pd.read_csv(csv_path)

    if conf_col is not None:
        df["confidence"] = pd.to_numeric(df[conf_col], errors="coerce")
    else:
        df["confidence"] = pd.NA
        df.loc[df["idk_flag"] == 0, "confidence"] = df.loc[df["idk_flag"] == 0, "first_confidence"]
        df.loc[df["idk_flag"] == 1, "confidence"] = df.loc[df["idk_flag"] == 1, "best_guess_confidence"]
        df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")

    df = df.dropna(subset=["confidence", "correct"]).copy()
    df["confidence"] = df["confidence"].clip(0, 1)
    df["is_false"] = 1 - df["correct"].astype(int)
    return df[["confidence", "is_false"]]


def cumulative_false_rate(df, x_as_quantile=False):
    """Sort by uncertainty ascending and compute the running false-answer rate.

    If x_as_quantile, the x-axis is each row's own-file quantile rank of
    uncertainty (fraction of that file's rows at or below it) instead of the
    raw uncertainty value, so the three files' differing confidence scales
    become directly comparable.
    """
    df = df.copy()
    df["uncertainty"] = 1 - df["confidence"]
    df = df.sort_values("uncertainty").reset_index(drop=True)
    df["cum_false_rate"] = df["is_false"].cumsum() / (df.index + 1)
    if x_as_quantile:
        x = (df.index + 1) / len(df)
    else:
        x = df["uncertainty"].values
    return x, df["cum_false_rate"].values


def plot_model(model_key):
    cfg = MODELS[model_key]
    runs = {}
    if cfg["idk_csv"] is not None:
        runs["Token-prob baseline"] = load_confidence(
            os.path.join(PLOT_DIR, cfg["idk_csv"]), conf_col="api_confidence_avg"
        )
    runs["Verbal-conf baseline"] = load_confidence(os.path.join(PLOT_DIR, cfg["verbal_csv"]), conf_col=None)
    runs["I-CALM"] = load_confidence(os.path.join(PLOT_DIR, cfg["icalm_csv"]), conf_col=None)

    plt.rcParams.update({
        "font.size": 17,
        "axes.labelsize": 19,
        "legend.fontsize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "lines.linewidth": 2,
    })

    x_as_quantile = cfg.get("x_as_quantile", False)

    fig, ax = plt.subplots(figsize=(7, 5))
    y_max = 0.0
    for name, df in runs.items():
        x, y = cumulative_false_rate(df, x_as_quantile=x_as_quantile)
        print(f"[{model_key}] {name}: n={len(x)}, overall false rate={df['is_false'].mean():.4f}")
        ax.plot(x, y, label=name, color=COLORS[name])
        y_max = max(y_max, y.max())

    ax.set_xlabel("Uncertainty quantile (within each file)" if x_as_quantile else "Uncertainty signal (1 - confidence)")
    ax.set_ylabel("Cumulative false-answer rate")
    ax.set_xlim(0, 1)
    if cfg.get("auto_ylim", False):
        ax.set_ylim(0, y_max * 1.1)
    else:
        ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()

    out_path = os.path.join(PLOT_DIR, cfg["out_png"])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    for model_key in MODELS:
        plot_model(model_key)


if __name__ == "__main__":
    main()
