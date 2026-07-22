import pandas as pd
import os
import glob
from scipy.stats import binom


# Modify confidence interval output to show +/- format
def calculate_ci_pm(successes, trials, confidence=0.95):
    if trials == 0:
        return 0
    p_hat = successes / trials
    margin = binom.ppf(1 - (1 - confidence) / 2, trials, p_hat) / trials - p_hat
    return margin


def summarize(df, label):

        acc = df["correct"].mean()
        answered = df["idk_flag"] == 0
        guess = df["idk_flag"] == 1  # second-round best-guess rows
        total = len(df)

        far_e1 = df.loc[answered, "false_answer_flag"].sum() / answered.sum()
        far_e2 = 1 - acc
        far_e1_margin = calculate_ci_pm(df.loc[answered, "false_answer_flag"].sum(), answered.sum())
        far_e2_margin = calculate_ci_pm(len(df) - df["correct"].sum(), len(df))

        coverage = answered.sum() / total if total else float("nan")
        # Best-guess wrongness uses `correct`, not false_answer_flag: for IDK rows the
        # model abstained, so false_answer_flag is 0 even when the best guess is wrong.
        guess_wrong = ((df["idk_flag"] == 1) & (df["correct"] == 0)).sum()
        far_guess = (guess_wrong / guess.sum()) if guess.sum() else float("nan")
        far_guess_margin = calculate_ci_pm(guess_wrong, guess.sum())
        delta_risk = far_guess - far_e1

        # This ablation removes confidence elicitation, not the reward, so a
        # "score" column is still present.
        total_utility = df["score"].sum() if "score" in df.columns else float("nan")
        total_wrong = (df["correct"] == 0).sum()
        total_correct = (df["correct"] == 1).sum()
        answered_wrong = ((df["idk_flag"] == 0) & (df["correct"] == 0)).sum()
        answered_correct = ((df["idk_flag"] == 0) & (df["correct"] == 1)).sum()
        surfaced_wrong = df.loc[answered, "false_answer_flag"].sum()
        ecr = (guess_wrong / total_wrong) if total_wrong else float("nan")
        car = (answered_correct / total_correct) if total_correct else float("nan")
        hallucination_rate = (surfaced_wrong / total) if total else float("nan")

        # Printed in the same order as popQA/eval.py's summarize().
        print(f"\n📊 {label} Summary:")
        print(f"  Coverage: {coverage:.3f} ({answered.sum()}/{total})")
        print(f"  False-Answer Rate - Answered: {far_e1:.3f} ± {far_e1_margin:.3f}")
        print(f"  False-Answer Rate - Guess (2nd round): {far_guess:.3f} ± {far_guess_margin:.3f}")
        print(f"  False-Answer Rate - Overall: {far_e2:.3f} ± {far_e2_margin:.3f}")
        print(f"  Delta Risk (FAR guess - FAR answered): {delta_risk:.3f}")
        print(f"  Total Utility: {total_utility}")
        print(f"  Answered questions: {answered.sum()} out of {total}")
        print(f"  Wrong answered questions: {answered_wrong} out of {answered.sum()}")
        print(f"  Total wrong answered questions: {total_wrong}")
        print(f"  ECR (abstained wrong / all wrong): {ecr:.3f}")
        print(f"  CAR (answered correct / all correct): {car:.3f}")
        print(f"  Hallucination Rate (surfaced wrong / total): {hallucination_rate:.3f}")


def get_latest_file(pattern):
    """Find the most recent file matching the pattern."""
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No files found matching pattern: {pattern}")
    # Sort by modification time, most recent first
    latest_file = max(files, key=os.path.getmtime)
    return latest_file


def main():

    pattern = "outputs/popqa_*_results_*.csv"

    file = get_latest_file(pattern)

    df = pd.read_csv(file)

    print(f"📊 Loaded data file:")
    print(f"  no confidence: {file}")

    summarize(df, "no confidence")


if __name__ == "__main__":
    main()
