#!/usr/bin/env python3
"""
Coverage-matched abstention baseline for PopQA-style CSV logs.

This script supports the supplementary comparison where we:
1. Take a reference policy CSV (for example Scheme B Norm).
2. Read its realized coverage from `idk_flag == 0`.
3. Take a baseline CSV with a scalar confidence score per row
   (for example `self_confidence` or `api_confidence_avg`).
4. Rank rows by that score, accept the top-k rows matching the reference
   coverage exactly, and treat the rest as abstentions.

The baseline can therefore be compared to the reference policy using the same
coverage, while keeping the confidence source explicit.

Metric definitions used here:
- coverage = surfaced answers / total
- far_answered = surfaced wrong / surfaced answers
- far_overall = total wrong / total
- surfaced_error = surfaced wrong / total
- abstention_precision = answered correct / overall correct
- aer = abstained wrong / total wrong
- total_utility = answered_correct * reward_correct
                  + surfaced_wrong * reward_incorrect
                  + abstentions * reward_abstain

For synthetic threshold baselines, "surfaced wrong" is counted from accepted
rows with `correct == 0`, even if those rows were originally `idk_flag == 1`
in the logged run. This is intentional: the thresholded policy overrides the
original abstain/surface decision.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute coverage-matched threshold baselines from PopQA CSV logs."
    )
    parser.add_argument(
        "--baseline-csv",
        type=Path,
        required=True,
        help="CSV to threshold, e.g. preliminary PopQA verbal-confidence or pure-eval logs.",
    )
    parser.add_argument(
        "--score-col",
        type=str,
        required=True,
        help="Confidence column used for ranking, e.g. self_confidence or api_confidence_avg.",
    )
    parser.add_argument(
        "--reference-csv",
        type=Path,
        default=None,
        help="Optional CSV whose answered coverage (idk_flag == 0) defines the target coverage.",
    )
    parser.add_argument(
        "--reference-score-col",
        type=str,
        default=None,
        help=(
            "Optional confidence column for the reference CSV if you also want "
            "accepted-only Brier/ECE for the reference policy."
        ),
    )
    parser.add_argument(
        "--target-answered",
        type=int,
        default=None,
        help="Optional explicit target number of surfaced answers.",
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=10,
        help="Number of bins for accepted-only ECE. Default: 10.",
    )
    parser.add_argument(
        "--reward-correct",
        type=float,
        default=1.0,
        help="Utility for a correct surfaced answer. Default: 1.0.",
    )
    parser.add_argument(
        "--reward-incorrect",
        type=float,
        default=-1.0,
        help="Utility for an incorrect surfaced answer. Default: -1.0.",
    )
    parser.add_argument(
        "--reward-abstain",
        type=float,
        default=0.4,
        help="Utility for an abstention. Default: 0.4.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional JSON path for saving the full report.",
    )
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.95,
        help="Confidence level for CI calculations. Default: 0.95.",
    )
    return parser.parse_args()


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    value = value.strip()
    if value == "":
        return None
    return float(value)


def parse_int(value: str | None) -> int | None:
    parsed = parse_float(value)
    if parsed is None:
        return None
    return int(parsed)


def load_rows(csv_path: Path, score_col: str | None = None) -> list[dict]:
    rows: list[dict] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if score_col is not None and reader.fieldnames is not None and score_col not in reader.fieldnames:
            raise ValueError(f"Column `{score_col}` not found in {csv_path}")
        for index, row in enumerate(reader):
            score_value = None
            if score_col is not None:
                score_value = parse_float(row.get(score_col))

            dataset_id = parse_int(row.get("dataset_id"))
            if dataset_id is None:
                dataset_id = index

            correct = parse_int(row.get("correct"))
            if correct is None:
                raise ValueError(f"Missing `correct` value in {csv_path} row {index}")

            idk_flag = parse_int(row.get("idk_flag"))
            if idk_flag is None:
                raise ValueError(f"Missing `idk_flag` value in {csv_path} row {index}")

            false_answer_flag = parse_int(row.get("false_answer_flag"))
            if false_answer_flag is None:
                false_answer_flag = 1 - correct

            rows.append(
                {
                    "dataset_id": dataset_id,
                    "correct": correct,
                    "idk_flag": idk_flag,
                    "false_answer_flag": false_answer_flag,
                    "score": score_value,
                }
            )
    return rows


def mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return float("nan")
    return sum(values) / len(values)


def calculate_brier(rows: list[dict]) -> float:
    valid = [row for row in rows if row["score"] is not None and math.isfinite(row["score"])]
    if not valid:
        return float("nan")
    return mean((max(0.0, min(1.0, row["score"])) - row["correct"]) ** 2 for row in valid)


def z_value(confidence_level: float) -> float:
    return NormalDist().inv_cdf(1 - (1 - confidence_level) / 2)


def wilson_ci(successes: int, trials: int, confidence_level: float = 0.95) -> tuple[float, float]:
    if trials == 0:
        return (float("nan"), float("nan"))

    z = z_value(confidence_level)
    p_hat = successes / trials
    denom = 1 + (z ** 2) / trials
    center = (p_hat + (z ** 2) / (2 * trials)) / denom
    margin = (
        z
        * math.sqrt((p_hat * (1 - p_hat) + (z ** 2) / (4 * trials)) / trials)
        / denom
    )
    return (max(0.0, center - margin), min(1.0, center + margin))


def ci_to_pm(ci: tuple[float, float]) -> float:
    low, high = ci
    if math.isnan(low) or math.isnan(high):
        return float("nan")
    return (high - low) / 2


def calculate_brier_with_ci(
    rows: list[dict],
    confidence_level: float = 0.95,
) -> tuple[float, tuple[float, float], float]:
    valid = [row for row in rows if row["score"] is not None and math.isfinite(row["score"])]
    if not valid:
        return (float("nan"), (float("nan"), float("nan")), float("nan"))

    errors = [
        (max(0.0, min(1.0, row["score"])) - row["correct"]) ** 2
        for row in valid
    ]
    brier = mean(errors)
    n = len(errors)

    if n <= 1:
        return (brier, (float("nan"), float("nan")), float("nan"))

    variance = sum((error - brier) ** 2 for error in errors) / (n - 1)
    std_err = math.sqrt(variance / n)
    margin = z_value(confidence_level) * std_err
    return (brier, (brier - margin, brier + margin), margin)


def calculate_ece(rows: list[dict], num_bins: int = 10) -> float:
    valid = [row for row in rows if row["score"] is not None and math.isfinite(row["score"])]
    if not valid:
        return float("nan")

    # Match the repository convention: partition [0, 1] into equal-width bins.
    bins = [i / num_bins for i in range(num_bins + 1)]
    ece = 0.0
    total = len(valid)

    for bin_idx in range(num_bins):
        low = bins[bin_idx]
        high = bins[bin_idx + 1]
        if bin_idx == 0:
            members = [row for row in valid if low <= max(0.0, min(1.0, row["score"])) <= high]
        else:
            members = [row for row in valid if low < max(0.0, min(1.0, row["score"])) <= high]

        if not members:
            continue

        avg_conf = mean(max(0.0, min(1.0, row["score"])) for row in members)
        accuracy = mean(row["correct"] for row in members)
        ece += (len(members) / total) * abs(accuracy - avg_conf)

    return ece


def base_metrics(
    rows: list[dict],
    answered_rows: list[dict],
    abstained_rows: list[dict],
    *,
    surfaced_wrong_from_false_answer_flag: bool,
    reward_correct: float,
    reward_incorrect: float,
    reward_abstain: float,
    confidence_level: float,
) -> dict:
    total = len(rows)
    answered_n = len(answered_rows)
    abstained_n = len(abstained_rows)

    if surfaced_wrong_from_false_answer_flag:
        surfaced_wrong = sum(row["false_answer_flag"] for row in answered_rows)
    else:
        surfaced_wrong = sum(1 - row["correct"] for row in answered_rows)
    abstained_wrong = sum(1 - row["correct"] for row in abstained_rows)
    total_wrong = sum(1 - row["correct"] for row in rows)
    answered_correct = answered_n - surfaced_wrong
    total_correct = total - total_wrong
    total_utility = (
        answered_correct * reward_correct
        + surfaced_wrong * reward_incorrect
        + abstained_n * reward_abstain
    )
    far_answered_ci_95 = wilson_ci(surfaced_wrong, answered_n, confidence_level)
    far_overall_ci_95 = wilson_ci(total_wrong, total, confidence_level)

    return {
        "total_examples": total,
        "answered_examples": answered_n,
        "abstained_examples": abstained_n,
        "answered_correct": answered_correct,
        "coverage": answered_n / total if total else float("nan"),
        "surfaced_wrong": surfaced_wrong,
        "incorrect_answered": surfaced_wrong,
        "abstained_wrong": abstained_wrong,
        "total_wrong": total_wrong,
        "incorrect_overall": total_wrong,
        "total_correct": total_correct,
        "far_answered": surfaced_wrong / answered_n if answered_n else float("nan"),
        "far_answered_ci_95": far_answered_ci_95,
        "far_answered_ci_pm": ci_to_pm(far_answered_ci_95),
        "far_overall": total_wrong / total if total else float("nan"),
        "far_overall_ci_95": far_overall_ci_95,
        "far_overall_ci_pm": ci_to_pm(far_overall_ci_95),
        "surfaced_error": surfaced_wrong / total if total else float("nan"),
        "abstention_precision": (
            answered_correct / total_correct if total_correct else float("nan")
        ),
        # Under the paper definition, AER is the abstention detector's recall on
        # overall-incorrect examples.
        "abstention_recall": abstained_wrong / total_wrong if total_wrong else float("nan"),
        "aer": abstained_wrong / total_wrong if total_wrong else float("nan"),
        "total_utility": total_utility,
        "reward_correct": reward_correct,
        "reward_incorrect": reward_incorrect,
        "reward_abstain": reward_abstain,
    }


def compute_reference_policy_metrics(
    rows: list[dict],
    *,
    num_bins: int = 10,
    reward_correct: float,
    reward_incorrect: float,
    reward_abstain: float,
    confidence_level: float,
) -> dict:
    answered_rows = [row for row in rows if row["idk_flag"] == 0]
    abstained_rows = [row for row in rows if row["idk_flag"] == 1]

    report = base_metrics(
        rows,
        answered_rows,
        abstained_rows,
        surfaced_wrong_from_false_answer_flag=True,
        reward_correct=reward_correct,
        reward_incorrect=reward_incorrect,
        reward_abstain=reward_abstain,
        confidence_level=confidence_level,
    )
    report["policy_type"] = "logged_reference_policy"
    answered_brier, answered_brier_ci_95, answered_brier_ci_pm = calculate_brier_with_ci(
        answered_rows,
        confidence_level=confidence_level,
    )
    overall_brier, overall_brier_ci_95, overall_brier_ci_pm = calculate_brier_with_ci(
        rows,
        confidence_level=confidence_level,
    )
    report["accepted_only_brier"] = answered_brier
    report["accepted_only_brier_ci_95"] = answered_brier_ci_95
    report["accepted_only_brier_ci_pm"] = answered_brier_ci_pm
    report["accepted_only_ece"] = calculate_ece(
        [row for row in answered_rows if row["score"] is not None],
        num_bins=num_bins,
    )
    report["brier_answered"] = report["accepted_only_brier"]
    report["brier_answered_ci_95"] = report["accepted_only_brier_ci_95"]
    report["brier_answered_ci_pm"] = report["accepted_only_brier_ci_pm"]
    report["ece_answered"] = report["accepted_only_ece"]
    report["overall_brier"] = overall_brier
    report["overall_brier_ci_95"] = overall_brier_ci_95
    report["overall_brier_ci_pm"] = overall_brier_ci_pm
    report["overall_ece"] = calculate_ece(rows, num_bins=num_bins)
    report["brier_overall"] = report["overall_brier"]
    report["brier_overall_ci_95"] = report["overall_brier_ci_95"]
    report["brier_overall_ci_pm"] = report["overall_brier_ci_pm"]
    report["ece_overall"] = report["overall_ece"]
    return report


def compute_threshold_policy_metrics(
    rows: list[dict],
    target_answered: int,
    score_col: str,
    *,
    num_bins: int = 10,
    reward_correct: float,
    reward_incorrect: float,
    reward_abstain: float,
    confidence_level: float,
) -> dict:
    total = len(rows)
    if target_answered < 0 or target_answered > total:
        raise ValueError(
            f"target_answered must be between 0 and {total}, got {target_answered}"
        )

    # Missing scores are sent to the bottom so they abstain first.
    sorted_rows = sorted(
        rows,
        key=lambda row: (
            math.inf if row["score"] is None else -row["score"],
            row["dataset_id"],
        ),
    )

    accepted_rows = sorted_rows[:target_answered]
    abstained_rows = sorted_rows[target_answered:]

    report = base_metrics(
        rows,
        accepted_rows,
        abstained_rows,
        surfaced_wrong_from_false_answer_flag=False,
        reward_correct=reward_correct,
        reward_incorrect=reward_incorrect,
        reward_abstain=reward_abstain,
        confidence_level=confidence_level,
    )
    report["policy_type"] = "coverage_matched_threshold"
    report["score_col"] = score_col
    report["target_answered"] = target_answered
    report["tie_break"] = "dataset_id ascending after score descending"
    report["threshold_score"] = (
        accepted_rows[-1]["score"] if accepted_rows else float("nan")
    )
    report["accepted_missing_score"] = sum(row["score"] is None for row in accepted_rows)
    report["abstained_missing_score"] = sum(row["score"] is None for row in abstained_rows)
    report["accepted_rows_originally_answered"] = sum(
        row["idk_flag"] == 0 for row in accepted_rows
    )
    report["accepted_rows_originally_idk"] = sum(
        row["idk_flag"] == 1 for row in accepted_rows
    )
    answered_brier, answered_brier_ci_95, answered_brier_ci_pm = calculate_brier_with_ci(
        accepted_rows,
        confidence_level=confidence_level,
    )
    overall_brier, overall_brier_ci_95, overall_brier_ci_pm = calculate_brier_with_ci(
        rows,
        confidence_level=confidence_level,
    )
    report["accepted_only_brier"] = answered_brier
    report["accepted_only_brier_ci_95"] = answered_brier_ci_95
    report["accepted_only_brier_ci_pm"] = answered_brier_ci_pm
    report["accepted_only_ece"] = calculate_ece(accepted_rows, num_bins=num_bins)
    report["threshold"] = report["threshold_score"]
    report["brier_answered"] = report["accepted_only_brier"]
    report["brier_answered_ci_95"] = report["accepted_only_brier_ci_95"]
    report["brier_answered_ci_pm"] = report["accepted_only_brier_ci_pm"]
    report["ece_answered"] = report["accepted_only_ece"]
    report["overall_brier"] = overall_brier
    report["overall_brier_ci_95"] = overall_brier_ci_95
    report["overall_brier_ci_pm"] = overall_brier_ci_pm
    report["overall_ece"] = calculate_ece(rows, num_bins=num_bins)
    report["brier_overall"] = report["overall_brier"]
    report["brier_overall_ci_95"] = report["overall_brier_ci_95"]
    report["brier_overall_ci_pm"] = report["overall_brier_ci_pm"]
    report["ece_overall"] = report["overall_ece"]
    return report


def round_floats(value):
    if isinstance(value, dict):
        return {key: round_floats(val) for key, val in value.items()}
    if isinstance(value, list):
        return [round_floats(item) for item in value]
    if isinstance(value, tuple):
        return [round_floats(item) for item in value]
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return value
        return round(value, 6)
    return value


def print_metrics(label: str, metrics: dict) -> None:
    print(f"\n{label}")
    print("-" * len(label))
    print(
        f"coverage: {metrics['coverage']:.6f} "
        f"({metrics['answered_examples']}/{metrics['total_examples']})"
    )
    if "threshold" in metrics or "threshold_score" in metrics:
        print(
            f"threshold: {metrics.get('threshold', metrics.get('threshold_score', float('nan')))}"
        )
    print(
        f"far_answered: {metrics['far_answered']:.6f} "
        f"(95% CI: [{metrics['far_answered_ci_95'][0]:.6f}, {metrics['far_answered_ci_95'][1]:.6f}], "
        f"± {metrics['far_answered_ci_pm']:.6f})"
    )
    print(
        f"far_overall: {metrics['far_overall']:.6f} "
        f"(95% CI: [{metrics['far_overall_ci_95'][0]:.6f}, {metrics['far_overall_ci_95'][1]:.6f}], "
        f"± {metrics['far_overall_ci_pm']:.6f})"
    )
    print(f"total_utility: {metrics['total_utility']:.6f}")
    print(f"incorrect_answered: {metrics['incorrect_answered']}")
    print(f"incorrect_overall: {metrics['incorrect_overall']}")
    print(f"surfaced_error: {metrics['surfaced_error']:.6f}")
    print(f"abstention_precision: {metrics['abstention_precision']:.6f}")
    print(f"abstention_recall: {metrics['abstention_recall']:.6f}")
    print(f"aer: {metrics['aer']:.6f}")
    print(
        f"accepted_only_brier: {metrics['accepted_only_brier']:.6f} "
        f"(95% CI: [{metrics['accepted_only_brier_ci_95'][0]:.6f}, {metrics['accepted_only_brier_ci_95'][1]:.6f}], "
        f"± {metrics['accepted_only_brier_ci_pm']:.6f})"
    )
    print(f"accepted_only_ece: {metrics['accepted_only_ece']:.6f}")
    print(
        f"overall_brier: {metrics['overall_brier']:.6f} "
        f"(95% CI: [{metrics['overall_brier_ci_95'][0]:.6f}, {metrics['overall_brier_ci_95'][1]:.6f}], "
        f"± {metrics['overall_brier_ci_pm']:.6f})"
    )
    print(f"overall_ece: {metrics['overall_ece']:.6f}")

    if "threshold_score" in metrics:
        print(f"threshold_score: {metrics['threshold_score']}")
        print(
            "accepted rows that were originally IDK: "
            f"{metrics['accepted_rows_originally_idk']}"
        )
        print(
            "abstained rows with missing score: "
            f"{metrics['abstained_missing_score']}"
        )


def main() -> None:
    args = parse_args()

    if args.reference_csv is None and args.target_answered is None:
        raise ValueError("Provide either --reference-csv or --target-answered.")

    baseline_rows = load_rows(args.baseline_csv, score_col=args.score_col)

    reference_report = None
    target_answered = args.target_answered

    if args.reference_csv is not None:
        reference_rows = load_rows(args.reference_csv, score_col=args.reference_score_col)
        if len(reference_rows) != len(baseline_rows):
            raise ValueError(
                "Reference and baseline CSVs must have the same number of rows to "
                "match coverage on the same dataset."
            )
        reference_report = compute_reference_policy_metrics(
            reference_rows,
            num_bins=args.num_bins,
            reward_correct=args.reward_correct,
            reward_incorrect=args.reward_incorrect,
            reward_abstain=args.reward_abstain,
            confidence_level=args.confidence_level,
        )
        target_answered = reference_report["answered_examples"]

    assert target_answered is not None

    baseline_report = compute_threshold_policy_metrics(
        baseline_rows,
        target_answered=target_answered,
        score_col=args.score_col,
        num_bins=args.num_bins,
        reward_correct=args.reward_correct,
        reward_incorrect=args.reward_incorrect,
        reward_abstain=args.reward_abstain,
        confidence_level=args.confidence_level,
    )

    payload = {
        "baseline_csv": str(args.baseline_csv),
        "score_col": args.score_col,
        "reference_csv": str(args.reference_csv) if args.reference_csv else None,
        "confidence_level": args.confidence_level,
        "reference_policy": round_floats(reference_report) if reference_report else None,
        "coverage_matched_baseline": round_floats(baseline_report),
    }

    if reference_report is not None:
        print_metrics("Reference policy", reference_report)
    print_metrics("Coverage-matched baseline", baseline_report)

    print("\nJSON summary")
    print("------------")
    print(json.dumps(round_floats(payload), indent=2, ensure_ascii=True))

    if args.output_json is not None:
        args.output_json.write_text(
            json.dumps(round_floats(payload), indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        print(f"\nSaved JSON report to {args.output_json}")


if __name__ == "__main__":
    main()
