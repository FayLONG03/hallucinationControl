#!/usr/bin/env python3
"""Evaluation: all metrics for the coverage-matched abstention baseline.

This is the metric engine of the baseline experiment, refactored from
``preliminary_exp/coverage_matched_baseline.py`` into a reusable module. Given
a baseline CSV with a scalar confidence column, it ranks rows by that score,
accepts the top-k to match a target coverage (either an explicit count or the
answered coverage of a reference policy CSV), treats the rest as abstentions,
and reports the full metric suite.

Metric definitions:
- coverage             = surfaced answers / total
- far_answered         = surfaced wrong / surfaced answers
- far_overall          = total wrong / total
- surfaced_error       = surfaced wrong / total
- abstention_precision = answered correct / overall correct
- aer / recall         = abstained wrong / total wrong
- total_utility        = answered_correct * reward_correct
                         + surfaced_wrong * reward_incorrect
                         + abstentions * reward_abstain
- accepted_only / overall Brier and ECE from the confidence column

For synthetic threshold baselines, "surfaced wrong" is counted from accepted
rows with ``correct == 0``, even if those rows were originally ``idk_flag == 1``
in the logged run: the thresholded policy overrides the original decision.

Usable both as a library (`evaluate(...)`) and as a CLI.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Iterable


# ---------------------------------------------------------------------------
# Row loading
# ---------------------------------------------------------------------------

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
    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if (
            score_col is not None
            and reader.fieldnames is not None
            and score_col not in reader.fieldnames
        ):
            raise ValueError(f"Column `{score_col}` not found in {csv_path}")
        for index, row in enumerate(reader):
            score_value = parse_float(row.get(score_col)) if score_col is not None else None

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

            # Logged per-row reward from the CSV `score` column (distinct from the
            # confidence `score` used for ranking). Summed for total_utility.
            logged_score = parse_float(row.get("score"))

            rows.append(
                {
                    "dataset_id": dataset_id,
                    "correct": correct,
                    "idk_flag": idk_flag,
                    "false_answer_flag": false_answer_flag,
                    "score": score_value,
                    "logged_score": logged_score,
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return float("nan")
    return sum(values) / len(values)


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
        z * math.sqrt((p_hat * (1 - p_hat) + (z ** 2) / (4 * trials)) / trials) / denom
    )
    return (max(0.0, center - margin), min(1.0, center + margin))


def ci_to_pm(ci: tuple[float, float]) -> float:
    low, high = ci
    if math.isnan(low) or math.isnan(high):
        return float("nan")
    return (high - low) / 2


def calculate_brier_with_ci(
    rows: list[dict], confidence_level: float = 0.95
) -> tuple[float, tuple[float, float], float]:
    valid = [r for r in rows if r["score"] is not None and math.isfinite(r["score"])]
    if not valid:
        return (float("nan"), (float("nan"), float("nan")), float("nan"))

    errors = [(max(0.0, min(1.0, r["score"])) - r["correct"]) ** 2 for r in valid]
    brier = mean(errors)
    n = len(errors)
    if n <= 1:
        return (brier, (float("nan"), float("nan")), float("nan"))

    variance = sum((e - brier) ** 2 for e in errors) / (n - 1)
    std_err = math.sqrt(variance / n)
    margin = z_value(confidence_level) * std_err
    return (brier, (brier - margin, brier + margin), margin)


def calculate_ece(rows: list[dict], num_bins: int = 10) -> float:
    valid = [r for r in rows if r["score"] is not None and math.isfinite(r["score"])]
    if not valid:
        return float("nan")

    bins = [i / num_bins for i in range(num_bins + 1)]
    ece = 0.0
    total = len(valid)
    for bin_idx in range(num_bins):
        low, high = bins[bin_idx], bins[bin_idx + 1]
        if bin_idx == 0:
            members = [r for r in valid if low <= max(0.0, min(1.0, r["score"])) <= high]
        else:
            members = [r for r in valid if low < max(0.0, min(1.0, r["score"])) <= high]
        if not members:
            continue
        avg_conf = mean(max(0.0, min(1.0, r["score"])) for r in members)
        accuracy = mean(r["correct"] for r in members)
        ece += (len(members) / total) * abs(accuracy - avg_conf)
    return ece


# ---------------------------------------------------------------------------
# Metric assembly
# ---------------------------------------------------------------------------

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
        surfaced_wrong = sum(r["false_answer_flag"] for r in answered_rows)
    else:
        surfaced_wrong = sum(1 - r["correct"] for r in answered_rows)
    abstained_wrong = sum(1 - r["correct"] for r in abstained_rows)
    total_wrong = sum(1 - r["correct"] for r in rows)
    answered_correct = answered_n - surfaced_wrong
    total_correct = total - total_wrong
    # Total utility = sum of the logged per-row reward (CSV `score` column),
    # over all rows. Rows without a logged score are skipped.
    total_utility = sum(row["logged_score"] for row in rows if row.get("logged_score") is not None)
    far_answered_ci_95 = wilson_ci(surfaced_wrong, answered_n, confidence_level)
    far_overall_ci_95 = wilson_ci(total_wrong, total, confidence_level)

    # FAR (guess): among the abstained rows (those NOT accepted into the top-k),
    # the fraction whose answer would have been wrong had it been surfaced. This
    # is the risk avoided by abstaining, under this report's accept/abstain
    # policy. For the logged reference policy the abstained set is exactly the
    # model's IDK rows (idk_flag == 1).
    guess_rows = abstained_rows
    guess_n = len(guess_rows)
    guess_wrong = sum(1 - r["correct"] for r in guess_rows)
    far_guess = guess_wrong / guess_n if guess_n else float("nan")
    far_guess_ci_95 = wilson_ci(guess_wrong, guess_n, confidence_level)
    far_answered = surfaced_wrong / answered_n if answered_n else float("nan")
    delta_risk = far_guess - far_answered

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
        "far_answered": far_answered,
        "far_answered_ci_95": far_answered_ci_95,
        "far_answered_ci_pm": ci_to_pm(far_answered_ci_95),
        "far_guess": far_guess,
        "far_guess_ci_95": far_guess_ci_95,
        "far_guess_ci_pm": ci_to_pm(far_guess_ci_95),
        "guess_examples": guess_n,
        "guess_wrong": guess_wrong,
        "far_overall": total_wrong / total if total else float("nan"),
        "far_overall_ci_95": far_overall_ci_95,
        "far_overall_ci_pm": ci_to_pm(far_overall_ci_95),
        "delta_risk": delta_risk,
        "surfaced_error": surfaced_wrong / total if total else float("nan"),
        "abstention_precision": (
            answered_correct / total_correct if total_correct else float("nan")
        ),
        # AER is the abstention detector's recall on overall-incorrect examples.
        "abstention_recall": abstained_wrong / total_wrong if total_wrong else float("nan"),
        "aer": abstained_wrong / total_wrong if total_wrong else float("nan"),
        "total_utility": total_utility,
        "reward_correct": reward_correct,
        "reward_incorrect": reward_incorrect,
        "reward_abstain": reward_abstain,
    }


def _attach_calibration(
    report: dict,
    accepted_or_answered_rows: list[dict],
    all_rows: list[dict],
    *,
    num_bins: int,
    confidence_level: float,
) -> None:
    answered_brier, answered_brier_ci_95, answered_brier_ci_pm = calculate_brier_with_ci(
        accepted_or_answered_rows, confidence_level=confidence_level
    )
    overall_brier, overall_brier_ci_95, overall_brier_ci_pm = calculate_brier_with_ci(
        all_rows, confidence_level=confidence_level
    )
    report["accepted_only_brier"] = answered_brier
    report["accepted_only_brier_ci_95"] = answered_brier_ci_95
    report["accepted_only_brier_ci_pm"] = answered_brier_ci_pm
    report["accepted_only_ece"] = calculate_ece(
        [r for r in accepted_or_answered_rows if r["score"] is not None], num_bins=num_bins
    )
    report["brier_answered"] = report["accepted_only_brier"]
    report["brier_answered_ci_95"] = report["accepted_only_brier_ci_95"]
    report["brier_answered_ci_pm"] = report["accepted_only_brier_ci_pm"]
    report["ece_answered"] = report["accepted_only_ece"]
    report["overall_brier"] = overall_brier
    report["overall_brier_ci_95"] = overall_brier_ci_95
    report["overall_brier_ci_pm"] = overall_brier_ci_pm
    report["overall_ece"] = calculate_ece(all_rows, num_bins=num_bins)
    report["brier_overall"] = report["overall_brier"]
    report["brier_overall_ci_95"] = report["overall_brier_ci_95"]
    report["brier_overall_ci_pm"] = report["overall_brier_ci_pm"]
    report["ece_overall"] = report["overall_ece"]


# Canonical results-table order: (column label, value key, ci-margin key or None).
# Metrics with a margin key are rendered as "value ± margin" (95% CI half-width).
SUMMARY_FIELDS = [
    ("C (coverage)", "coverage", None),
    ("threshold", "threshold", None),
    ("FAR (answered)", "far_answered", "far_answered_ci_pm"),
    ("FAR (guess)", "far_guess", "far_guess_ci_pm"),
    ("FAR (overall)", "far_overall", "far_overall_ci_pm"),
    ("delta risk", "delta_risk", None),
    ("total utility", "total_utility", None),
    ("# answered", "answered_examples", None),
    ("# incorrect (answered)", "incorrect_answered", None),
    ("# incorrect (overall)", "incorrect_overall", None),
    ("ECR", "aer", None),
    ("CAR", "abstention_precision", None),
    ("hallucination rate", "surfaced_error", None),
    ("ECE (answered)", "ece_answered", None),
    ("ECE (overall)", "ece_overall", None),
    ("brier (answered)", "brier_answered", "brier_answered_ci_pm"),
    ("brier (overall)", "brier_overall", "brier_overall_ci_pm"),
]


def _fmt_pm(value, margin) -> str:
    """Render "value ± margin" for a 95% CI, e.g. "0.631 ± 0.008"."""
    if (
        isinstance(value, (int, float))
        and isinstance(margin, (int, float))
        and not (math.isnan(value) or math.isnan(margin))
    ):
        return f"{value:.3f} ± {margin:.3f}"
    return value


def summary_row(report: dict) -> dict:
    """Return the metrics as an ordered dict in the canonical results-table
    order. Metrics with a 95% CI are rendered as "value ± margin" strings;
    `threshold` is None for direct (non-thresholded) policies."""
    values = dict(report)
    values.setdefault("threshold", report.get("threshold_score"))
    out = {}
    for label, key, pm_key in SUMMARY_FIELDS:
        if pm_key is not None:
            out[label] = _fmt_pm(values.get(key), values.get(pm_key))
        else:
            out[label] = values.get(key)
    return out


def compute_reference_policy_metrics(
    rows: list[dict],
    *,
    num_bins: int = 10,
    reward_correct: float,
    reward_incorrect: float,
    reward_abstain: float,
    confidence_level: float,
) -> dict:
    answered_rows = [r for r in rows if r["idk_flag"] == 0]
    abstained_rows = [r for r in rows if r["idk_flag"] == 1]

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
    _attach_calibration(
        report, answered_rows, rows, num_bins=num_bins, confidence_level=confidence_level
    )
    report["summary"] = summary_row(report)
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
        raise ValueError(f"target_answered must be between 0 and {total}, got {target_answered}")

    # Missing scores sink to the bottom so they abstain first.
    sorted_rows = sorted(
        rows,
        key=lambda r: (math.inf if r["score"] is None else -r["score"], r["dataset_id"]),
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
    report["threshold_score"] = accepted_rows[-1]["score"] if accepted_rows else float("nan")
    report["accepted_missing_score"] = sum(r["score"] is None for r in accepted_rows)
    report["abstained_missing_score"] = sum(r["score"] is None for r in abstained_rows)
    report["accepted_rows_originally_answered"] = sum(r["idk_flag"] == 0 for r in accepted_rows)
    report["accepted_rows_originally_idk"] = sum(r["idk_flag"] == 1 for r in accepted_rows)
    _attach_calibration(
        report, accepted_rows, rows, num_bins=num_bins, confidence_level=confidence_level
    )
    report["threshold"] = report["threshold_score"]
    report["summary"] = summary_row(report)
    return report


# ---------------------------------------------------------------------------
# Top-level entry point (library API)
# ---------------------------------------------------------------------------

def evaluate(
    baseline_csv: Path | str,
    score_col: str,
    *,
    reference_csv: Path | str | None = None,
    reference_score_col: str | None = None,
    target_answered: int | None = None,
    num_bins: int = 10,
    reward_correct: float = 1.0,
    reward_incorrect: float = -1.0,
    reward_abstain: float = 0.4,
    confidence_level: float = 0.95,
) -> dict:
    """Compute the full report. Returns a payload dict with reference (optional)
    and coverage-matched baseline metrics.

    Provide either ``reference_csv`` (to match its answered coverage) or an
    explicit ``target_answered`` count.
    """
    if reference_csv is None and target_answered is None:
        raise ValueError("Provide either reference_csv or target_answered.")

    baseline_rows = load_rows(baseline_csv, score_col=score_col)
    reference_report = None

    if reference_csv is not None:
        reference_rows = load_rows(reference_csv, score_col=reference_score_col)
        if len(reference_rows) != len(baseline_rows):
            raise ValueError(
                "Reference and baseline CSVs must have the same number of rows to "
                "match coverage on the same dataset."
            )
        reference_report = compute_reference_policy_metrics(
            reference_rows,
            num_bins=num_bins,
            reward_correct=reward_correct,
            reward_incorrect=reward_incorrect,
            reward_abstain=reward_abstain,
            confidence_level=confidence_level,
        )
        target_answered = reference_report["answered_examples"]

    assert target_answered is not None
    baseline_report = compute_threshold_policy_metrics(
        baseline_rows,
        target_answered=target_answered,
        score_col=score_col,
        num_bins=num_bins,
        reward_correct=reward_correct,
        reward_incorrect=reward_incorrect,
        reward_abstain=reward_abstain,
        confidence_level=confidence_level,
    )

    return {
        "baseline_csv": str(baseline_csv),
        "score_col": score_col,
        "reference_csv": str(reference_csv) if reference_csv else None,
        "confidence_level": confidence_level,
        "reference_policy": reference_report,
        "coverage_matched_baseline": baseline_report,
    }


# ---------------------------------------------------------------------------
# Reference coverage (external reference_data_results runs)
# ---------------------------------------------------------------------------

# Root of the logged reference runs whose answered coverage the threshold
# baselines are matched to. Structured as ``<root>/<dataset>/<dataset>_<model>.csv``.
REFERENCE_ROOT = Path(__file__).resolve().parent / "reference_data_results"

# dataset.name -> (subfolder under REFERENCE_ROOT, reference filename prefix).
DATASET_REFERENCE_DIR = {
    "popQA": ("popQA", "popqa"),
    "triviaQA": ("triviaQA", "triviaqa"),
    "simpleQA-verified": ("simpleQA", "simpleqa"),
}

# Short model name (from ``client.model``, provider prefix stripped) -> the
# model tag used in the reference filenames.
MODEL_REFERENCE_TAG = {
    "llama-3-8b-instruct": "llama",
    "qwen3-4b-instruct": "qwen",
    "gpt-4o-mini": "gpt4omini",
    "gpt-5-mini": "gpt5mini",
    "gemini-3.1-flash-lite": "gemini",
}


def resolve_reference_csv(
    dataset_name: str,
    model: str,
    *,
    root: Path | str = REFERENCE_ROOT,
) -> Path | None:
    """Locate the reference run CSV for a (dataset, model) pair.

    Returns the path to ``<root>/<dataset>/<prefix>_<tag>.csv`` if it exists,
    otherwise ``None`` (e.g. datasets/models without a logged reference yet).
    """
    dataset_entry = DATASET_REFERENCE_DIR.get(dataset_name)
    if dataset_entry is None:
        return None
    subfolder, prefix = dataset_entry

    model_short = model.split("/")[-1]
    tag = MODEL_REFERENCE_TAG.get(model_short)
    if tag is None:
        return None

    candidate = Path(root) / subfolder / f"{prefix}_{tag}.csv"
    return candidate if candidate.exists() else None


def reference_for_run(dataset_name: str, model: str, *, matching: str) -> Path | None:
    """Resolve the reference run for a (dataset, model) pair and report the choice.

    ``matching`` names the score being coverage-matched (e.g. "token-prob"), and
    only shapes the log line. A ``None`` return is not an error: the caller then
    falls back to matching the run's own IDK coverage.
    """
    reference_csv = resolve_reference_csv(dataset_name, model)
    if reference_csv is None:
        print(
            f"No reference run found for ({dataset_name}, {model}); "
            "matching coverage to the run's own IDK policy."
        )
    else:
        print(f"Matching {matching} coverage to reference run {reference_csv}")
    return reference_csv


def reference_coverage(reference_csv: Path | str, baseline_total: int) -> dict:
    """Coverage target for a threshold baseline, taken from a reference run.

    Coverage is ``answered / all questions`` (``idk_flag == 0`` over the full
    question set). Questions the reference run dropped (gaps in ``dataset_id``)
    still count in the denominator as not-answered, so an incomplete reference
    log does not inflate the target; the full question count is inferred from
    the highest logged ``dataset_id``.

    The target is the reference run's answered count. When the reference and
    baseline question counts differ, the reference *coverage fraction* is
    applied to the baseline total instead, so the two remain coverage-matched
    on differently sized runs.
    """
    reference_rows = load_rows(reference_csv)
    max_id = max((r["dataset_id"] for r in reference_rows), default=-1)
    ref_total = max(len(reference_rows), max_id + 1)
    ref_answered = sum(1 for r in reference_rows if r["idk_flag"] == 0)
    ref_cov = ref_answered / ref_total if ref_total else float("nan")

    if ref_total == baseline_total:
        target = ref_answered
    else:
        target = round(ref_cov * baseline_total)
    target = max(0, min(baseline_total, target))

    return {
        "reference_csv": str(reference_csv),
        "reference_total": ref_total,
        "reference_rows_logged": len(reference_rows),
        "reference_missing_rows": ref_total - len(reference_rows),
        "reference_answered": ref_answered,
        "reference_coverage": ref_cov,
        "target_answered": target,
    }


# ---------------------------------------------------------------------------
# Experiment-level entry points
# ---------------------------------------------------------------------------

def evaluate_idk_and_token_prob(
    csv_path: Path | str,
    *,
    token_col: str = "api_confidence_avg",
    reference_csv: Path | str | None = None,
    num_bins: int = 10,
    reward_correct: float = 1.0,
    reward_incorrect: float = -1.0,
    reward_abstain: float = 0.4,
    confidence_level: float = 0.95,
) -> dict:
    """Exp 1: IDK & Token Prob.

    Two metric views over the *same* run:
      - ``idk_policy``: computed directly from the result file, using the
        model's own "I don't know" as the abstention decision (Brier/ECE use
        the token-probability column). Unaffected by the reference.
      - ``token_prob_threshold``: rank rows by the token-probability column and
        accept the top-k to match a target coverage. When ``reference_csv`` is
        given, the coverage is matched to that reference run's answered coverage
        (from ``reference_data_results``); otherwise it falls back to this run's
        own IDK coverage.
    """
    rows = load_rows(csv_path, score_col=token_col)
    idk_policy = compute_reference_policy_metrics(
        rows,
        num_bins=num_bins,
        reward_correct=reward_correct,
        reward_incorrect=reward_incorrect,
        reward_abstain=reward_abstain,
        confidence_level=confidence_level,
    )

    if reference_csv is not None:
        coverage_source = reference_coverage(reference_csv, len(rows))
        target_answered = coverage_source["target_answered"]
    else:
        coverage_source = None
        target_answered = idk_policy["answered_examples"]

    token_prob_threshold = compute_threshold_policy_metrics(
        rows,
        target_answered=target_answered,
        score_col=token_col,
        num_bins=num_bins,
        reward_correct=reward_correct,
        reward_incorrect=reward_incorrect,
        reward_abstain=reward_abstain,
        confidence_level=confidence_level,
    )
    return {
        "experiment": "idk_and_token_prob",
        "csv": str(csv_path),
        "token_col": token_col,
        "reference_csv": str(reference_csv) if reference_csv else None,
        "coverage_source": coverage_source,
        "idk_policy": idk_policy,
        "token_prob_threshold": token_prob_threshold,
    }


def evaluate_verbal_conf(
    csv_path: Path | str,
    *,
    conf_col: str = "self_confidence",
    reference_csv: Path | str | None = None,
    num_bins: int = 10,
    reward_correct: float = 1.0,
    reward_incorrect: float = -1.0,
    reward_abstain: float = 0.4,
    confidence_level: float = 0.95,
) -> dict:
    """Exp 2: Verbal Conf.

    Rank rows by the verbal confidence column and accept the top-k to match a
    target coverage (coverage-matched thresholding, as in
    coverage_matched_baseline.py). When ``reference_csv`` is given, the coverage
    is matched to that reference run's answered coverage (from
    ``reference_data_results``); otherwise it falls back to this run's own IDK
    coverage (``idk_flag == 0``).
    """
    rows = load_rows(csv_path, score_col=conf_col)

    if reference_csv is not None:
        coverage_source = reference_coverage(reference_csv, len(rows))
        target_answered = coverage_source["target_answered"]
    else:
        coverage_source = None
        target_answered = sum(1 for r in rows if r["idk_flag"] == 0)

    verbal_conf_threshold = compute_threshold_policy_metrics(
        rows,
        target_answered=target_answered,
        score_col=conf_col,
        num_bins=num_bins,
        reward_correct=reward_correct,
        reward_incorrect=reward_incorrect,
        reward_abstain=reward_abstain,
        confidence_level=confidence_level,
    )
    return {
        "experiment": "verbal_conf",
        "csv": str(csv_path),
        "conf_col": conf_col,
        "reference_csv": str(reference_csv) if reference_csv else None,
        "coverage_source": coverage_source,
        "verbal_conf_threshold": verbal_conf_threshold,
    }


# ---------------------------------------------------------------------------
# Output helpers + CLI
# ---------------------------------------------------------------------------

def round_floats(value):
    if isinstance(value, dict):
        return {k: round_floats(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [round_floats(v) for v in value]
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return value
        return round(value, 6)
    return value


def _format_value(value) -> str:
    if isinstance(value, (list, tuple)):
        if len(value) == 2 and all(isinstance(v, float) for v in value):
            return f"[{value[0]:.6f}, {value[1]:.6f}]"
        return str(value)
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6f}"
    if value is None:
        return "-"
    return str(value)


def print_metrics(label: str, metrics: dict) -> None:
    """Print the metrics in the canonical results-table order."""
    print(f"\n{label}")
    print("-" * len(label))
    for name, value in summary_row(metrics).items():
        print(f"{name}: {_format_value(value)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute coverage-matched threshold baselines from PopQA CSV logs."
    )
    parser.add_argument("--baseline-csv", type=Path, required=True)
    parser.add_argument("--score-col", type=str, required=True)
    parser.add_argument("--reference-csv", type=Path, default=None)
    parser.add_argument("--reference-score-col", type=str, default=None)
    parser.add_argument("--target-answered", type=int, default=None)
    parser.add_argument("--num-bins", type=int, default=10)
    parser.add_argument("--reward-correct", type=float, default=1.0)
    parser.add_argument("--reward-incorrect", type=float, default=-1.0)
    parser.add_argument("--reward-abstain", type=float, default=0.4)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = evaluate(
        baseline_csv=args.baseline_csv,
        score_col=args.score_col,
        reference_csv=args.reference_csv,
        reference_score_col=args.reference_score_col,
        target_answered=args.target_answered,
        num_bins=args.num_bins,
        reward_correct=args.reward_correct,
        reward_incorrect=args.reward_incorrect,
        reward_abstain=args.reward_abstain,
        confidence_level=args.confidence_level,
    )

    if payload["reference_policy"] is not None:
        print_metrics("Reference policy", payload["reference_policy"])
    print_metrics("Coverage-matched baseline", payload["coverage_matched_baseline"])

    if args.output_json is not None:
        args.output_json.write_text(
            json.dumps(round_floats(payload), indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        print(f"\nSaved JSON report to {args.output_json}")


if __name__ == "__main__":
    main()
