#!/usr/bin/env python3
"""
Recompute remove-norm ablation metrics using the same logic as PopQA eval.py.

This script is meant for post-processing completed ablation CSVs without
changing the active experiment runner while jobs are already in flight.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from datetime import datetime
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
REMOVE_NORM_DIR = SCRIPT_PATH.parent
POPQA_DIR = SCRIPT_PATH.parents[2]
SOURCE_CODE_DIR = SCRIPT_PATH.parents[4]
EVAL_PATH = POPQA_DIR / "eval.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute remove-norm metrics using PopQA eval.py logic."
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help=(
            "Batch root directory, a single drop_norm_x directory, or a single "
            "ablation result CSV file."
        ),
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Confidence level for CI calculations. Default: 0.95.",
    )
    parser.add_argument(
        "--write-per-run",
        action="store_true",
        help="Retained for compatibility. Per-run outputs are always written.",
    )
    return parser.parse_args()


def load_eval_module():
    spec = importlib.util.spec_from_file_location("popqa_eval_module", EVAL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load eval module from {EVAL_PATH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "This script uses the same dependencies as main_exp/popQA/eval.py. "
            "Please run it with the same Python environment that can run eval.py."
        ) from exc
    return pd


def find_result_csvs(target: Path) -> list[Path]:
    if target.is_file():
        return [target.resolve()]

    direct_csvs = sorted(path.resolve() for path in target.glob("popqa_scheme_b_norm_*.csv"))
    if direct_csvs:
        return direct_csvs

    batch_csvs = sorted(
        path.resolve() for path in target.glob("drop_norm_*/popqa_scheme_b_norm_*.csv")
    )
    if batch_csvs:
        return batch_csvs

    raise FileNotFoundError(f"No ablation result CSVs found under {target}")


def parse_variant_name(csv_path: Path) -> str:
    return csv_path.parent.name


def parse_norm_index(variant_name: str) -> int | None:
    if not variant_name.startswith("drop_norm_"):
        return None
    try:
        return int(variant_name.split("_")[-1])
    except ValueError:
        return None


def pretty_label(variant_name: str) -> str:
    return variant_name.replace("_", " ").title()


def try_read_removed_norm_text(run_dir: Path) -> str | None:
    prompt_jsons = sorted(run_dir.glob("prompt_variants_*.json"))
    if not prompt_jsons:
        return None

    payload = json.loads(prompt_jsons[0].read_text(encoding="utf-8"))
    variants = payload.get("variants", [])
    if not variants:
        return None
    return variants[0].get("removed_norm_text")


def format_pm(value: float, margin: float, digits: int = 3) -> str:
    if math.isnan(value) or math.isnan(margin):
        return "nan"
    return f"{value:.{digits}f} ± {margin:.{digits}f}"


def to_builtin(value):
    if isinstance(value, dict):
        return {key: to_builtin(val) for key, val in value.items()}
    if isinstance(value, list):
        return [to_builtin(item) for item in value]
    if isinstance(value, tuple):
        return [to_builtin(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def build_summary_metrics(df, eval_module, confidence: float) -> dict[str, object]:
    acc = float(df["correct"].mean())
    answered = df["idk_flag"] == 0
    answered_n = int(answered.sum())
    total_n = int(len(df))

    far_answered_count = int(df.loc[answered, "false_answer_flag"].sum())
    far_answered = (
        far_answered_count / answered_n if answered_n > 0 else float("nan")
    )
    far_overall_count = int(total_n - df["correct"].sum())
    far_overall = 1 - acc

    far_answered_margin = eval_module.calculate_ci_pm(
        far_answered_count,
        answered_n,
        confidence=confidence,
    )
    far_overall_margin = eval_module.calculate_ci_pm(
        far_overall_count,
        total_n,
        confidence=confidence,
    )

    total_score = float(df["score"].sum())
    wrong_answered = df[answered & (df["correct"] == 0)]
    total_wrong = df[df["correct"] == 0]
    correct_answered = int(df.loc[answered, "correct"].sum())
    total_correct = int(df["correct"].sum())
    abstention_precision = (
        correct_answered / total_correct if total_correct > 0 else 0.0
    )

    return {
        "accuracy": acc,
        "false_answer_rate_answered": {
            "count": far_answered_count,
            "n": answered_n,
            "rate": far_answered,
            "ci_pm": float(far_answered_margin),
            "display": format_pm(far_answered, float(far_answered_margin), digits=3),
        },
        "false_answer_rate_overall": {
            "count": far_overall_count,
            "n": total_n,
            "rate": far_overall,
            "ci_pm": float(far_overall_margin),
            "display": format_pm(far_overall, float(far_overall_margin), digits=3),
        },
        "total_score": total_score,
        "total_examples": total_n,
        "answered_questions": answered_n,
        "wrong_answered_questions": int(len(wrong_answered)),
        "correct_answered_questions": correct_answered,
        "total_correct_questions": total_correct,
        "total_wrong_questions": int(len(total_wrong)),
        "abstention_precision": {
            "rate": float(abstention_precision),
            "display": f"{abstention_precision:.3f}",
        },
    }


def build_brier_metrics(df, eval_module, confidence: float) -> dict[str, object]:
    overall_score, overall_margin = eval_module.calculate_brier_score_with_ci(
        df.copy(),
        confidence=confidence,
    )
    answered_score, answered_margin = eval_module.calculate_brier_score_with_ci(
        df.copy(),
        confidence=confidence,
        answered_only=True,
    )

    return {
        "overall": {
            "score": float(overall_score),
            "ci_pm": float(overall_margin),
            "display": format_pm(float(overall_score), float(overall_margin), digits=4),
        },
        "answered_only": {
            "score": float(answered_score),
            "ci_pm": float(answered_margin),
            "display": format_pm(float(answered_score), float(answered_margin), digits=4),
        },
    }


def build_calibration_metrics(df, eval_module, analysis_label: str) -> dict[str, object]:
    results = eval_module.calculate_calibration_by_category(
        df.copy(),
        analysis_label,
        num_bins=10,
    )
    return to_builtin(results)


def print_cli_summary(label: str, csv_path: Path, summary: dict[str, object]) -> None:
    print(f"\n{'=' * 80}")
    print(f"Evaluating: {label}")
    print(f"File: {csv_path}")
    print(f"{'=' * 80}")
    print(f"\nSummary: {label}")
    print(
        "  False-Answer Rate - Answered: "
        f"{summary['false_answer_rate_answered']['display']}"
    )
    print(
        "  False-Answer Rate - Overall: "
        f"{summary['false_answer_rate_overall']['display']}"
    )
    print(
        f"  Total Score: {summary['total_score']} out of "
        f"{summary['total_examples']}"
    )
    print(
        f"  Answered questions: {summary['answered_questions']} out of "
        f"{summary['total_examples']}"
    )
    print(
        f"  Wrong answered questions: {summary['wrong_answered_questions']} out of "
        f"{summary['answered_questions']}"
    )
    print(
        f"  Total wrong answered questions: "
        f"{summary['total_wrong_questions']}"
    )
    print(
        f"  Correct answered questions: {summary['correct_answered_questions']} out of "
        f"{summary['total_correct_questions']}"
    )
    print(
        "  Abstention precision: "
        f"{summary['abstention_precision']['display']}"
    )


def print_cli_brier(brier: dict[str, object]) -> None:
    print(f"\nBrier Score (overall): {brier['overall']['display']}")
    print(f"Brier Score (answered only): {brier['answered_only']['display']}")


def flatten_record(record: dict[str, object]) -> dict[str, object]:
    summary = record["summary"]
    brier = record["brier"]
    calibration = record["calibration"]

    flat = {
        "variant": record["variant"],
        "removed_norm_index": record["removed_norm_index"],
        "removed_norm_text": record["removed_norm_text"],
        "result_csv": record["result_csv"],
        "generated_at": record["generated_at"],
        "far_answered_display": summary["false_answer_rate_answered"]["display"],
        "far_answered_rate": summary["false_answer_rate_answered"]["rate"],
        "far_answered_ci_pm": summary["false_answer_rate_answered"]["ci_pm"],
        "far_overall_display": summary["false_answer_rate_overall"]["display"],
        "far_overall_rate": summary["false_answer_rate_overall"]["rate"],
        "far_overall_ci_pm": summary["false_answer_rate_overall"]["ci_pm"],
        "total_score": summary["total_score"],
        "answered_questions": summary["answered_questions"],
        "wrong_answered_questions": summary["wrong_answered_questions"],
        "correct_answered_questions": summary["correct_answered_questions"],
        "total_correct_questions": summary["total_correct_questions"],
        "total_wrong_questions": summary["total_wrong_questions"],
        "abstention_precision": summary["abstention_precision"]["rate"],
        "brier_overall_display": brier["overall"]["display"],
        "brier_overall_score": brier["overall"]["score"],
        "brier_overall_ci_pm": brier["overall"]["ci_pm"],
        "brier_answered_display": brier["answered_only"]["display"],
        "brier_answered_score": brier["answered_only"]["score"],
        "brier_answered_ci_pm": brier["answered_only"]["ci_pm"],
        "ece_answered": None,
        "ece_overall": None,
    }

    if calibration.get("answered") is not None:
        flat["ece_answered"] = calibration["answered"]["ECE"]
    if calibration.get("overall") is not None:
        flat["ece_overall"] = calibration["overall"]["ECE"]

    return flat


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(to_builtin(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("No rows to write.")

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_per_run_outputs(run_dir: Path, record: dict[str, object]) -> tuple[Path, Path]:
    json_path = run_dir / "recomputed_metrics_summary.json"
    csv_path = run_dir / "recomputed_metrics_summary.csv"
    write_json(json_path, record)
    write_csv(csv_path, [flatten_record(record)])
    return json_path, csv_path


def main() -> None:
    args = parse_args()
    pd = load_pandas()
    eval_module = load_eval_module()
    target = args.root.resolve()

    csv_paths = find_result_csvs(target)
    csv_paths = sorted(csv_paths, key=lambda path: (parse_norm_index(parse_variant_name(path)) is None, parse_norm_index(parse_variant_name(path)) or 0))

    written_files: list[tuple[Path, Path]] = []

    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        variant = parse_variant_name(csv_path)
        label = pretty_label(variant)
        analysis_label = variant.upper()
        run_dir = csv_path.parent

        summary = build_summary_metrics(df, eval_module, confidence=args.confidence)
        calibration = build_calibration_metrics(df, eval_module, analysis_label)
        brier = build_brier_metrics(df, eval_module, confidence=args.confidence)

        print_cli_summary(label, csv_path, summary)
        eval_module.print_calibration_report(calibration, label)
        print_cli_brier(brier)

        record = {
            "variant": variant,
            "removed_norm_index": parse_norm_index(variant),
            "removed_norm_text": try_read_removed_norm_text(run_dir),
            "result_csv": str(csv_path),
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "confidence": args.confidence,
            "source_eval_script": str(EVAL_PATH),
            "summary": summary,
            "calibration": calibration,
            "brier": brier,
        }

        written_files.append(write_per_run_outputs(run_dir, record))

    for json_path, csv_path in written_files:
        print(f"\nSaved JSON: {json_path}")
        print(f"Saved CSV: {csv_path}")


if __name__ == "__main__":
    main()
