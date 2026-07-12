#!/usr/bin/env python3
"""Recompute the `.eval.json` of existing baseline runs without re-querying the model.

The run CSVs hold the model's own outputs; only the coverage target comes from
``reference_data_results``. So after editing a reference file, the metrics can be
rebuilt straight from the logged CSVs.

The dataset, experiment tag and model are read back out of the run filename
(``<dataset>-<tag>-<model_slug>-<timestamp>.csv``), which is how ``run.py`` wrote
them, so the reference lookup resolves exactly as it did during the original run.

Usage:
    python baseline/reeval.py baseline/outputs/popQA-idk-llama-3-8b-instruct-*.csv
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import eval as evaluation

RUN_RE = re.compile(r"^(?P<dataset>.+)-(?P<tag>idk|verbal)-(?P<model>.+)-\d{8}_\d{6}$")


def reeval(csv_path: Path) -> dict:
    match = RUN_RE.match(csv_path.stem)
    if match is None:
        raise ValueError(f"Not a run CSV produced by run.py: {csv_path.name}")
    dataset, tag = match["dataset"], match["tag"]
    model = match["model"].replace("_", "/")

    reference_csv = evaluation.resolve_reference_csv(dataset, model)
    if reference_csv is None:
        print(f"No reference run for ({dataset}, {model}); matching to the run's own IDK coverage.")
    else:
        print(f"Matching coverage to reference run {reference_csv}")

    if tag == "idk":
        payload = evaluation.evaluate_idk_and_token_prob(csv_path, reference_csv=reference_csv)
        evaluation.print_metrics("Exp 1 - IDK policy (direct)", payload["idk_policy"])
        evaluation.print_metrics(
            "Exp 1 - Token-prob threshold (coverage-matched)", payload["token_prob_threshold"]
        )
    else:
        payload = evaluation.evaluate_verbal_conf(csv_path, reference_csv=reference_csv)
        evaluation.print_metrics(
            "Exp 2 - Verbal Conf threshold (coverage-matched)", payload["verbal_conf_threshold"]
        )

    eval_path = csv_path.with_suffix(".eval.json")
    eval_path.write_text(
        json.dumps(evaluation.round_floats(payload), indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved metrics to {eval_path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="+", type=Path, help="run CSV(s) under baseline/outputs")
    args = parser.parse_args()
    for csv_path in args.csv:
        print(f"\n=== {csv_path.name} ===")
        reeval(csv_path)


if __name__ == "__main__":
    main()
