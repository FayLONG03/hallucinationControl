#!/usr/bin/env python3
"""Baseline experiment runner.

Wires a model client (`model/`) to a dataset (`dataset/`), generates per-example
records, writes them to a CSV, and optionally evaluates them (`eval.py`) with the
coverage-matched abstention metrics.

Pipeline:
    dataset.load()  ->  build prompts  ->  client.complete_batch()
                    ->  dataset.parse()  ->  CSV  ->  eval.evaluate()

Configure the run in the ``__main__`` block at the bottom, or import
``run_experiment`` from another script.
"""

from __future__ import annotations

import asyncio
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Allow running as a script from anywhere: put this folder on the path so the
# `model` and `dataset` packages resolve.
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from dotenv import load_dotenv

from dataset.base import BaseDataset
from dataset.popqa import PopQADataset
from model import OpenAIClient
from model.base import BaseClient

import eval as evaluation

load_dotenv()


async def generate(
    client: BaseClient,
    dataset: BaseDataset,
    *,
    num_questions: int | None = None,
    batch_size: int = 10,
) -> list[dict]:
    """Run the client over the dataset and return parsed record rows."""
    examples = dataset.load(num_questions=num_questions)
    total = len(examples)
    print(f"Loaded {total} examples from {dataset.name}")

    records: list[dict] = []
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        print(f"\nProcessing {start + 1}-{end} of {total}...")
        batch = examples[start:end]
        prompts = [dataset.build_prompt(ex) for ex in batch]
        completions = await client.complete_batch(prompts)

        for offset, (ex, completion) in enumerate(zip(batch, completions)):
            dataset_id = start + offset
            if not completion.ok:
                print(f"Question {dataset_id} failed: {completion.error}")
                continue
            record = dataset.parse(dataset_id, ex, completion)
            if record is None:
                continue
            records.append(record)
            idk = "yes" if record["idk_flag"] else "no"
            answer = record["best_guess"] if record["idk_flag"] else record["first_answer"]
            print(f"Question {dataset_id} done. IDK: {idk}, Answer: {answer}")

    return records


def write_csv(records: list[dict], out_path: Path) -> Path:
    if not records:
        raise ValueError("No records to write.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"\nResults saved to {out_path}")
    return out_path


def generate_csv(
    client: BaseClient,
    dataset: BaseDataset,
    *,
    tag: str,
    num_questions: int | None = None,
    batch_size: int = 10,
    results_folder: Path = Path("baseline/outputs"),
) -> Path:
    """Generate records and write them to a tagged CSV; return the path."""
    records = asyncio.run(
        generate(client, dataset, num_questions=num_questions, batch_size=batch_size)
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = client.model.replace("/", "_")
    return write_csv(
        records,
        Path(results_folder) / f"{dataset.name}-{tag}-{model_slug}-{timestamp}.csv",
    )


def _save_metrics(csv_path: Path, payload: dict) -> None:
    eval_path = csv_path.with_suffix(".eval.json")
    eval_path.write_text(
        json.dumps(evaluation.round_floats(payload), indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved metrics to {eval_path}")


def run_exp1_idk_token_prob(
    client: BaseClient,
    *,
    confidence_scale: float = 1,
    num_questions: int | None = None,
    batch_size: int = 10,
    results_folder: Path = Path("baseline/outputs"),
    token_col: str = "api_confidence_avg",
) -> dict:
    """Exp 1: run the IDK prompt, then evaluate both the IDK policy (direct) and
    the coverage-matched token-probability threshold."""
    dataset = PopQADataset(confidence_scale=confidence_scale, variant="idk")
    csv_path = generate_csv(
        client,
        dataset,
        tag="idk",
        num_questions=num_questions,
        batch_size=batch_size,
        results_folder=results_folder,
    )
    payload = evaluation.evaluate_idk_and_token_prob(csv_path, token_col=token_col)
    evaluation.print_metrics("Exp 1 - IDK policy (direct)", payload["idk_policy"])
    evaluation.print_metrics(
        "Exp 1 - Token-prob threshold (coverage-matched)", payload["token_prob_threshold"]
    )
    _save_metrics(csv_path, payload)
    return {"csv_path": str(csv_path), "metrics": payload}


def run_exp2_verbal_conf(
    client: BaseClient,
    *,
    confidence_scale: float = 1,
    num_questions: int | None = None,
    batch_size: int = 10,
    results_folder: Path = Path("baseline/outputs"),
    conf_col: str = "self_confidence",
) -> dict:
    """Exp 2: run the verbal-confidence prompt, then evaluate the IDK policy
    directly from the file (verbal confidence used only for Brier/ECE)."""
    dataset = PopQADataset(confidence_scale=confidence_scale, variant="verbal")
    csv_path = generate_csv(
        client,
        dataset,
        tag="verbal",
        num_questions=num_questions,
        batch_size=batch_size,
        results_folder=results_folder,
    )
    payload = evaluation.evaluate_verbal_conf(csv_path, conf_col=conf_col)
    evaluation.print_metrics("Exp 2 - Verbal Conf (direct)", payload["idk_policy"])
    _save_metrics(csv_path, payload)
    return {"csv_path": str(csv_path), "metrics": payload}


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Configuration (OpenRouter)
    # ------------------------------------------------------------------
    API_URL = "https://openrouter.ai/api/v1/chat/completions"
    API_KEY_ENV = "OPENROUTER_API_KEY"
    CONFIDENCE_SCALE = 1  # must match the verbal prompt's reported range
    BATCH_SIZE = 10
    NUM_QUESTIONS = None  # int to limit (e.g. 10 for a smoke test)
    RESULTS_FOLDER = Path("baseline/outputs")

    # Models to run, as OpenRouter model IDs. GPT-5 models reject a custom
    # temperature, so we omit it (temperature=None) for those.
    MODELS = [
        {"model": "openai/gpt-4o-mini", "temperature": 0.0},
        {"model": "openai/gpt-5-mini", "temperature": None},
    ]

    api_key = os.getenv(API_KEY_ENV)
    if not api_key:
        print(f"Error: set {API_KEY_ENV} in your environment or .env file.")
        sys.exit(1)

    for cfg in MODELS:
        client = OpenAIClient(
            api_key=api_key,
            model=cfg["model"],
            api_url=API_URL,
            temperature=cfg["temperature"],
        )

        print("\n" + "=" * 70)
        print(f"[{cfg['model']}] Experiment 1: IDK & Token Prob")
        print("=" * 70)
        run_exp1_idk_token_prob(
            client,
            confidence_scale=CONFIDENCE_SCALE,
            num_questions=NUM_QUESTIONS,
            batch_size=BATCH_SIZE,
            results_folder=RESULTS_FOLDER,
        )

        print("\n" + "=" * 70)
        print(f"[{cfg['model']}] Experiment 2: Verbal Conf")
        print("=" * 70)
        run_exp2_verbal_conf(
            client,
            confidence_scale=CONFIDENCE_SCALE,
            num_questions=NUM_QUESTIONS,
            batch_size=BATCH_SIZE,
            results_folder=RESULTS_FOLDER,
        )
