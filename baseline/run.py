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

import argparse
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
from model import HFClient, OpenAIClient
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
    """Exp 2: run the verbal-confidence prompt, then evaluate the coverage-matched
    verbal-confidence threshold (coverage matched to the run's own IDK coverage)."""
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
    evaluation.print_metrics(
        "Exp 2 - Verbal Conf threshold (coverage-matched)", payload["verbal_conf_threshold"]
    )
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
    NUM_QUESTIONS = None  # int to limit (e.g. 10 for a smoke test); None = full set
    RESULTS_FOLDER = Path("baseline/outputs")

    # Registry of available models, keyed by the short name used for `--model`
    # and output filenames. Each entry sets ``provider``:
    #   "openai" -> hosted OpenAI-style API (uses API_URL / API_KEY_ENV).
    #              GPT-5 models reject a custom temperature; use temperature=None.
    #   "hf"     -> local transformers model run on the GPU. ``model_id`` is the
    #              HuggingFace repo; ``model`` is the short name for filenames.
    MODEL_REGISTRY = {
        "llama-3-8b-instruct": {
            "provider": "hf",
            "model_id": "meta-llama/Meta-Llama-3-8B-Instruct",
            "model": "llama-3-8b-instruct",
            "temperature": 0.0,
        },
        "qwen3-4b-instruct": {
            "provider": "hf",
            "model_id": "Qwen/Qwen3-4B-Instruct-2507",
            "model": "qwen3-4b-instruct",
            "temperature": 0.0,
        },
        "gpt-4o-mini": {
            "provider": "openai",
            "model": "openai/gpt-4o-mini",
            "temperature": 0.0,
        },
    }

    parser = argparse.ArgumentParser(description="Baseline PopQA experiment runner.")
    parser.add_argument(
        "-m",
        "--model",
        choices=[*MODEL_REGISTRY, "all"],
        default="llama-3-8b-instruct",
        help="Which model to run (or 'all' to run every registered model).",
    )
    args = parser.parse_args()

    if args.model == "all":
        MODELS = list(MODEL_REGISTRY.values())
    else:
        MODELS = [MODEL_REGISTRY[args.model]]

    def build_client(cfg: dict) -> BaseClient:
        provider = cfg.get("provider", "openai")
        if provider == "hf":
            return HFClient(
                model_id=cfg["model_id"],
                model=cfg.get("model"),
                temperature=cfg.get("temperature", 0.0),
                enable_thinking=cfg.get("enable_thinking"),
            )
        # OpenAI-compatible provider. A model may override the endpoint/key env
        # (e.g. Gemini via Google's OpenAI-compatible endpoint); otherwise it
        # falls back to the shared OpenRouter defaults.
        api_key_env = cfg.get("api_key_env", API_KEY_ENV)
        api_key = os.getenv(api_key_env)
        if not api_key:
            print(f"Error: set {api_key_env} in your environment or .env file.")
            sys.exit(1)
        return OpenAIClient(
            api_key=api_key,
            model=cfg["model"],
            api_url=cfg.get("api_url", API_URL),
            temperature=cfg.get("temperature", 0.0),
        )

    for cfg in MODELS:
        client = build_client(cfg)

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
