#!/usr/bin/env python3
"""Baseline TriviaQA experiment runner.

Same pipeline as ``run.py`` (PopQA), but pointed at the TriviaQA dataset
(``mandarjoshi/trivia_qa``, ``rc.nocontext`` validation split). Reuses the
shared generation/eval/client-building helpers so only the dataset changes.

    dataset.load()  ->  build prompts  ->  client.complete_batch()
                    ->  dataset.parse()  ->  CSV  ->  eval.evaluate()

Configure the run in the ``__main__`` block, or import ``run_exp1_idk_token_prob``
/ ``run_exp2_verbal_conf`` from another script.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Allow running as a script from anywhere: put this folder on the path so the
# `model` and `dataset` packages resolve.
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from dotenv import load_dotenv

from dataset.triviaqa import TriviaQADataset
from model import HFClient, OpenAIClient
from model.base import BaseClient

import eval as evaluation
from run import _save_metrics, generate_csv

load_dotenv()


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
    dataset = TriviaQADataset(confidence_scale=confidence_scale, variant="idk")
    csv_path = generate_csv(
        client,
        dataset,
        tag="idk",
        num_questions=num_questions,
        batch_size=batch_size,
        results_folder=results_folder,
    )
    reference_csv = evaluation.reference_for_run(
        dataset.name, client.model, matching="token-prob"
    )
    payload = evaluation.evaluate_idk_and_token_prob(
        csv_path, token_col=token_col, reference_csv=reference_csv
    )
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
    verbal-confidence threshold (coverage matched to the reference run when one
    exists, else to this run's own IDK coverage)."""
    dataset = TriviaQADataset(confidence_scale=confidence_scale, variant="verbal")
    csv_path = generate_csv(
        client,
        dataset,
        tag="verbal",
        num_questions=num_questions,
        batch_size=batch_size,
        results_folder=results_folder,
    )
    reference_csv = evaluation.reference_for_run(
        dataset.name, client.model, matching="verbal-conf"
    )
    payload = evaluation.evaluate_verbal_conf(
        csv_path, conf_col=conf_col, reference_csv=reference_csv
    )
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
    # and output filenames. See run.py for the provider semantics.
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
        "gpt-5-mini": {
            "provider": "openai",
            "model": "openai/gpt-5-mini",
            "temperature": None,
            "logprobs": False,
            "reasoning_effort": "minimal",
        },
        "gemini-3.1-flash-lite": {
            "provider": "openai",
            "model": "google/gemini-3.1-flash-lite",
            "temperature": 0.0,
            "logprobs": False,
        },
    }

    parser = argparse.ArgumentParser(description="Baseline TriviaQA experiment runner.")
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
        # OpenAI-compatible provider; falls back to the shared OpenRouter defaults.
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
            logprobs=cfg.get("logprobs", True),
            reasoning_effort=cfg.get("reasoning_effort"),
        )

    for cfg in MODELS:
        client = build_client(cfg)

        print("\n" + "=" * 70)
        print(f"[{cfg['model']}] TriviaQA Experiment 1: IDK & Token Prob")
        print("=" * 70)
        run_exp1_idk_token_prob(
            client,
            confidence_scale=CONFIDENCE_SCALE,
            num_questions=NUM_QUESTIONS,
            batch_size=BATCH_SIZE,
            results_folder=RESULTS_FOLDER,
        )

        print("\n" + "=" * 70)
        print(f"[{cfg['model']}] TriviaQA Experiment 2: Verbal Conf")
        print("=" * 70)
        run_exp2_verbal_conf(
            client,
            confidence_scale=CONFIDENCE_SCALE,
            num_questions=NUM_QUESTIONS,
            batch_size=BATCH_SIZE,
            results_folder=RESULTS_FOLDER,
        )
