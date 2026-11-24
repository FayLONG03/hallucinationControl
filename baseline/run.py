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


async def generate_wu_chaining(
    client: BaseClient,
    dataset: PopQADataset,
    *,
    num_questions: int | None = None,
    batch_size: int = 10,
) -> list[dict]:
    """Wu et al. (2025) prompt chaining: three sequential calls per question
    (skill 1 answer -> skill 2 confidence -> skill 3 Y/N decision). A question
    that fails at any skill is skipped."""
    examples = dataset.load(num_questions=num_questions)
    total = len(examples)
    print(f"Loaded {total} examples from {dataset.name}")

    records: list[dict] = []
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        print(f"\nProcessing {start + 1}-{end} of {total}...")
        # (dataset_id, example, [skill outputs so far])
        live = [(start + k, ex, []) for k, ex in enumerate(examples[start:end])]

        for skill in (1, 2, 3):
            if skill == 1:
                prompts = [dataset.build_prompt(ex) for _, ex, _ in live]
            elif skill == 2:
                prompts = [dataset.wu_skill2_prompt(ex, outs[0]) for _, ex, outs in live]
            else:
                prompts = [dataset.wu_skill3_prompt(ex, outs[0], outs[1]) for _, ex, outs in live]
            completions = await client.complete_batch(prompts)
            survivors = []
            for (dataset_id, ex, outs), completion in zip(live, completions):
                if not completion.ok:
                    print(f"Question {dataset_id} failed at skill {skill}: {completion.error}")
                    continue
                survivors.append((dataset_id, ex, outs + [completion.content]))
            live = survivors

        for dataset_id, ex, outs in live:
            record = dataset.parse_wu_chain(dataset_id, ex, *outs)
            records.append(record)
            print(
                f"Question {dataset_id} done. Choice: {record['chain_choice']}, "
                f"Answer: {record['chain_answer']}"
            )

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
    generator=generate,
) -> Path:
    """Generate records and write them to a tagged CSV; return the path."""
    records = asyncio.run(
        generator(client, dataset, num_questions=num_questions, batch_size=batch_size)
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


def run_exp3_wang(
    client: BaseClient,
    *,
    num_questions: int | None = None,
    batch_size: int = 10,
    results_folder: Path = Path("baseline/outputs"),
) -> dict:
    """Exp 3: run the Wang et al. (2026) prompt (+1 / -1 / 0 for IDK, numeric
    confidence), then evaluate the model's own IDK policy directly."""
    # The prompt gives no confidence range; "auto" reads values above 1 as percents.
    dataset = PopQADataset(confidence_scale="auto", variant="wang")
    csv_path = generate_csv(
        client,
        dataset,
        tag="wang",
        num_questions=num_questions,
        batch_size=batch_size,
        results_folder=results_folder,
    )
    payload = evaluation.evaluate_wang(csv_path)
    evaluation.print_metrics("Exp 3 - Wang et al. (direct)", payload["wang_policy"])
    _save_metrics(csv_path, payload)
    return {"csv_path": str(csv_path), "metrics": payload}


def _wu_tag(name: str, rewards: tuple[float, float, float]) -> str:
    return f"{name}_" + "_".join(f"{r:+g}" for r in rewards)


def run_exp4_wu_risk_informing(
    client: BaseClient,
    *,
    rewards: tuple[float, float, float] = (1.0, -1.0, 0.4),
    num_questions: int | None = None,
    batch_size: int = 10,
    results_folder: Path = Path("baseline/outputs"),
) -> dict:
    """Exp 4: run the Wu et al. (2025) risk-informing prompt (rewards stated,
    refuse with N, best guess after a refusal), then evaluate the model's own
    refusal policy directly."""
    dataset = PopQADataset(variant="wu_risk_informing", rewards=rewards)
    csv_path = generate_csv(
        client,
        dataset,
        tag=_wu_tag("wu_risk_informing", dataset.rewards),
        num_questions=num_questions,
        batch_size=batch_size,
        results_folder=results_folder,
    )
    rc, ri, ra = dataset.rewards
    payload = evaluation.evaluate_wu_risk_informing(
        csv_path, reward_correct=rc, reward_incorrect=ri, reward_abstain=ra
    )
    evaluation.print_metrics(
        "Exp 4 - Wu et al. risk-informing (direct)", payload["wu_risk_informing_policy"]
    )
    _save_metrics(csv_path, payload)
    return {"csv_path": str(csv_path), "metrics": payload}


def run_exp5_wu_chaining(
    client: BaseClient,
    *,
    rewards: tuple[float, float, float] = (1.0, -1.0, 0.4),
    num_questions: int | None = None,
    batch_size: int = 10,
    results_folder: Path = Path("baseline/outputs"),
) -> dict:
    """Exp 5: run the Wu et al. (2025) prompt chain (answer -> confidence ->
    expected-value decision), then evaluate the chain's refusal policy directly."""
    dataset = PopQADataset(variant="wu_chaining", rewards=rewards)
    csv_path = generate_csv(
        client,
        dataset,
        tag=_wu_tag("wu_chaining", dataset.rewards),
        num_questions=num_questions,
        batch_size=batch_size,
        results_folder=results_folder,
        generator=generate_wu_chaining,
    )
    rc, ri, ra = dataset.rewards
    payload = evaluation.evaluate_wu_chaining(
        csv_path, reward_correct=rc, reward_incorrect=ri, reward_abstain=ra
    )
    evaluation.print_metrics(
        "Exp 5 - Wu et al. prompt chaining (direct)", payload["wu_chaining_policy"]
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
        # GPT-5-mini is a reasoning model; it rejects a custom temperature and
        # does not support logprobs. api_confidence_avg/min will be None — Exp 1
        # IDK-policy metrics are valid, but the token-prob threshold is not.
        "gpt-5-mini": {
            "provider": "openai",
            "model": "openai/gpt-5-mini",
            "temperature": None,
            "logprobs": False,
            "reasoning_effort": "minimal",
        },
        # Gemini 3.1 Flash Lite via OpenRouter (OpenAI-compatible endpoint).
        # Logprobs are not available for Gemini 3.x models. Same caveat as
        # gpt-5-mini: token-prob threshold results are not meaningful.
        "gemini-3.1-flash-lite": {
            "provider": "openai",
            "model": "google/gemini-3.1-flash-lite",
            "temperature": 0.0,
            "logprobs": False,
        },
        # Ministral 8B (instruction-tuned, text-only) run locally on the GPU.
        # Its chat template adds no default system prompt. Logprobs feed the
        # IDK experiment's token-prob threshold (they do not change the text).
        "ministral-8b-instruct": {
            "provider": "hf",
            "model_id": "mistralai/Ministral-8B-Instruct-2410",
            "model": "ministral-8b-instruct",
            "temperature": 0.0,
            "logprobs": True,
        },
        # Kimi K2 (0905, non-reasoning) via OpenRouter, pinned to Novita (fp8,
        # the only host) with no fallback. Novita returns no logprobs, so only
        # the IDK policy and verbal baselines are meaningful.
        "kimi-k2": {
            "provider": "openai",
            "model": "moonshotai/kimi-k2-0905",
            "temperature": 0.0,
            "logprobs": False,
            "batch_size": 500,
            "max_concurrent": 16,
            "max_tokens": 1024,
            "openrouter_provider": {"order": ["novita/fp8"], "allow_fallbacks": False},
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
    parser.add_argument(
        "-e",
        "--experiment",
        choices=[
            "idk", "verbal", "both", "wang",
            "wu_risk_informing", "wu_chaining", "wu", "all",
        ],
        default="both",
        help="Which experiment to run: 'idk' (Exp 1), 'verbal' (Exp 2), 'both' "
        "(Exp 1 + 2), 'wang' (Exp 3: Wang et al.), 'wu_risk_informing' (Exp 4: "
        "Wu et al. risk-informing), 'wu_chaining' (Exp 5: Wu et al. prompt "
        "chaining), 'wu' (Exp 4 + 5), or 'all' (Exp 1-5).",
    )
    parser.add_argument(
        "-rc", "--reward-correct", type=float, default=1.0,
        help="Wu experiments only: reward for a correct answer (default: 1).",
    )
    parser.add_argument(
        "-ri", "--reward-incorrect", type=float, default=-1.0,
        help="Wu experiments only: reward for an incorrect answer (default: -1).",
    )
    parser.add_argument(
        "-ra", "--reward-abstain", type=float, default=0.4,
        help="Wu experiments only: reward for refusing (default: 0.4).",
    )
    parser.add_argument(
        "-n",
        "--num-questions",
        type=int,
        default=NUM_QUESTIONS,
        help="Limit the number of questions (default: full set).",
    )
    parser.add_argument(
        "--gen-batch-size",
        type=int,
        default=None,
        help="Local (hf) models only: prompts per generate() call, overriding the "
        "registry (default: the registry's value; unset there = one at a time).",
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
                logprobs=cfg.get("logprobs", True),
                enable_thinking=cfg.get("enable_thinking"),
                gen_batch_size=args.gen_batch_size or cfg.get("gen_batch_size"),
                attn_implementation=cfg.get("attn_implementation"),
                extra_eos_tokens=cfg.get("extra_eos_tokens", ()),
                max_new_tokens=cfg.get("max_new_tokens", 512),
                chat_template_kwargs=cfg.get("chat_template_kwargs"),
                final_channel_marker=cfg.get("final_channel_marker"),
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
            logprobs=cfg.get("logprobs", True),
            reasoning_effort=cfg.get("reasoning_effort"),
            max_concurrent=cfg.get("max_concurrent", 5),
            provider_routing=cfg.get("openrouter_provider"),
            max_tokens=cfg.get("max_tokens"),
        )

    for cfg in MODELS:
        client = build_client(cfg)
        batch_size = cfg.get("batch_size", BATCH_SIZE)
        if args.gen_batch_size and cfg.get("provider") == "hf":
            # Hand the client whole generate() batches
            batch_size = max(batch_size, args.gen_batch_size)

        if args.experiment in ("idk", "both", "all"):
            print("\n" + "=" * 70)
            print(f"[{cfg['model']}] Experiment 1: IDK & Token Prob")
            print("=" * 70)
            run_exp1_idk_token_prob(
                client,
                confidence_scale=CONFIDENCE_SCALE,
                num_questions=args.num_questions,
                batch_size=batch_size,
                results_folder=RESULTS_FOLDER,
            )

        if args.experiment in ("verbal", "both", "all"):
            print("\n" + "=" * 70)
            print(f"[{cfg['model']}] Experiment 2: Verbal Conf")
            print("=" * 70)
            run_exp2_verbal_conf(
                client,
                confidence_scale=CONFIDENCE_SCALE,
                num_questions=args.num_questions,
                batch_size=batch_size,
                results_folder=RESULTS_FOLDER,
            )

        if args.experiment in ("wang", "all"):
            print("\n" + "=" * 70)
            print(f"[{cfg['model']}] Experiment 3: Wang et al.")
            print("=" * 70)
            run_exp3_wang(
                client,
                num_questions=args.num_questions,
                batch_size=batch_size,
                results_folder=RESULTS_FOLDER,
            )

        wu_rewards = (args.reward_correct, args.reward_incorrect, args.reward_abstain)

        if args.experiment in ("wu_risk_informing", "wu", "all"):
            print("\n" + "=" * 70)
            print(f"[{cfg['model']}] Experiment 4: Wu et al. risk-informing")
            print("=" * 70)
            run_exp4_wu_risk_informing(
                client,
                rewards=wu_rewards,
                num_questions=args.num_questions,
                batch_size=batch_size,
                results_folder=RESULTS_FOLDER,
            )

        if args.experiment in ("wu_chaining", "wu", "all"):
            print("\n" + "=" * 70)
            print(f"[{cfg['model']}] Experiment 5: Wu et al. prompt chaining")
            print("=" * 70)
            run_exp5_wu_chaining(
                client,
                rewards=wu_rewards,
                num_questions=args.num_questions,
                batch_size=batch_size,
                results_folder=RESULTS_FOLDER,
            )
