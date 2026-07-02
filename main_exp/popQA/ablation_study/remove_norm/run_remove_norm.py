#!/usr/bin/env python3
"""
Run PopQA norm ablations for scheme_b_norm by modifying the normative prompt.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv


SCRIPT_PATH = Path(__file__).resolve()
REMOVE_NORM_DIR = SCRIPT_PATH.parent
POPQA_DIR = SCRIPT_PATH.parents[2]
MAIN_EXP_DIR = SCRIPT_PATH.parents[3]
SOURCE_CODE_DIR = SCRIPT_PATH.parents[4]

for path in (str(POPQA_DIR), str(MAIN_EXP_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from prompts import prompts as prompt_module


AVAILABLE_MODELS = {
    "gpt-4o-mini": "gpt",
    "gpt-5-mini": "gpt",
    "qwen-3": "qwen",
    "qwen-3.5": "qwen",
    "llama-3": "llama",
    "gemini-3.1-flash-lite": "gemini",
}

EXPERIMENTS = {
    "scheme_b_norm": {
        "description": "B with normative system prompt",
        "use_system_prompt": True,
    }
}

MODEL_MODULES = {
    "gpt": ("models.gpt_client", "GPTClient"),
    "qwen": ("models.qwen_client", "QwenClient"),
    "llama": ("models.llama_client", "LlamaClient"),
    "gemini": ("models.gemini_client", "GeminiClient"),
}


@dataclass(frozen=True)
class PromptVariant:
    name: str
    removed_norm_index: int | None
    removed_norm_text: str | None
    removed_norm_indexes: tuple[int, ...]
    removed_norm_texts: tuple[str, ...]
    kept_norm_indexes: tuple[int, ...]
    kept_norm_texts: tuple[str, ...]
    system_prompt: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run PopQA scheme_b_norm ablations by removing one system-prompt norm at a time."
        )
    )
    parser.add_argument(
        "--model",
        choices=list(AVAILABLE_MODELS.keys()),
        default="gemini-3.1-flash-lite",
        help="Model to use for the ablation run.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Number of PopQA test examples to run. Default: full dataset.",
    )
    parser.add_argument(
        "--reward-correct",
        type=float,
        default=1.0,
        help="Reward for a correct answer. Default: 1.0.",
    )
    parser.add_argument(
        "--reward-incorrect",
        type=float,
        default=-1.0,
        help="Reward/penalty for an incorrect answer. Default: -1.0.",
    )
    parser.add_argument(
        "--reward-abstain",
        type=float,
        default=0.4,
        help="Reward for answering 'I don't know'. Default: 0.4.",
    )
    parser.add_argument(
        "--drop-index",
        type=int,
        action="append",
        default=None,
        help=(
            "Norm index to remove. Repeat to run multiple specific ablations. "
            "If omitted, the script runs the base prompt plus all one-item removals."
        ),
    )
    parser.add_argument(
        "--keep-index",
        type=int,
        action="append",
        default=None,
        help=(
            "Keep only these norm indexes as a single combined variant. "
            "Repeat the flag to specify multiple kept norms."
        ),
    )
    parser.add_argument(
        "--variant-name",
        type=str,
        default=None,
        help="Optional explicit variant name for a custom kept-norm combination.",
    )
    parser.add_argument(
        "--skip-base",
        action="store_true",
        help="Skip the base prompt and run only removal variants.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the generated prompt variants without calling any model APIs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for dataset selection. Default: 42.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REMOVE_NORM_DIR / "outputs",
        help="Directory for ablation outputs and summaries.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run identifier used in output filenames.",
    )
    return parser.parse_args()


def format_reward(value: float) -> str:
    return f"{value:+g}"


def load_environment() -> None:
    load_dotenv(SOURCE_CODE_DIR / ".env")


def check_required_credentials(model_type: str) -> None:
    required_env = {
        "gpt": "API_KEY",
        "gemini": "OPENROUTER_API_KEY",
    }.get(model_type)

    if required_env and not os.getenv(required_env):
        raise EnvironmentError(
            f"{required_env} is not set. Add it to {SOURCE_CODE_DIR / '.env'} or your environment."
        )


def extract_norms(system_prompt: str) -> tuple[str, list[str]]:
    lines = [line.strip() for line in system_prompt.strip().splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError("SYSTEM_PROMPT does not contain any numbered norm items.")

    header = lines[0]
    norms = []
    for line in lines[1:]:
        match = re.match(r"^\d+\.\s*(.+)$", line)
        if not match:
            raise ValueError(f"Could not parse norm line: {line}")
        norms.append(match.group(1).strip())

    return header, norms


def build_system_prompt(header: str, norms: list[str]) -> str:
    lines = [header]
    for index, norm in enumerate(norms, start=1):
        lines.append(f"{index}. {norm}")
    return "\n".join(lines) + "\n"


def normalize_indexes(indexes: list[int] | None, max_index: int, label: str) -> list[int]:
    if not indexes:
        return []

    invalid = sorted(index for index in indexes if index < 1 or index > max_index)
    if invalid:
        raise ValueError(f"Invalid {label} values: {invalid}. Valid range: 1-{max_index}.")

    normalized: list[int] = []
    seen = set()
    for index in indexes:
        if index not in seen:
            seen.add(index)
            normalized.append(index)
    return sorted(normalized)


def build_variant_from_kept(
    header: str,
    norms: list[str],
    kept_indexes: list[int],
    variant_name: str | None = None,
) -> PromptVariant:
    if not kept_indexes:
        raise ValueError("At least one norm must be kept.")

    kept_set = set(kept_indexes)
    removed_indexes = [index for index in range(1, len(norms) + 1) if index not in kept_set]

    kept_norms = [norms[index - 1] for index in kept_indexes]
    removed_norms = [norms[index - 1] for index in removed_indexes]

    if variant_name:
        name = variant_name
    elif len(removed_indexes) == 0:
        name = "base_all_norms"
    elif len(removed_indexes) == 1:
        name = f"drop_norm_{removed_indexes[0]}"
    else:
        name = "keep_norms_" + "_".join(str(index) for index in kept_indexes)

    removed_norm_index = removed_indexes[0] if len(removed_indexes) == 1 else None
    removed_norm_text = removed_norms[0] if len(removed_norms) == 1 else None

    return PromptVariant(
        name=name,
        removed_norm_index=removed_norm_index,
        removed_norm_text=removed_norm_text,
        removed_norm_indexes=tuple(removed_indexes),
        removed_norm_texts=tuple(removed_norms),
        kept_norm_indexes=tuple(kept_indexes),
        kept_norm_texts=tuple(kept_norms),
        system_prompt=build_system_prompt(header, kept_norms),
    )


def build_prompt_variants(
    base_prompt: str,
    drop_indexes: list[int] | None,
    keep_indexes: list[int] | None,
    include_base: bool,
    variant_name: str | None = None,
) -> list[PromptVariant]:
    header, norms = extract_norms(base_prompt)
    max_index = len(norms)

    normalized_keep_indexes = normalize_indexes(keep_indexes, max_index, "keep-index")
    normalized_drop_indexes = normalize_indexes(drop_indexes, max_index, "drop-index")

    if normalized_keep_indexes and normalized_drop_indexes:
        raise ValueError("Use either --keep-index or --drop-index, not both.")

    if normalized_keep_indexes:
        variants: list[PromptVariant] = []
        if include_base:
            variants.append(
                build_variant_from_kept(
                    header=header,
                    norms=norms,
                    kept_indexes=list(range(1, max_index + 1)),
                )
            )
        variants.append(
            build_variant_from_kept(
                header=header,
                norms=norms,
                kept_indexes=normalized_keep_indexes,
                variant_name=variant_name,
            )
        )
        return variants

    if normalized_drop_indexes:
        target_indexes = normalized_drop_indexes
    else:
        target_indexes = list(range(1, max_index + 1))

    variants: list[PromptVariant] = []
    if include_base:
        variants.append(
            build_variant_from_kept(
                header=header,
                norms=norms,
                kept_indexes=list(range(1, max_index + 1)),
            )
        )

    for index in target_indexes:
        variants.append(
            build_variant_from_kept(
                header=header,
                norms=norms,
                kept_indexes=[position for position in range(1, max_index + 1) if position != index],
            )
        )

    return variants


def load_dataset_split(n_samples: int | None, seed: int):
    import numpy as np
    from datasets import load_dataset

    random.seed(seed)
    np.random.seed(seed)

    print("📚 Loading PopQA dataset...")
    dataset = load_dataset("akariasai/PopQA")["test"]
    if n_samples is not None:
        print(f"⚙️  Running with {n_samples} samples only")
        dataset = dataset.select(range(n_samples))
    print(f"✅ Dataset loaded: {len(dataset)} samples")
    return dataset


def import_model_runner(model_type: str):
    module_name, class_name = MODEL_MODULES[model_type]
    module = importlib.import_module(module_name)
    runner_cls = getattr(module, class_name)
    return module, runner_cls


def patch_system_prompt(prompt_value: str, model_module) -> str:
    original_prompt = prompt_module.SYSTEM_PROMPT
    prompt_module.SYSTEM_PROMPT = prompt_value
    model_module.SYSTEM_PROMPT = prompt_value
    return original_prompt


def restore_system_prompt(original_prompt: str, model_module) -> None:
    prompt_module.SYSTEM_PROMPT = original_prompt
    model_module.SYSTEM_PROMPT = original_prompt


def run_existing_pipeline(
    dataset,
    model_name: str,
    reward_correct: float,
    reward_abstain: float,
    reward_incorrect: float,
    variant: PromptVariant,
) -> pd.DataFrame:
    model_type = AVAILABLE_MODELS[model_name]
    model_module, runner_cls = import_model_runner(model_type)
    original_prompt = patch_system_prompt(variant.system_prompt, model_module)

    try:
        if model_type == "gpt":
            runner = runner_cls(experiments=EXPERIMENTS)
            model_config = runner.MODEL_CONFIGS[model_name]
            result = runner.run_multi_batch_experiment(
                dataset,
                "scheme_b_norm",
                reward_correct,
                reward_abstain,
                reward_incorrect,
                model_name,
                model_config,
            )
        else:
            runner = runner_cls(experiments=EXPERIMENTS, model_name=model_name)
            model_config = runner.MODEL_CONFIGS[model_name]
            result = runner.run_multi_batch_experiment(
                dataset,
                "scheme_b_norm",
                reward_correct,
                reward_abstain,
                reward_incorrect,
                model_config,
            )
    finally:
        restore_system_prompt(original_prompt, model_module)

    if isinstance(result, tuple):
        result_df = result[0]
    else:
        result_df = result

    if result_df is None or len(result_df) == 0:
        raise RuntimeError(f"No results returned for variant '{variant.name}'.")

    return result_df.copy()


def annotate_results(result_df: pd.DataFrame, variant: PromptVariant) -> pd.DataFrame:
    annotated = result_df.copy()
    annotated["ablation_variant"] = variant.name
    annotated["removed_norm_index"] = variant.removed_norm_index
    annotated["removed_norm_text"] = variant.removed_norm_text
    annotated["removed_norm_indexes"] = ",".join(str(index) for index in variant.removed_norm_indexes)
    annotated["removed_norm_texts"] = " | ".join(variant.removed_norm_texts)
    annotated["kept_norm_indexes"] = ",".join(str(index) for index in variant.kept_norm_indexes)
    annotated["kept_norm_texts"] = " | ".join(variant.kept_norm_texts)
    return annotated


def summarize_results(result_df: pd.DataFrame, variant: PromptVariant) -> dict[str, object]:
    answered_mask = result_df["idk_flag"] == 0
    answered_count = int(answered_mask.sum())
    answered_accuracy = (
        float(result_df.loc[answered_mask, "correct"].mean()) if answered_count > 0 else None
    )

    return {
        "ablation_variant": variant.name,
        "removed_norm_index": variant.removed_norm_index,
        "removed_norm_text": variant.removed_norm_text,
        "removed_norm_indexes": list(variant.removed_norm_indexes),
        "removed_norm_texts": list(variant.removed_norm_texts),
        "kept_norm_indexes": list(variant.kept_norm_indexes),
        "kept_norm_texts": list(variant.kept_norm_texts),
        "n_examples": int(len(result_df)),
        "accuracy": float(result_df["correct"].mean()),
        "abstain_rate": float(result_df["idk_flag"].mean()),
        "false_answer_rate": float(result_df["false_answer_flag"].mean()),
        "mean_score": float(result_df["score"].mean()),
        "answered_count": answered_count,
        "answered_accuracy": answered_accuracy,
    }


def save_variant_results(
    result_df: pd.DataFrame,
    output_dir: Path,
    model_name: str,
    variant: PromptVariant,
    reward_correct: float,
    reward_incorrect: float,
    reward_abstain: float,
    run_id: str,
) -> Path:
    model_slug = model_name.replace(".", "_").replace("-", "_")
    reward_slug = (
        f"{format_reward(reward_correct)}_{format_reward(reward_incorrect)}_{format_reward(reward_abstain)}"
        .replace(".", "p")
    )
    file_path = output_dir / (
        f"popqa_scheme_b_norm_{model_slug}_{variant.name}_{reward_slug}_{run_id}.csv"
    )
    result_df.to_csv(file_path, index=False)
    return file_path


def save_variant_metadata(
    variants: list[PromptVariant],
    output_dir: Path,
    model_name: str,
    run_id: str,
) -> Path:
    metadata_path = output_dir / (
        f"prompt_variants_{model_name.replace('.', '_').replace('-', '_')}_{run_id}.json"
    )
    payload = {
        "model": model_name,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "variants": [asdict(variant) for variant in variants],
    }
    metadata_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return metadata_path


def save_summary_json(
    summaries: list[dict[str, object]],
    output_dir: Path,
    model_name: str,
    run_id: str,
    args: argparse.Namespace,
    metadata_path: Path,
) -> Path:
    summary_json_path = output_dir / (
        f"summary_{model_name.replace('.', '_').replace('-', '_')}_{run_id}.json"
    )
    payload = {
        "model": model_name,
        "run_id": run_id,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "scheme": "scheme_b_norm",
        "reward_correct": args.reward_correct,
        "reward_incorrect": args.reward_incorrect,
        "reward_abstain": args.reward_abstain,
        "samples": args.samples,
        "output_dir": str(output_dir),
        "prompt_metadata_file": str(metadata_path),
        "variants": summaries,
    }
    summary_json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary_json_path


def print_variant_preview(variants: list[PromptVariant]) -> None:
    for variant in variants:
        print(f"\n{'=' * 80}")
        print(f"Variant: {variant.name}")
        if variant.removed_norm_indexes:
            print(
                "Removed norms: "
                + ", ".join(str(index) for index in variant.removed_norm_indexes)
            )
        else:
            print("Removed norms: none")
        print(
            "Kept norms: "
            + ", ".join(str(index) for index in variant.kept_norm_indexes)
        )
        print(f"{'-' * 80}")
        print(variant.system_prompt.rstrip())


def main() -> None:
    args = parse_args()

    os.chdir(SOURCE_CODE_DIR)
    load_environment()

    model_type = AVAILABLE_MODELS[args.model]

    base_prompt = prompt_module.SYSTEM_PROMPT
    variants = build_prompt_variants(
        base_prompt=base_prompt,
        drop_indexes=args.drop_index,
        keep_indexes=args.keep_index,
        include_base=not args.skip_base,
        variant_name=args.variant_name,
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    metadata_path = save_variant_metadata(variants, output_dir, args.model, run_id)

    if args.dry_run:
        print_variant_preview(variants)
        print(f"\n📝 Prompt metadata saved to: {metadata_path}")
        return

    check_required_credentials(model_type)

    if args.samples is None and len(variants) > 1:
        print(
            "⚠️  You are about to run multiple full PopQA experiments. "
            "Consider using --samples first for a smaller smoke test."
        )

    dataset = load_dataset_split(args.samples, args.seed)
    import pandas as pd

    summaries: list[dict[str, object]] = []
    saved_files: list[Path] = []

    for position, variant in enumerate(variants, start=1):
        print(f"\n{'#' * 80}")
        print(f"Running variant {position}/{len(variants)}: {variant.name}")
        if not variant.removed_norm_indexes:
            print("Base prompt with all norm items.")
        elif len(variant.removed_norm_indexes) == 1:
            print(
                f"Removed norm {variant.removed_norm_index}: {variant.removed_norm_text}"
            )
        else:
            print(
                "Removed norms: "
                + ", ".join(str(index) for index in variant.removed_norm_indexes)
            )
            print(
                "Kept norms: "
                + ", ".join(str(index) for index in variant.kept_norm_indexes)
            )
        print(f"{'#' * 80}\n")

        result_df = run_existing_pipeline(
            dataset=dataset,
            model_name=args.model,
            reward_correct=args.reward_correct,
            reward_abstain=args.reward_abstain,
            reward_incorrect=args.reward_incorrect,
            variant=variant,
        )

        annotated_df = annotate_results(result_df, variant)
        output_path = save_variant_results(
            result_df=annotated_df,
            output_dir=output_dir,
            model_name=args.model,
            variant=variant,
            reward_correct=args.reward_correct,
            reward_incorrect=args.reward_incorrect,
            reward_abstain=args.reward_abstain,
            run_id=run_id,
        )
        saved_files.append(output_path)

        summary = summarize_results(annotated_df, variant)
        summary["result_file"] = str(output_path)
        summaries.append(summary)

        print(
            "📊 "
            f"accuracy={summary['accuracy']:.4f}, "
            f"abstain_rate={summary['abstain_rate']:.4f}, "
            f"false_answer_rate={summary['false_answer_rate']:.4f}, "
            f"mean_score={summary['mean_score']:.4f}"
        )
        print(f"💾 Saved: {output_path}")

    summary_df = pd.DataFrame(summaries)
    summary_path = output_dir / (
        f"summary_{args.model.replace('.', '_').replace('-', '_')}_{run_id}.csv"
    )
    summary_df.to_csv(summary_path, index=False)
    summary_json_path = save_summary_json(
        summaries=summaries,
        output_dir=output_dir,
        model_name=args.model,
        run_id=run_id,
        args=args,
        metadata_path=metadata_path,
    )

    print(f"\n{'=' * 80}")
    print("Norm ablation run complete")
    print(f"Summary: {summary_path}")
    print(f"Summary JSON: {summary_json_path}")
    print(f"Prompt metadata: {metadata_path}")
    print("Result files:")
    for file_path in saved_files:
        print(f"  - {file_path}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
