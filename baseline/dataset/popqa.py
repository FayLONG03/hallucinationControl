"""PopQA dataset.

Ports the prompt, parsing, correctness, and logprob-aggregation logic from
``preliminary_exp/run_PopQA.py`` into the shared dataset interface. The model
is asked to answer, optionally say "I don't know", and self-report a confidence
in [0, 1]; token-level confidences (geometric-mean and exp(min)) are aggregated
over the answer span.
"""

from __future__ import annotations

import ast
import math
import re
import string
from datetime import datetime
from typing import Any

from datasets import load_dataset

from model.base import Completion

from .base import BaseDataset


# ---------------------------------------------------------------------------
# Prompts
#
# Two experiment variants:
#   "idk"    -> IDK / Best Guess, no verbal confidence (Exp 1: IDK & Token Prob).
#              Confidence signal comes from token logprobs, not the text.
#   "verbal" -> IDK / Best Guess *with* a self-reported confidence in [0, 1]
#              (Exp 2: Verbal Conf). Keep CONFIDENCE_SCALE consistent with the
#              reported range below.
# ---------------------------------------------------------------------------

def _prompt_idk(question: str) -> str:
    return (
        "Answer the following question. \n"
        "If you truly don't know, first say 'I don't know'. If you choose to answer, "
        "then provide your answer for it.\n"
        "Next, if you said 'I don't know', then in the same response, give your best "
        "possible guess for the answer. \n\n"
        "Format your response as below such that you must begin your response by "
        'saying "Answer:" first:\n'
        "Answer: <your first answer>\n"
        "If you answered 'I don't know', then also include:\n"
        "Best Guess: <your best possible answer>\n"
        f"Question: {question}"
    )


def _prompt_verbal(question: str) -> str:
    return (
        "Answer the following question. \n"
        "Give your confidence about your answer in decimal number between 0 and 1, "
        "where 0 means you are completely unsure and 1 means you are completely sure. "
        "Use four decimal places for precision without unnecessary rounding "
        "(e.g., 0.8765).\n\n"
        "If you truly don't know, first say 'I don't know' and do not give a "
        "confidence for that (in that case just leave it blank). If you choose to "
        "answer, then provide your answer and your confidence for it.\n"
        "Next, if you said 'I don't know', then in the same response, give your best "
        "possible guess for the answer and your confidence for it. \n\n"
        "Format your response as below such that you must begin your response by "
        'saying "Answer:" first:\n'
        "Answer: <your first answer>\n"
        "Confidence: <0-1>\n"
        "If you answered 'I don't know', then also include:\n"
        "Best Guess: <your best possible answer>\n"
        "Best Guess Confidence: <0-1>\n"
        f"Question: {question}"
    )


PROMPTS = {"idk": _prompt_idk, "verbal": _prompt_verbal}


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _extract_fields(text: str | None):
    """Parse response fields; handles blank confidence and % signs."""
    if not text:
        return None, None, None, None
    text = text.strip().replace("\r\n", "\n")

    ans_match = re.search(r"(?im)^\s*(?:answer|response)\s*:\s*(.+?)\s*$", text)
    conf_match = re.search(r"(?im)^\s*confidence\s*:\s*([0-9]+(?:\.[0-9]+)?)?\s*%?\s*$", text)
    best_match = re.search(r"(?im)^\s*best\s*guess\s*:\s*(.+?)\s*$", text)
    best_conf_match = re.search(
        r"(?im)^\s*best\s*guess\s*confidence\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*%?\s*$", text
    )

    answer = ans_match.group(1).strip() if ans_match else None
    conf = float(conf_match.group(1)) if (conf_match and conf_match.group(1)) else None
    best_guess = best_match.group(1).strip().rstrip(",") if best_match else None
    if best_guess and best_guess.lower() in {"", "none", "n/a"}:
        best_guess = None
    best_conf = float(best_conf_match.group(1)) if best_conf_match else None
    return answer, conf, best_guess, best_conf


def _confidence_to_01(value, scale: float = 1):
    """Map a parsed confidence onto [0, 1] given the self-report scale."""
    if value is None or (isinstance(value, float) and value != value):  # NaN
        return None
    v = float(value)
    if scale == 20:
        v = v / 20.0
    elif scale == 100 or scale in ("percent", "pct"):
        v = v / 100.0
    return max(0.0, min(1.0, v))


def _normalize(text: str) -> str:
    return text.lower().translate(str.maketrans("", "", string.punctuation)).strip()


def _is_correct(llm_answer: str | None, gold_answers) -> bool:
    if all(isinstance(g, str) and len(g) == 1 for g in gold_answers):
        try:
            gold_answers = ast.literal_eval("".join(gold_answers))
        except Exception:
            gold_answers = ["".join(gold_answers)]
    if llm_answer is None:
        return False
    norm_llm = _normalize(llm_answer)
    for gold in gold_answers:
        norm_gold = _normalize(gold)
        if norm_gold in norm_llm or norm_llm in norm_gold:
            return True
    return False


def _answer_token_span(tokens: list[dict], idk_flag: bool):
    """Return (start, end) token indices for the answer span.

    The answer text sits after a start label ("Answer:" or "Best Guess:") and
    ends at whichever comes first: the matching confidence label (present only
    in the verbal-confidence prompt), the next newline (end of the answer line),
    or end of text. This makes span extraction work for both the IDK prompt
    (no Confidence line) and the verbal-confidence prompt.
    """
    if not tokens:
        return None, None
    full_text = "".join(t.get("token", "") for t in tokens)
    if idk_flag:
        start_pattern = r"(?i)best\s*guess\s*:"
        end_patterns = [r"(?i)best\s*guess\s*confidence\s*:", r"\n"]
    else:
        start_pattern = r"(?i)answer\s*:"
        end_patterns = [r"(?i)confidence\s*:", r"\n"]

    start_match = re.search(start_pattern, full_text)
    if not start_match:
        return None, None
    start_char = start_match.end()

    # Earliest end delimiter strictly after the answer starts; else end of text.
    end_char = len(full_text)
    for pattern in end_patterns:
        end_match = re.search(pattern, full_text[start_char:])
        if end_match:
            end_char = min(end_char, start_char + end_match.start())

    start_idx = end_idx = None
    char_pos = 0
    for i, token_info in enumerate(tokens):
        if start_idx is None and char_pos >= start_char:
            start_idx = i
        if end_idx is None and char_pos >= end_char:
            end_idx = i
            break
        char_pos += len(token_info.get("token", ""))
    if end_idx is None:
        end_idx = len(tokens)
    return start_idx, end_idx


def _aggregate_logprobs(tokens: list[dict] | None, idk_flag: bool):
    """Return (api_confidence_avg, api_confidence_min) over the answer span."""
    if not tokens:
        return None, None
    start_idx, end_idx = _answer_token_span(tokens, idk_flag)
    span = tokens[start_idx:end_idx] if (start_idx is not None and end_idx is not None) else tokens
    logprobs = [t.get("logprob") for t in span if t.get("logprob") is not None]
    if not logprobs:
        return None, None
    api_avg = math.exp(sum(logprobs) / len(logprobs))
    api_min = math.exp(min(logprobs))
    return api_avg, api_min


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PopQADataset(BaseDataset):
    name = "popQA"

    def __init__(self, confidence_scale: float = 1, variant: str = "verbal") -> None:
        # 1 = model reports [0,1]; 20 = [0,20] then /20; 100/'percent' = /100.
        self.confidence_scale = confidence_scale
        # "idk"    -> Exp 1 (IDK & Token Prob): no verbal confidence.
        # "verbal" -> Exp 2 (Verbal Conf): self-reported confidence in [0, 1].
        if variant not in PROMPTS:
            raise ValueError(f"variant must be one of {sorted(PROMPTS)}, got {variant!r}")
        self.variant = variant

    def load(self, num_questions: int | None = None) -> list[dict[str, Any]]:
        ds = load_dataset("akariasai/PopQA")["test"]
        if num_questions is not None and num_questions > 0:
            ds = ds.select(range(num_questions))
        return [dict(ex) for ex in ds]

    def build_prompt(self, example: dict[str, Any]) -> str:
        return PROMPTS[self.variant](example["question"])

    def parse(
        self, dataset_id: int, example: dict[str, Any], completion: Completion
    ) -> dict[str, Any] | None:
        if not completion.ok:
            return None

        content = completion.content
        ans, conf, best, best_conf = _extract_fields(content)
        conf_01 = _confidence_to_01(conf, self.confidence_scale)
        best_conf_01 = _confidence_to_01(best_conf, self.confidence_scale)

        idk_flag = bool(re.search(r"i don't know", content or "", re.IGNORECASE))
        gold = example["possible_answers"]

        if idk_flag:
            correct = _is_correct(best, gold)
            self_conf = best_conf_01
        else:
            correct = _is_correct(ans, gold)
            self_conf = conf_01

        api_avg, api_min = _aggregate_logprobs(completion.tokens, idk_flag)

        return {
            "dataset_id": dataset_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "s_pop": example.get("s_pop"),
            "o_pop": example.get("o_pop"),
            "question": example["question"],
            "possible_answers": str(gold),
            "first_answer": ans,
            "first_confidence": conf_01,
            "best_guess": best,
            "best_guess_confidence": best_conf_01,
            "correct": int(correct),
            "score": int(correct),
            "idk_flag": int(idk_flag),
            "false_answer_flag": int(not correct),
            "self_confidence": self_conf,
            "api_confidence_min": api_min,
            "api_confidence_avg": api_avg,
        }
