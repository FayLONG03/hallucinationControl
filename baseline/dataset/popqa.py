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
from .prompts import (
    prompt_wu_risk_informing,
    prompt_wu_skill2,
    prompt_wu_skill3,
    wu_scoring_criteria,
)


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
    if scale == "auto":
        # No range given in the prompt (Wang): read values above 1 as percents.
        if v > 1:
            v = v / 100.0
    elif scale == 20:
        v = v / 20.0
    elif scale == 100 or scale in ("percent", "pct"):
        v = v / 100.0
    return max(0.0, min(1.0, v))


# --- Wu et al. (2025) parsers, moved from main_exp/popQA/baseline_wu_* -------

_WU_ANSWER_LINE = re.compile(r"(?im)^[\s*#>_-]*answer[\s*_]*:\s*(.+?)\s*$")
_WU_BEST_GUESS = re.compile(r"(?i)best[\s*_]*guess[\s*_]*:\s*([^\n]+)")


def _wu_clean_answer(text: str) -> str:
    """Strip markdown, quotes and a leading '$' left over from the format placeholder."""
    text = text.strip().strip("*_`").strip()
    text = text.lstrip("$").strip()
    return text.strip("\"'“”‘’").strip().rstrip(".").strip()


def _wu_is_refusal(answer: str | None) -> bool:
    return answer is not None and answer.strip().upper() == "N"


def _parse_wu_risk_informing(text: str):
    """Return (answer, best_guess) from a Figure 2 response.

    Takes the last ANSWER and BEST GUESS (reasoning comes first). A BEST GUESS
    written on the ANSWER line is split off. Falls back to the last non-empty
    line that is not a BEST GUESS line when no ANSWER line is present.
    """
    text = text.strip().replace("\r\n", "\n")
    guesses = _WU_BEST_GUESS.findall(text)
    best = _wu_clean_answer(guesses[-1]) if guesses else None

    matches = _WU_ANSWER_LINE.findall(text)
    if matches:
        raw = matches[-1]
    else:
        lines = [l for l in text.splitlines() if l.strip() and not _WU_BEST_GUESS.search(l)]
        raw = lines[-1] if lines else ""
    answer = _wu_clean_answer(_WU_BEST_GUESS.split(raw)[0]) if raw else None
    return answer or None, best or None


def _parse_wu_skill1(out: str):
    """Return (predicted_answer, parsed_ok). Falls back to the last non-empty line."""
    ans, _, _, _ = _extract_fields(out)
    if ans:
        return ans, True
    lines = [l.strip() for l in out.strip().splitlines() if l.strip()]
    return (lines[-1] if lines else ""), False


def _parse_wu_skill2(out: str):
    """Return (confidence in [0, 1] or None, text passed to skill 3)."""
    _, conf, _, _ = _extract_fields(out)
    m = re.search(r"(?i)confidence\s*:\s*\**\s*([0-9]+(?:\.[0-9]+)?)\s*(%?)", out)
    if conf is None and m:
        conf = float(m.group(1))
    if conf is None:
        return None, out.strip()
    # The prompt asks for a percentage; treat a bare decimal <= 1 without "%" (e.g. "0.85") as a fraction.
    is_fraction = m is not None and not m.group(2) and "." in m.group(1) and conf <= 1
    prob = conf if is_fraction else conf / 100.0
    prob = min(max(prob, 0.0), 1.0)
    return prob, f"{prob * 100:g}%"


def _parse_wu_skill3(out: str):
    """Return (choice in {'Y','N'}, parsed_ok). Unparsed choices default to answering (Y)."""
    matches = re.findall(r"(?i)choice\s*:\s*\**\s*\(?([YN])\b", out)
    if matches:
        return matches[-1].upper(), True
    return "Y", False


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

    def __init__(
        self,
        confidence_scale: float = 1,
        variant: str = "verbal",
        rewards: tuple[float, float, float] = (1.0, -1.0, 0.4),
    ) -> None:
        super().__init__(confidence_scale=confidence_scale, variant=variant)
        # (correct, incorrect, abstain); shown in the Wu prompts and used for their score.
        self.rewards = tuple(float(r) for r in rewards)

    def load(self, num_questions: int | None = None) -> list[dict[str, Any]]:
        ds = load_dataset("akariasai/PopQA")["test"]
        if num_questions is not None and num_questions > 0:
            ds = ds.select(range(num_questions))
        return [dict(ex) for ex in ds]

    def build_prompt(self, example: dict[str, Any]) -> str:
        if self.variant == "wu_risk_informing":
            return prompt_wu_risk_informing(example[self.question_field], *self.rewards)
        return super().build_prompt(example)

    # --- Wu et al. prompt chaining (skills 2 and 3) -------------------------

    def wu_skill2_prompt(self, example: dict[str, Any], skill1_out: str) -> str:
        answer, _ = _parse_wu_skill1(skill1_out)
        return prompt_wu_skill2(example[self.question_field], answer)

    def wu_skill3_prompt(
        self, example: dict[str, Any], skill1_out: str, skill2_out: str
    ) -> str:
        answer, _ = _parse_wu_skill1(skill1_out)
        _, conf_text = _parse_wu_skill2(skill2_out)
        return prompt_wu_skill3(
            example[self.question_field], answer, conf_text, wu_scoring_criteria(*self.rewards)
        )

    def parse_wu_chain(
        self,
        dataset_id: int,
        example: dict[str, Any],
        skill1_out: str,
        skill2_out: str,
        skill3_out: str,
    ) -> dict[str, Any]:
        """Map the three chain outputs onto the record schema.

        CHOICE == Y -> answered: first_answer = skill-1 answer, first_confidence = skill-2 conf
        CHOICE == N -> abstained: first_answer = "I don't know",
                       best_guess = skill-1 answer, best_guess_confidence = skill-2 conf
        """
        rc, ri, ra = self.rewards
        answer, ans_ok = _parse_wu_skill1(skill1_out)
        prob, _ = _parse_wu_skill2(skill2_out)
        choice, choice_ok = _parse_wu_skill3(skill3_out)
        idk = choice == "N"
        gold = example["possible_answers"]
        correct = _is_correct(answer, gold)
        return {
            "dataset_id": dataset_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "s_pop": example.get("s_pop"),
            "o_pop": example.get("o_pop"),
            "question": example["question"],
            "possible_answers": str(gold),
            "first_answer": "I don't know" if idk else answer,
            "first_confidence": None if idk else prob,
            "best_guess": answer if idk else None,
            "best_guess_confidence": prob if idk else None,
            "correct": int(correct),
            "score": ra if idk else (rc if correct else ri),
            "idk_flag": int(idk),
            "false_answer_flag": int(not correct),
            "self_confidence": prob,
            # No token probabilities for this baseline.
            "api_confidence_min": None,
            "api_confidence_avg": None,
            # Chain-specific columns
            "chain_answer": answer,
            "chain_confidence": prob,
            "chain_choice": choice,
            "answer_parsed": int(ans_ok),
            "confidence_parsed": int(prob is not None),
            "choice_parsed": int(choice_ok),
        }

    def _parse_wu_risk_informing_record(
        self, dataset_id: int, example: dict[str, Any], content: str
    ) -> dict[str, Any]:
        """Abstention is a refusal by ANSWER: N; correctness is judged on the
        answer when answered and on the best guess when refused."""
        rc, ri, ra = self.rewards
        ans, best = _parse_wu_risk_informing(content)
        idk_flag = _wu_is_refusal(ans)
        gold = example["possible_answers"]
        correct = _is_correct(best if idk_flag else ans, gold)
        return {
            "dataset_id": dataset_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "s_pop": example.get("s_pop"),
            "o_pop": example.get("o_pop"),
            "question": example["question"],
            "possible_answers": str(gold),
            "first_answer": ans,
            "first_confidence": None,
            "best_guess": best,
            "best_guess_confidence": None,
            "correct": int(correct),
            "score": ra if idk_flag else (rc if correct else ri),
            "idk_flag": int(idk_flag),
            "false_answer_flag": int(not correct),
            # The prompt asks for no confidence, and token probabilities are not used.
            "self_confidence": None,
            "api_confidence_min": None,
            "api_confidence_avg": None,
        }

    def parse(
        self, dataset_id: int, example: dict[str, Any], completion: Completion
    ) -> dict[str, Any] | None:
        if not completion.ok:
            return None

        content = completion.content
        if self.variant == "wu_risk_informing":
            return self._parse_wu_risk_informing_record(dataset_id, example, content)
        ans, conf, best, best_conf = _extract_fields(content)
        conf_01 = _confidence_to_01(conf, self.confidence_scale)
        best_conf_01 = _confidence_to_01(best_conf, self.confidence_scale)

        idk_flag = bool(ans) and bool(re.search(r"i don't know", ans, re.IGNORECASE))
        gold = example["possible_answers"]

        if idk_flag:
            correct = _is_correct(best, gold)
            self_conf = best_conf_01
        else:
            correct = _is_correct(ans, gold)
            self_conf = conf_01

        api_avg, api_min = _aggregate_logprobs(completion.tokens, idk_flag)

        if self.variant == "wang":
            # The Wang prompt's own scoring: +1 correct, -1 incorrect, 0 for IDK.
            score = 0 if idk_flag else (1 if correct else -1)
        else:
            score = int(correct)

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
            "score": score,
            "idk_flag": int(idk_flag),
            "false_answer_flag": int(not correct),
            "self_confidence": self_conf,
            "api_confidence_min": api_min,
            "api_confidence_avg": api_avg,
        }
