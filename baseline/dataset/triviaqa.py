"""TriviaQA dataset.

Mirrors ``popqa.py`` but sources questions/gold answers from
``mandarjoshi/trivia_qa`` (``rc.nocontext``, validation split). The prompting,
answer/confidence parsing, correctness check, and logprob aggregation are all
shared with PopQA (imported from ``.popqa``); only example loading and the
gold-answer extraction differ.

Gold answers are collected from the TriviaQA ``answer`` dict (value,
normalized_value, aliases, normalized_aliases) so the shared ``_is_correct``
containment check has every accepted surface form to match against.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from datasets import load_dataset

from model.base import Completion

from .base import BaseDataset
from .popqa import (
    _aggregate_logprobs,
    _confidence_to_01,
    _extract_fields,
    _is_correct,
)


def _gold_answers(example: dict[str, Any]) -> list[str]:
    """Collect every accepted surface form for a TriviaQA answer."""
    answers: list[str] = []

    def add(value) -> None:
        if value is None:
            return
        s = str(value).strip()
        if s and s not in answers:
            answers.append(s)

    ans = example.get("answer")
    if isinstance(ans, dict):
        add(ans.get("value"))
        add(ans.get("normalized_value"))
        for alias in ans.get("aliases") or []:
            add(alias)
        for alias in ans.get("normalized_aliases") or []:
            add(alias)
    elif ans is not None:
        add(ans)
    return answers


class TriviaQADataset(BaseDataset):
    name = "triviaQA"

    def load(self, num_questions: int | None = None) -> list[dict[str, Any]]:
        ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext")["validation"]
        if num_questions is not None and num_questions > 0:
            ds = ds.select(range(num_questions))
        return [dict(ex) for ex in ds]

    def parse(
        self, dataset_id: int, example: dict[str, Any], completion: Completion
    ) -> dict[str, Any] | None:
        if not completion.ok:
            return None

        content = completion.content
        ans, conf, best, best_conf = _extract_fields(content)
        conf_01 = _confidence_to_01(conf, self.confidence_scale)
        best_conf_01 = _confidence_to_01(best_conf, self.confidence_scale)

        idk_flag = bool(ans) and bool(re.search(r"i don't know", ans, re.IGNORECASE))
        gold = _gold_answers(example)

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
            "question_id": example.get("question_id"),
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
