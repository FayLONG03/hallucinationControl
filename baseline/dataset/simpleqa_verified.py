"""SimpleQA-Verified dataset.

Mirrors ``triviaqa.py`` but sources questions/gold answers from
``codelion/SimpleQA-Verified`` (``test`` split, falling back to ``train``). The
prompting, answer/confidence parsing, correctness check, and logprob
aggregation are all shared with PopQA (imported from ``.popqa``); only example
loading, the question field, and the gold-answer extraction differ.

SimpleQA-Verified carries the question in the ``problem`` field and typically a
single string ``answer``; ``_gold_answers`` normalizes both the string and the
occasional list/dict form into the accepted-surface-form list that the shared
``_is_correct`` containment check expects.
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
    """Collect every accepted surface form for a SimpleQA-Verified answer."""
    answers: list[str] = []

    def add(value) -> None:
        if value is None:
            return
        s = str(value).strip()
        if s and s not in answers:
            answers.append(s)

    ans = example.get("answer")
    if isinstance(ans, dict):
        for key in ("value", "normalized_value", "answer"):
            add(ans.get(key))
        for alias in ans.get("aliases") or []:
            add(alias)
        for alias in ans.get("normalized_aliases") or []:
            add(alias)
    elif isinstance(ans, (list, tuple)):
        for value in ans:
            add(value)
    elif ans is not None:
        add(ans)
    return answers


class SimpleQAVerifiedDataset(BaseDataset):
    name = "simpleQA-verified"
    question_field = "problem"

    def load(self, num_questions: int | None = None) -> list[dict[str, Any]]:
        ds = load_dataset("codelion/SimpleQA-Verified")
        # Prefer the `test` split for evaluation consistency.
        if "test" in ds:
            split_name = "test"
        elif "train" in ds:
            split_name = "train"
        else:
            split_name = list(ds.keys())[0]
        ds = ds[split_name]
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

        idk_flag = bool(re.search(r"i don't know", content or "", re.IGNORECASE))
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
            "question": example[self.question_field],
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
