"""Dataset interface.

A *dataset* owns everything task-specific: loading examples, building the
prompt, and parsing a model completion into a scored record row. The record
schema is shared across datasets so that `eval.py` can consume any of them.

Record columns (one dict per example):
    dataset_id, question, possible_answers,
    first_answer, first_confidence, best_guess, best_guess_confidence,
    correct, score, idk_flag, false_answer_flag,
    self_confidence, api_confidence_min, api_confidence_avg
plus any dataset-specific extras.

To add a new dataset, subclass `BaseDataset` and implement the three methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from model.base import Completion


class BaseDataset(ABC):
    """Abstract dataset: loading, prompting, and scoring."""

    #: Short identifier used for output filenames (e.g. "popQA").
    name: str

    @abstractmethod
    def load(self, num_questions: int | None = None) -> list[dict[str, Any]]:
        """Return a list of raw examples (each a dict of fields)."""
        raise NotImplementedError

    @abstractmethod
    def build_prompt(self, example: dict[str, Any]) -> str:
        """Build the model prompt for a single example."""
        raise NotImplementedError

    @abstractmethod
    def parse(
        self, dataset_id: int, example: dict[str, Any], completion: Completion
    ) -> dict[str, Any] | None:
        """Parse a completion into a scored record row.

        Returns None if the completion failed and should be skipped.
        """
        raise NotImplementedError
