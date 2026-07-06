"""Datasets (task-specific prompting, parsing, and scoring)."""

from .base import BaseDataset
from .popqa import PopQADataset
from .simpleqa_verified import SimpleQAVerifiedDataset
from .triviaqa import TriviaQADataset

#: Registry so `run.py` can select a dataset by name.
DATASETS = {
    PopQADataset.name: PopQADataset,
    TriviaQADataset.name: TriviaQADataset,
    SimpleQAVerifiedDataset.name: SimpleQAVerifiedDataset,
}

__all__ = [
    "BaseDataset",
    "PopQADataset",
    "TriviaQADataset",
    "SimpleQAVerifiedDataset",
    "DATASETS",
]
