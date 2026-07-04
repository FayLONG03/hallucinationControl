"""Datasets (task-specific prompting, parsing, and scoring)."""

from .base import BaseDataset
from .popqa import PopQADataset

#: Registry so `run.py` can select a dataset by name.
DATASETS = {
    PopQADataset.name: PopQADataset,
}

__all__ = ["BaseDataset", "PopQADataset", "DATASETS"]
