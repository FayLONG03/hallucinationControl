"""Model client interface.

A *client* is a thin transport layer: it takes a batch of prompts and returns
raw completions (content text plus optional per-token logprobs). It knows
nothing about a specific dataset, prompt format, or scoring. All prompt
construction and response parsing lives in the `dataset` package instead.

To add a new provider, subclass `BaseClient` and implement `complete_batch`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Completion:
    """A single model completion.

    Attributes:
        content: The assistant message text, or None on failure.
        tokens: Raw logprob entries for the generated content, each a dict with
            at least ``{"token": str, "logprob": float}``. None if the provider
            did not return logprobs.
        raw: The full parsed JSON response from the provider (for debugging).
        error: Error string if the request failed, else None.
    """

    content: str | None = None
    tokens: list[dict[str, Any]] | None = None
    raw: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.content is not None


class BaseClient(ABC):
    """Abstract model client."""

    #: Human-readable identifier of the underlying model, used for logging and
    #: output filenames.
    model: str

    @abstractmethod
    async def complete_batch(self, prompts: list[str]) -> list[Completion]:
        """Complete a batch of prompts, preserving input order in the output."""
        raise NotImplementedError
