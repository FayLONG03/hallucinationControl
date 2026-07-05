"""Model clients (transport layer) for the baseline experiment."""

from .base import BaseClient, Completion
from .openai_client import OpenAIClient

# HFClient pulls in torch/transformers lazily; import guarded so environments
# without those installed can still use the OpenAI path.
try:
    from .hf_client import HFClient
except ImportError:  # pragma: no cover - optional local-inference dependency
    HFClient = None

__all__ = ["BaseClient", "Completion", "OpenAIClient", "HFClient"]
