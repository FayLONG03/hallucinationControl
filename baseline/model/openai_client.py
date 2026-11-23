"""OpenAI-compatible chat-completions client.

Works with any provider that exposes the ``/v1/chat/completions`` endpoint
(OpenAI, OpenRouter, etc.). Requests are issued concurrently with a bounded
semaphore and retried on rate limits (HTTP 429) with exponential backoff.
"""

from __future__ import annotations

import asyncio
import json

import aiohttp

from .base import BaseClient, Completion


class OpenAIClient(BaseClient):
    """Async client for OpenAI-style ``/v1/chat/completions`` APIs."""

    def __init__(
        self,
        api_key: str,
        model: str = "openai/gpt-4o-mini",
        api_url: str = "https://openrouter.ai/api/v1/chat/completions",
        *,
        temperature: float | None = 0.0,
        logprobs: bool = True,
        reasoning_effort: str | None = None,
        max_concurrent: int = 5,
        max_retries: int = 5,
        empty_retries: int = 3,
        provider_routing: dict | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.api_url = api_url
        # Some models (e.g. the GPT-5 family) reject a custom temperature; pass
        # temperature=None to omit it from the payload and use the default.
        self.temperature = temperature
        self.logprobs = logprobs
        # Reasoning models default to effort "medium" when this is omitted, and
        # the hidden reasoning tokens are billed as output tokens. Set "minimal"
        # (GPT-5 family) to suppress them; None leaves the field out entirely.
        self.reasoning_effort = reasoning_effort
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries
        # Some providers occasionally return a 200 with empty content; re-request
        # those instead of letting an empty answer be scored as a wrong one.
        self.empty_retries = empty_retries
        # OpenRouter provider preferences (e.g. {"order": ["z-ai"],
        # "allow_fallbacks": False}) sent as the payload's "provider" field.
        self.provider_routing = provider_routing
        # Output cap (reasoning tokens included); None leaves the provider default.
        self.max_tokens = max_tokens

    def _build_payload(self, prompt: str) -> dict:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.logprobs:
            payload["logprobs"] = True
        if self.reasoning_effort is not None:
            payload["reasoning"] = {"effort": self.reasoning_effort}
        if self.provider_routing is not None:
            payload["provider"] = self.provider_routing
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        return payload

    async def _post(self, session: aiohttp.ClientSession, payload: dict) -> dict:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        for attempt in range(self.max_retries + 1):
            try:
                async with session.post(self.api_url, json=payload, headers=headers) as response:
                    if response.status == 429 and attempt < self.max_retries:
                        retry_after = response.headers.get("Retry-After")
                        try:
                            wait = int(retry_after) if retry_after else 5 * (2 ** attempt)
                        except ValueError:
                            wait = 5 * (2 ** attempt)
                        print(
                            f"Rate limit (429). Waiting {wait}s before retry "
                            f"{attempt + 1}/{self.max_retries}..."
                        )
                        await asyncio.sleep(wait)
                        continue
                    response.raise_for_status()
                    return json.loads(await response.text())
            except aiohttp.ClientResponseError as exc:
                if exc.status == 429 and attempt < self.max_retries:
                    await asyncio.sleep(5 * (2 ** attempt))
                    continue
                raise
            except Exception as exc:  # noqa: BLE001 - retry any transient error
                if attempt < self.max_retries:
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise
        raise RuntimeError("Exhausted retries without a response")

    @staticmethod
    def _to_completion(raw: dict) -> Completion:
        try:
            choice = raw["choices"][0]
            content = choice["message"]["content"]
            tokens = None
            logprobs_obj = choice.get("logprobs")
            if logprobs_obj and logprobs_obj.get("content"):
                tokens = logprobs_obj["content"]
            return Completion(content=content, tokens=tokens, raw=raw)
        except (KeyError, IndexError, TypeError) as exc:
            return Completion(raw=raw, error=f"malformed response: {exc}")

    async def complete_batch(self, prompts: list[str]) -> list[Completion]:
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def one(session: aiohttp.ClientSession, prompt: str) -> Completion:
            async with semaphore:
                for _ in range(self.empty_retries):
                    try:
                        raw = await self._post(session, self._build_payload(prompt))
                    except Exception as exc:  # noqa: BLE001
                        return Completion(error=str(exc))
                    completion = self._to_completion(raw)
                    # A null content (no error) counts as empty and is retried too
                    if completion.error is not None or (completion.content or "").strip():
                        return completion
                return Completion(raw=raw, error="empty content after retries")

        async with aiohttp.ClientSession() as session:
            return await asyncio.gather(*(one(session, p) for p in prompts))
