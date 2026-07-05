"""Local HuggingFace transformers client.

Runs an instruction-tuned model locally (e.g. Llama 3 8B Instruct, Qwen3 4B
Instruct) and returns the same ``Completion`` shape as the OpenAI-style client,
including per-token logprobs over the generated text so the token-probability
confidence experiment works identically to the API path.

The user prompt is wrapped with the model's chat template
(``apply_chat_template``) and decoded greedily by default (temperature=0). Token
logprobs are computed with ``model.compute_transition_scores`` so each generated
token carries its own normalized log-probability, matching the
``{"token", "logprob"}`` entries the ``dataset`` package expects.
"""

from __future__ import annotations

import asyncio

from .base import BaseClient, Completion


class HFClient(BaseClient):
    """Async-compatible client backed by a local transformers model.

    Inference is synchronous under the hood; ``complete_batch`` runs it in a
    worker thread so it composes with the asyncio-based runner without blocking
    the event loop.
    """

    def __init__(
        self,
        model_id: str,
        model: str | None = None,
        *,
        temperature: float | None = 0.0,
        logprobs: bool = True,
        max_new_tokens: int = 512,
        dtype: str = "auto",
        device_map: str = "auto",
        enable_thinking: bool | None = None,
    ) -> None:
        # Imported lazily so importing the package doesn't require torch.
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.model_id = model_id
        # Human-readable name used for logging and output filenames.
        self.model = model or model_id
        self.temperature = temperature
        self.logprobs = logprobs
        self.max_new_tokens = max_new_tokens
        # Some chat templates (Qwen3) accept an `enable_thinking` kwarg; only
        # forward it when explicitly set to avoid breaking other templates.
        self.enable_thinking = enable_thinking

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        resolved_dtype = "auto" if dtype == "auto" else getattr(torch, dtype)
        self.hf_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=resolved_dtype,
            device_map=device_map,
        )
        self.hf_model.eval()

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _apply_template(self, prompt: str):
        messages = [{"role": "user", "content": prompt}]
        kwargs = {"add_generation_prompt": True, "return_tensors": "pt"}
        if self.enable_thinking is not None:
            kwargs["enable_thinking"] = self.enable_thinking
        return self.tokenizer.apply_chat_template(messages, **kwargs)

    def _generate_one(self, prompt: str) -> Completion:
        torch = self._torch
        try:
            input_ids = self._apply_template(prompt).to(self.hf_model.device)
            attention_mask = torch.ones_like(input_ids)
            input_len = input_ids.shape[1]

            gen_kwargs = {
                "max_new_tokens": self.max_new_tokens,
                "return_dict_in_generate": True,
                "output_scores": self.logprobs,
                "pad_token_id": self.tokenizer.pad_token_id,
            }
            if self.temperature and self.temperature > 0:
                gen_kwargs.update(do_sample=True, temperature=self.temperature)
            else:
                gen_kwargs["do_sample"] = False

            with torch.no_grad():
                outputs = self.hf_model.generate(
                    input_ids, attention_mask=attention_mask, **gen_kwargs
                )

            gen_ids = outputs.sequences[0, input_len:]
            content = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

            tokens = None
            if self.logprobs and outputs.scores:
                transition = self.hf_model.compute_transition_scores(
                    outputs.sequences, outputs.scores, normalize_logits=True
                )[0]
                special_ids = set(self.tokenizer.all_special_ids)
                tokens = []
                for tok_id, logprob in zip(gen_ids.tolist(), transition.tolist()):
                    if tok_id in special_ids:
                        continue
                    tokens.append(
                        {
                            "token": self.tokenizer.decode([tok_id]),
                            "logprob": float(logprob),
                        }
                    )

            return Completion(content=content, tokens=tokens, raw={})
        except Exception as exc:  # noqa: BLE001 - surface as a failed completion
            return Completion(error=str(exc))

    async def complete_batch(self, prompts: list[str]) -> list[Completion]:
        # Run the (synchronous) generations sequentially in a worker thread so
        # logprob alignment stays simple and the event loop is not blocked.
        def _run() -> list[Completion]:
            return [self._generate_one(p) for p in prompts]

        return await asyncio.to_thread(_run)
