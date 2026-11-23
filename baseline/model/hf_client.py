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
        gen_batch_size: int | None = None,
        attn_implementation: str | None = None,
        extra_eos_tokens: tuple[str, ...] = (),
        chat_template_kwargs: dict | None = None,
        final_channel_marker: str | None = None,
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
        # Prompts per generate() call. None keeps one prompt at a time (the
        # original behaviour, kept for earlier models' reproducibility); set it
        # to decode a left-padded batch in one pass, which is much faster.
        # Batched logprobs are recorded only for greedy decoding.
        if gen_batch_size is not None and logprobs and temperature:
            raise ValueError("batched logprobs need greedy decoding (temperature=0)")
        self.gen_batch_size = gen_batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        resolved_dtype = "auto" if dtype == "auto" else getattr(torch, dtype)
        load_kwargs = {}
        if attn_implementation is not None:
            load_kwargs["attn_implementation"] = attn_implementation
        self.hf_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=resolved_dtype,
            device_map=device_map,
            **load_kwargs,
        )
        self.hf_model.eval()

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Tokens that end a reply on top of generation_config's eos, for chat
        # templates whose turn-end token is not listed there (Gemma 2).
        self.eos_token_id = None
        if extra_eos_tokens:
            eos = self.hf_model.generation_config.eos_token_id
            eos = list(eos) if isinstance(eos, (list, tuple)) else [eos]
            self.eos_token_id = eos + self.tokenizer.convert_tokens_to_ids(
                list(extra_eos_tokens)
            )

        # Extra apply_chat_template kwargs (gpt-oss: reasoning_effort).
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        # Reasoning models that emit a trace before the reply (gpt-oss's
        # "<|channel|>final<|message|>"): text and token logprobs keep only
        # what follows the marker, so confidence reflects the final answer.
        # Batched path only.
        self.final_marker_ids = None
        if final_channel_marker is not None:
            if gen_batch_size is None:
                raise ValueError("final_channel_marker needs gen_batch_size")
            self.final_marker_ids = self.tokenizer.encode(
                final_channel_marker, add_special_tokens=False
            )

    def _apply_template(self, prompt: str):
        messages = [{"role": "user", "content": prompt}]
        kwargs = {
            "add_generation_prompt": True,
            "return_tensors": "pt",
            "return_dict": True,
        }
        if self.enable_thinking is not None:
            kwargs["enable_thinking"] = self.enable_thinking
        return self.tokenizer.apply_chat_template(messages, **kwargs)

    def _generate_one(self, prompt: str) -> Completion:
        torch = self._torch
        try:
            encoded = self._apply_template(prompt).to(self.hf_model.device)
            input_ids = encoded["input_ids"]
            attention_mask = encoded.get(
                "attention_mask", torch.ones_like(input_ids)
            )
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
                # Plain greedy: Qwen2.5's generation_config sets repetition_penalty
                gen_kwargs.update(do_sample=False, repetition_penalty=1.0)

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

    def _greedy_logprob_recorder(self, store: list):
        """Logits processor appending each step's chosen-token logprob per row.

        Under greedy decoding the chosen token is the argmax, so its normalized
        logprob is the max of the step's log-softmax: the value
        compute_transition_scores(normalize_logits=True) gives, without keeping
        every step's full-vocabulary scores in memory.
        """
        torch = self._torch
        from transformers import LogitsProcessor

        class _Recorder(LogitsProcessor):
            def __call__(self, input_ids, scores):
                store.append(torch.log_softmax(scores.float(), dim=-1).max(dim=-1).values.cpu())
                return scores

        return _Recorder()

    def _generate_batch(self, prompts: list[str]) -> list[Completion]:
        """Greedy-decode a left-padded batch of prompts in one generate() call."""
        torch = self._torch
        try:
            kwargs = {"tokenize": False, "add_generation_prompt": True}
            if self.enable_thinking is not None:
                kwargs["enable_thinking"] = self.enable_thinking
            kwargs.update(self.chat_template_kwargs)
            texts = [
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}], **kwargs
                )
                for p in prompts
            ]
            self.tokenizer.padding_side = "left"
            # The rendered template already carries the BOS / prefix tokens
            encoded = self.tokenizer(
                texts, return_tensors="pt", padding=True, add_special_tokens=False
            ).to(self.hf_model.device)

            gen_kwargs = {
                "max_new_tokens": self.max_new_tokens,
                "pad_token_id": self.tokenizer.pad_token_id,
            }
            if self.eos_token_id is not None:
                gen_kwargs["eos_token_id"] = self.eos_token_id
            if self.temperature and self.temperature > 0:
                gen_kwargs.update(do_sample=True, temperature=self.temperature)
            else:
                # Clear any sampling defaults from generation_config for greedy
                gen_kwargs.update(do_sample=False, top_k=None, top_p=None, repetition_penalty=1.0)

            step_logprobs: list = []
            if self.logprobs:
                from transformers import LogitsProcessorList

                gen_kwargs["logits_processor"] = LogitsProcessorList(
                    [self._greedy_logprob_recorder(step_logprobs)]
                )

            with torch.inference_mode():
                sequences = self.hf_model.generate(**encoded, **gen_kwargs)

            # With left padding every row's prompt spans the full padded width
            trimmed = sequences[:, encoded["input_ids"].shape[1]:]
            # Where each row's reply starts: 0, or just past the final-channel
            # marker (len(row) when it never appears, e.g. a truncated trace).
            starts = [0] * len(prompts)
            if self.final_marker_ids is not None:
                m = self.final_marker_ids
                for row in range(len(prompts)):
                    ids = trimmed[row].tolist()
                    starts[row] = next(
                        (i + len(m) for i in range(len(ids) - len(m) + 1) if ids[i:i + len(m)] == m),
                        len(ids),
                    )
            contents = [
                self.tokenizer.decode(trimmed[row, starts[row]:], skip_special_tokens=True)
                for row in range(len(prompts))
            ]
            if not self.logprobs:
                return [Completion(content=c, tokens=None, raw={}) for c in contents]

            # Step t's logprob belongs to generated token t; a row ends at its
            # first eos, after which generate() only pads.
            per_step = torch.stack(step_logprobs, dim=1)  # (batch, steps)
            eos_ids = self.eos_token_id or self.hf_model.generation_config.eos_token_id
            eos_ids = set(eos_ids if isinstance(eos_ids, (list, tuple)) else [eos_ids])
            special_ids = set(self.tokenizer.all_special_ids)
            completions = []
            for row, content in enumerate(contents):
                tokens = []
                for t, tok_id in enumerate(trimmed[row].tolist()):
                    if t < starts[row]:
                        continue
                    if tok_id in eos_ids:
                        break
                    if tok_id in special_ids:
                        continue
                    tokens.append(
                        {
                            "token": self.tokenizer.decode([tok_id]),
                            "logprob": float(per_step[row, t]),
                        }
                    )
                completions.append(Completion(content=content, tokens=tokens, raw={}))
            return completions
        except Exception as exc:  # noqa: BLE001 - surface as failed completions
            return [Completion(error=str(exc)) for _ in prompts]

    async def complete_batch(self, prompts: list[str]) -> list[Completion]:
        # Run the (synchronous) generations in a worker thread so the event
        # loop is not blocked: one prompt at a time (keeps logprob alignment
        # simple) unless gen_batch_size is set.
        def _run() -> list[Completion]:
            if self.gen_batch_size is None:
                return [self._generate_one(p) for p in prompts]
            results: list[Completion] = []
            for i in range(0, len(prompts), self.gen_batch_size):
                results.extend(self._generate_batch(prompts[i : i + self.gen_batch_size]))
            return results

        return await asyncio.to_thread(_run)
