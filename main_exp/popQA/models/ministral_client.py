"""
Ministral-specific implementation for running experiments with Hugging Face models.
Reuses the LLaMA runner (batching, scoring, saving); only model loading
(bfloat16) and batched generation (chat template that already starts with <s>,
left-padded batch trimming) differ.
"""

import os

import torch
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM

from prompts.prompts import SYSTEM_PROMPT
from models.llama_client import LlamaClient


class MinistralClient(LlamaClient):
    """
    Runner for Ministral experiments using Hugging Face models.

    Ministral-8B-Instruct-2410 is a text-only MistralForCausalLM. Its chat
    template has no system role: a system prompt is prepended to the first user
    turn inside [INST], and none is added when the scheme doesn't use one.
    """

    # Model configurations
    MODEL_CONFIGS = {
        "ministral-8b": {
            "full_name": "mistralai/Ministral-8B-Instruct-2410",
            "batch_size": 64,
            "short_name": "ministral-8b"
        }
    }

    TRACKING_DIR = "main_exp/popQA/outputs/ministral_batch_files"

    def __init__(self, experiments, model_name="ministral-8b"):
        """
        Initialize Ministral experiment runner.

        Args:
            experiments: Dictionary of experiment configurations
            model_name: Name of the model to use
        """
        self.experiments = experiments
        self.model_name = model_name

        if model_name not in self.MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_name}")

        model_config = self.MODEL_CONFIGS[model_name]
        self.model_full_name = model_config["full_name"]

        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            login(token=hf_token)

        print(f"🔧 Loading model: {self.model_full_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_full_name, token=hf_token)
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_full_name,
            dtype=torch.bfloat16,
            device_map="auto",
            token=hf_token,
        )
        self.model.eval()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print(f"✅ Model loaded successfully")

    def generate_responses_batch(self, prompts, exp_type=None, temperature=0, max_tokens=500):
        """Generate responses for a batch of prompts in one call."""
        use_system_prompt = self.experiments.get(exp_type, {}).get("use_system_prompt", False)
        messages_list = []
        for prompt in prompts:
            messages = []
            if use_system_prompt and SYSTEM_PROMPT:
                messages.append({"role": "system", "content": SYSTEM_PROMPT})
            messages.append({"role": "user", "content": prompt})
            messages_list.append(messages)
        texts = [
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            for messages in messages_list
        ]

        # The chat template already starts with <s>
        model_inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, add_special_tokens=False
        ).to(self.model.device)

        do_sample = temperature is not None and temperature > 0
        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": do_sample,
        }
        if do_sample:
            gen_kwargs["temperature"] = float(temperature)
        else:
            # Clear any sampling defaults from generation_config for plain greedy
            gen_kwargs.update(top_k=None, top_p=None, repetition_penalty=1.0)

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **model_inputs,
                **gen_kwargs
            )

        # With left padding every row's prompt spans the full padded width
        trimmed = generated_ids[:, model_inputs.input_ids.shape[1]:]
        return self.tokenizer.batch_decode(trimmed, skip_special_tokens=True)
