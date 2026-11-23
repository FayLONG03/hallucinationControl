"""
Kimi K2 implementation for running experiments via OpenRouter.
Builds on the Gemini OpenRouter runner, adding concurrent prefetch of each
batch, retries on empty responses, --resume, and halting when a whole batch
fails.
"""

from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from tqdm import tqdm

from prompts.prompts import get_experiment_prompt
from models.gemini_client import GeminiClient


class KimiClient(GeminiClient):
    """
    Runner for Kimi K2 (non-reasoning instruct) via OpenRouter.
    """

    # Model configurations
    MODEL_CONFIGS = {
        "kimi-k2": {
            "full_name": "moonshotai/kimi-k2-0905",
            "batch_size": 500,
            "short_name": "kimik2"
        }
    }

    TRACKING_DIR = "main_exp/popQA/outputs/kimi_batch_files"

    # Concurrent requests per batch
    MAX_WORKERS = 16

    # K2 does not reason, so replies are a few lines; the cap only bounds cost
    # if a call runs away.
    MAX_TOKENS = 1024

    # Attempts per prompt when the API returns empty content
    EMPTY_RETRIES = 3

    # Novita (fp8) is the only OpenRouter host; no fallback so a request fails
    # rather than silently moving to another host if one is added later.
    EXTRA_PAYLOAD = {"provider": {"order": ["novita/fp8"], "allow_fallbacks": False}}

    def __init__(self, experiments, model_name="kimi-k2", resume_from=None):
        """
        Args:
            resume_from: Optional results CSV from an earlier run; dataset ids already
                in it are carried over instead of re-queried.
        """
        super().__init__(experiments, model_name=model_name)
        self._prefetched = {}
        # Set once a whole batch fails (e.g. 403 key limit); remaining batches are
        # skipped so the run ends quickly and saves what it has for a later resume.
        self._halted = False
        self._resume_df = pd.read_csv(resume_from) if resume_from else None
        if self._resume_df is not None:
            print(f"♻️  Resuming from {resume_from} ({len(self._resume_df)} rows)")

    def generate_response(self, prompt, exp_config, temperature=0, max_tokens=None,
                          retries=3, retry_delay=5):
        """Return the prefetched response for this prompt, else call the API."""
        if prompt in self._prefetched:
            return self._prefetched[prompt]
        return super().generate_response(
            prompt, exp_config, temperature=temperature,
            max_tokens=max_tokens or self.MAX_TOKENS,
            retries=retries, retry_delay=retry_delay
        )

    def run_batch(self, dataset, start_idx, end_idx, exp_type, reward_correct,
                  reward_abstain, reward_incorrect, model_config):
        """
        Fetch all responses in the slice concurrently, then score them with the
        parent's sequential run_batch (which reads from the prefetched cache).
        Ids already present in the resume CSV are carried over without a request.
        """
        if exp_type not in self.experiments:
            raise ValueError(f"Unknown experiment type: {exp_type}")

        prior = None
        done = set()
        if self._resume_df is not None:
            ids = self._resume_df["dataset_id"]
            prior = self._resume_df[(ids >= start_idx) & (ids < end_idx)]
            done = set(prior["dataset_id"])
        todo = [idx for idx in range(start_idx, end_idx) if idx not in done]
        if not todo or self._halted:
            return prior

        exp_config = self.experiments[exp_type]
        prompts = [
            get_experiment_prompt(
                reward_correct=reward_correct,
                reward_abstain=reward_abstain,
                reward_incorrect=reward_incorrect,
                question=dataset[idx]["question"],
                exp_type=exp_type
            )
            for idx in todo
        ]
        # Duplicate questions in the slice share one request
        unique_prompts = list(dict.fromkeys(prompts))

        def fetch(prompt):
            # Providers occasionally return a 200 with empty content; retry those
            for attempt in range(1, self.EMPTY_RETRIES + 1):
                out = super(KimiClient, self).generate_response(
                    prompt, exp_config, temperature=0, max_tokens=self.MAX_TOKENS
                )
                if out:
                    break
                print(f"⚠️  Empty response (attempt {attempt}/{self.EMPTY_RETRIES})")
            return prompt, out

        with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as pool:
            for prompt, out in tqdm(pool.map(fetch, unique_prompts), total=len(unique_prompts),
                                    desc=f"Fetching [{start_idx}-{end_idx}]"):
                self._prefetched[prompt] = out

        if len(unique_prompts) >= 5 and not any(self._prefetched.values()):
            self._halted = True
            self._prefetched = {}
            print("❌ Every request in this batch failed (check the API key limit). "
                  "Skipping remaining batches; rerun with --resume <saved csv>.")
            return prior

        # Score contiguous runs of missing ids with the parent's run_batch
        runs = []
        for idx in todo:
            if runs and idx == runs[-1][1]:
                runs[-1][1] = idx + 1
            else:
                runs.append([idx, idx + 1])
        try:
            new = [
                super(KimiClient, self).run_batch(
                    dataset, s, e, exp_type,
                    reward_correct, reward_abstain, reward_incorrect, model_config
                )
                for s, e in runs
            ]
        finally:
            self._prefetched = {}
        return pd.concat([prior, *new], ignore_index=True)
