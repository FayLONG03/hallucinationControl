# baseline

A clean, modular version of the coverage-matched abstention baseline
(originally `preliminary_exp/coverage_matched_baseline.py` +
`preliminary_exp/run_PopQA.py`). The pipeline is split into three concerns:

```
baseline/
├── model/                 # transport layer — how we talk to a model
│   ├── base.py            #   BaseClient + Completion interface
│   └── openai_client.py   #   OpenRouter (OpenAI-compatible) chat-completions client
├── dataset/               # task layer — prompting, parsing, scoring
│   ├── base.py            #   BaseDataset interface + record schema
│   └── popqa.py           #   PopQA (load, prompt, parse, correctness, logprobs)
├── eval.py                # metrics — coverage-matched thresholding + all metrics
├── run.py                 # orchestrator — wires model + dataset -> CSV -> eval
└── outputs/               # generated CSVs and metric JSON (created on first run)
```

## Pipeline

```
dataset.load() -> build prompts -> client.complete_batch()
              -> dataset.parse() -> CSV -> eval
```

- **model** knows nothing about a dataset — it just returns completion text and
  per-token logprobs.
- **dataset** owns the prompt format, response parsing, correctness check, and
  confidence aggregation, producing one record row per example.
- **eval** computes every metric either directly from the file's IDK decisions
  or via coverage-matched thresholding on a confidence column. Each report
  carries an ordered `summary` (see `SUMMARY_FIELDS` / `summary_row` in
  `eval.py`) with these columns in order:

  `C (coverage)`, `threshold`, `FAR (answered)` + 95% CI, `FAR (guess)`,
  `FAR (overall)` + 95% CI, `delta risk`, `total utility`, `# answered`,
  `# incorrect (answered)`, `# incorrect (overall)`, `ECR` (= AER),
  `CAR` (= abstention precision), `hallucination rate` (= surfaced error),
  `ECE (answered)`, `ECE (overall)`, `brier (answered)`, `brier (overall)`.

  `FAR (guess)` = wrong best guesses / all second-round (`idk_flag == 1`) rows;
  `delta risk` = `FAR (guess) - FAR (answered)`.

## Experiments

`run.py` runs two experiments per model, each with its own prompt variant:

**1. IDK & Token Prob** (`variant="idk"`) — prompt asks the model to answer or
say "I don't know" (+ Best Guess), with **no** verbal confidence. Evaluated two
ways over the same run:
  - `idk_policy`: directly from the file, using the model's own IDK as the
    abstention decision.
  - `token_prob_threshold`: rank rows by `api_confidence_avg` (geometric mean of
    answer-span token probabilities) and accept the top-k to match the IDK
    policy's coverage, exactly like `coverage_matched_baseline.py`.

**2. Verbal Conf** (`variant="verbal"`) — prompt additionally asks for a
self-reported confidence in [0, 1]. Metrics are computed directly from the
file's IDK decisions; the verbal confidence is used only for Brier/ECE. **No
token probability is involved.**

Each run writes `popQA-<variant>-<model>-<timestamp>.csv` and a matching
`.eval.json` with the full metric payload.

## Running

Generate + evaluate both experiments. Calls OpenRouter, so set
`OPENROUTER_API_KEY` in your environment or `.env`. By default it runs both
`openai/gpt-4o-mini` and `openai/gpt-5-mini` over PopQA:

```bash
python baseline/run.py
```

Edit the config block at the bottom of `run.py` to change the model list or
question count. Note: the GPT-5 family rejects a custom temperature, so it is
configured with `temperature=None` (omitted).

Re-evaluate an existing CSV only (no model calls), using the generic
coverage-matched CLI:

```bash
python baseline/eval.py \
  --baseline-csv preliminary_exp/example_output/popQA-example.csv \
  --score-col self_confidence \
  --target-answered 30
```

Use `--reference-csv` instead of `--target-answered` to match the answered
coverage (`idk_flag == 0`) of a reference policy CSV. The experiment-level
helpers `evaluate_idk_and_token_prob()` and `evaluate_verbal_conf()` in
`eval.py` are also importable directly.

## Extending

- **New model provider:** subclass `model.base.BaseClient` and implement
  `complete_batch`.
- **New dataset:** subclass `dataset.base.BaseDataset`, implement `load`,
  `build_prompt`, and `parse` (returning the shared record schema), then add it
  to `DATASETS` in `dataset/__init__.py`.
