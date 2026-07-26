# I-CALM

**Incentivizing Confidence-Aware Abstention for LLM Selective Answering**

Code and experiments for I-CALM, a prompt-level intervention that turns an LLM's
answer/abstain choice into an explicit decision problem. The model is told the
payoff of each outcome — a reward for a correct answer, a penalty for an
incorrect one, and a smaller positive reward for saying *"I don't know"* — and is
asked to report a confidence for whatever it answers. A normative system prompt
("constitution") is layered on top. The result is a model that abstains when it
should, without any fine-tuning, extra decoding passes, or access to token
log-probabilities.

We evaluate against two standard selective-answering baselines (token-probability
thresholding and self-reported verbal confidence), both **coverage-matched** to
I-CALM so that abstention rates are held equal and only the error profile differs.

---

## Method

Every experiment is a *scheme*: a prompt template that varies what the model is
told about the reward structure. `scheme_b_norm` is I-CALM; the others are
ablations of it.

| Scheme | Rewards stated in prompt | Confidence asked | Normative system prompt | Role |
|--------|--------------------------|------------------|-------------------------|------|
| `pure_eval` | none | no | no | Unincentivized reference — just answer the question |
| `scheme_a_baseline` | correct, incorrect | yes | no | Rewards without any mention of abstention |
| `scheme_a` | correct, incorrect | yes | no | Abstention allowed but not rewarded |
| `scheme_b` | correct, abstain, incorrect | yes | no | Full reward structure, incl. abstention incentive |
| `scheme_b_norm` | correct, abstain, incorrect | yes | **yes** | **I-CALM** |

Under `scheme_a` / `scheme_b` / `scheme_b_norm` the model may answer *"I don't
know"*, and is then asked for a **best guess** plus a confidence for it in the
same response. That second-round guess is what makes the *delta risk* metric
possible: it measures how much worse the model would have done had it been forced
to answer the questions it chose to skip.

Prompt templates and the normative system prompt live in
[main_exp/prompts/prompts.py](main_exp/prompts/prompts.py).

---

## Installation

```bash
git clone <repository-url>
cd hallucinationControl

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### API keys

Create a `.env` in the repo root. Which keys you need depends on the models you
run; scripts load it via `python-dotenv`, so **do not** `source .env` yourself.

| Variable | Used by |
|----------|---------|
| `API_KEY` | OpenAI GPT clients in `main_exp/*/models/gpt_client*.py` (Batch API) |
| `OPENROUTER_API_KEY` | OpenRouter-routed models (Gemini, baselines in `baseline/`, PopQA ablations) |
| `HF_TOKEN` | Locally hosted Llama via `models/llama_client.py` |
| `API_KEY_PROJ` | `preliminary_exp/` and `main_exp/mmlu-pro` runners (override with `--api-key-env`) |

---

## Main experiments

Each dataset has its own runner under `main_exp/<dataset>/run_experiment.py`,
sharing the same CLI.

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--model` | `-m` | *required* | Model to run (see per-dataset `AVAILABLE_MODELS`) |
| `--scheme` | `-s` | `scheme_b` | One of the five schemes above |
| `--samples` | `-n` | full dataset | Number of questions |
| `--reward-correct` | `-rc` | `0` | Reward for a correct answer |
| `--reward-abstain` | `-ra` | `0` | Reward for *"I don't know"* (scheme B only) |
| `--reward-incorrect` | `-ri` | `0` | Penalty for an incorrect answer |

MMLU-Pro additionally takes `--category/-c` (default `biology`) and
`--api-key-env`.

**Datasets:** PopQA, TriviaQA, SimpleQA, SimpleQA-Verified, MMLU-Pro, GSM8K.
**Models** vary by dataset; PopQA supports the widest set — `gpt-4o-mini`,
`gpt-5-mini`, `qwen-3`, `qwen-3.5`, `llama-3`, `gemini-3.1-flash-lite`.

```bash
# I-CALM on PopQA with the headline reward setting
python main_exp/popQA/run_experiment.py -m gpt-5-mini -s scheme_b_norm -rc 1 -ra 0.4 -ri -1

# Unincentivized reference run
python main_exp/popQA/run_experiment.py -m gpt-5-mini -s pure_eval -n 500

# Other datasets, same interface
python main_exp/TriviaQA/run_experiment.py -m gpt-4o-mini -s scheme_b_norm -rc 1 -ra 0.4 -ri -1
python main_exp/mmlu-pro/run_experiment.py -m gpt-4o-mini -s scheme_b -c biology -rc 1 -ra 0.4 -ri -1
```

Runs are seeded (`SEED = 42`). GPT models go through the OpenAI **Batch API**, so
a run is submitted and polled; results land in
`main_exp/<dataset>/outputs/<model>_results/*.csv` (git-ignored) with one row per
question: the first answer and its confidence, the best guess and its confidence,
`idk_flag`, `correct`, and `score`.

---

## Evaluation

```bash
python main_exp/popQA/eval.py           -f <result.csv>
python main_exp/TriviaQA/eval.py        -f <result.csv>
python main_exp/simpleQA-verified/eval.py -f <result.csv>
python main_exp/mmlu-pro/eval.py        -f <result.csv>
```

Reported per run, with 95% CIs where applicable:

- **Coverage** — fraction of questions answered.
- **FAR (answered)** — false-answer rate among answered questions.
- **FAR (guess)** — error rate of the best guesses on abstained questions.
- **FAR (overall)** — error rate if every abstention were replaced by its guess.
- **Delta risk** — `FAR (guess) − FAR (answered)`; positive means the model
  abstained on genuinely harder questions.
- **Total utility** — realized payoff under the stated reward structure.
- **ECR / CAR** — abstained-wrong over all-wrong, answered-correct over all-correct.
- **Hallucination rate** — surfaced wrong answers over all questions.
- **ECE** (`--num-bins`, default 10) and **Brier score**, answered-only and overall.

If a CSV's derived columns drift from the grading logic (e.g. after a parser fix),
recompute them in place — `--dry-run` first:

```bash
python main_exp/popQA/recompute_derived.py <result.csv> --dry-run
```

---

## Baselines

[baseline/](baseline/) holds the coverage-matched selective-answering baselines,
split into transport (`model/`), task (`dataset/`), and metrics (`eval.py`). Two
experiments per model:

1. **IDK & token probability** — the model answers or says *"I don't know"*, with
   no verbal confidence. Scored both from its own IDK decision and by ranking on
   the geometric mean of answer-token probabilities.
2. **Verbal confidence** — the model self-reports a confidence in [0, 1]; rows are
   ranked by it. No token probabilities involved.

Both thresholded views are coverage-matched to the corresponding I-CALM run in
`baseline/reference_data_results/<dataset>/`, so coverage is held fixed across
methods.

```bash
python baseline/run.py                    # generate + evaluate (needs OPENROUTER_API_KEY)
python baseline/reeval.py <run.csv>       # re-score existing runs, no model calls
python baseline/eval.py --baseline-csv <csv> --score-col self_confidence --reference-csv <ref.csv>
```

See [baseline/README.md](baseline/README.md) for the record schema and how to add
a model or dataset.

---

## Ablations

Under [main_exp/popQA/ablation_study/](main_exp/popQA/ablation_study/), each
isolating one component of I-CALM:

- `remove_reward/run_no_reward.py` — drop the stated reward structure.
- `remove_confidence/run_no_confidence.py` — drop the confidence request.
- `remove_norm/run_remove_norm.py` — vary or remove the normative system prompt.

Related analyses: [empirical_far_control.py](main_exp/popQA/empirical_far_control.py)
(threshold selection for a target FAR, with a train/test split) and
[rare_common_facts.py](main_exp/popQA/rare_common_facts.py) (behavior split by
entity popularity).

---

## Preliminary experiment

[preliminary_exp/](preliminary_exp/) validates self-reported confidence against
token-level confidence on PopQA and MMLU-Pro — the motivating result for using
verbalized confidence at all.

```bash
python preliminary_exp/run_PopQA.py        # parameters are edited in __main__
python preliminary_exp/evaluate_PopQA.py    -f <popqa.csv>
python preliminary_exp/evaluate_MMLU-Pro.py -f <mmlu.csv>
```

Reports ECE, Brier, false-answer rate, and Pearson/Spearman/Kendall correlation
between self-reported and API-derived confidence. Requires a model that returns
log-probabilities. Details in [preliminary_exp/README.md](preliminary_exp/README.md).

---

## Figures

- [main_exp/popQA/plot/far_vs_abstention.py](main_exp/popQA/plot/far_vs_abstention.py) — FAR/abstention trade-off across reward settings, sized for a two-column layout.
- [main_exp/popQA/plot/model_comparison.py](main_exp/popQA/plot/model_comparison.py) — schemes across models.
- [main_exp/popQA/plot/plot_abstention.py](main_exp/popQA/plot/plot_abstention.py) — abstention rates by setting.
- [plot/cumulative_false_rate.py](plot/cumulative_false_rate.py) — cumulative false-answer rate vs. uncertainty for I-CALM and both baselines.

---

## Cluster runs

[slurm_script/](slurm_script/) has SLURM submission scripts for long runs
(experiments, baselines, re-evaluation). They `cd` to the submit directory,
activate `venv/`, and write logs to `slurm_script/slurm_logs/`. API keys come from
`.env` via the Python entrypoint, not from the job script.

```bash
sbatch slurm_script/run_triviaqa_gpt4omini_scheme_b_norm.slurm
```

---

## Repository layout

```
├── main_exp/                  # I-CALM reward-scheme experiments
│   ├── prompts/prompts.py     #   scheme prompt templates + normative system prompt
│   ├── utils/                 #   response parsing, abstention detection
│   ├── popQA/                 #   runner, eval, ablations, plots, FAR control
│   ├── TriviaQA/ simpleQA/ simpleQA-verified/ mmlu-pro/ gms8k/
│   │                          #   same runner/eval interface per dataset
│   └── */models/              #   gpt (Batch API), qwen, llama, gemini clients
├── baseline/                  # coverage-matched token-prob & verbal-confidence baselines
│   ├── model/ dataset/        #   transport and task layers
│   ├── eval.py run.py reeval.py
│   ├── reference_data_results/#   I-CALM runs supplying the target coverage
│   └── outputs/               #   run CSVs + .eval.json
├── preliminary_exp/           # validity of self-reported confidence
├── plot/                      # cross-method figures
├── slurm_script/              # cluster job scripts
├── requirements.txt
└── .env                       # API keys (create this)
```
