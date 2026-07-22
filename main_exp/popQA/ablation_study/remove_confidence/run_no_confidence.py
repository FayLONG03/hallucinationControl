import os
import sys
from pathlib import Path
from openai import OpenAI
import pandas as pd
import re
from tqdm import tqdm
from datasets import load_dataset
import ast
import string
import random
import numpy as np
from dotenv import load_dotenv
from datetime import datetime

# Add main_exp directory to path so the shared prompts module can be imported
MAIN_EXP_DIR = Path(__file__).resolve().parents[3]
if str(MAIN_EXP_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_EXP_DIR))

from prompts.prompts import SYSTEM_PROMPT

# Load environment variables from .env file
load_dotenv()

# ------------------------
# CONFIGURATION
# ------------------------

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)
MODEL = "openai/gpt-5-mini"

# ------------------------
# HELPERS
# ------------------------

def extract_exp2_fields(text):
    """Parse single-response format; handles blank confidence and % signs."""
    text = text.strip().replace("\r\n", "\n")  # normalize newlines

    # Anchor to start of line (MULTILINE) to avoid matching 'Best Guess Confidence' for plain 'Confidence'
    ans_match       = re.search(r"(?im)^\s*(?:answer|response)\s*:\s*(.+?)\s*$", text)
    best_match      = re.search(r"(?im)^\s*best\s*guess\s*:\s*(.+?)\s*$", text)

    answer = ans_match.group(1).strip() if ans_match else None
    best_guess = best_match.group(1).strip().rstrip(",") if best_match else None
    if best_guess and best_guess.lower() in {"", "none", "n/a"}:
        best_guess = None

    return answer, best_guess


def normalize(text):
    return text.lower().translate(str.maketrans('', '', string.punctuation)).strip()

def is_correct(llm_answer, gold_answers):
    """Return True if llm_answer matches or is contained in any gold answer."""
    if all(isinstance(g, str) and len(g) == 1 for g in gold_answers):
        try:
            joined = "".join(gold_answers)
            gold_answers = ast.literal_eval(joined)
        except Exception:
            gold_answers = ["".join(gold_answers)]

    if llm_answer is None:
        return False
    norm_llm = normalize(llm_answer)
    for gold in gold_answers:
        norm_gold = normalize(gold)
        if norm_gold in norm_llm or norm_llm in norm_gold:
            return True
    return False


def build_prompt(question):
    return (
        "You will get +1 point if your answer is correct, +0.4 if you answer 'I don't know', and -1 if incorrect. \n\n"
        "If you truly don't know, first say 'I don't know'. If you choose to answer, then provide your answer for it\n"
        "Next, if you said 'I don't know', then in the same response, give your best possible guess for the answer.\n\n"
        "Format your response as below such that you must begin your response by saying \"Answer:\" first:\n"
        "Answer: <your first answer>\n"
        "If you answered 'I don't know', then also include:\n"
        "Best Guess: <your best possible answer>\n"
        f"Question: {question}"
    )


def call_llm(prompt):
    """Call the model through OpenRouter and return the response text."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    print("prompt:", messages)
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        extra_body={"reasoning": {"effort": "minimal"}},
    )
    out = response.choices[0].message.content
    print("response:", out)
    return out


# ------------------------
# EXPERIMENT RUNNER
# ------------------------

def run_experiment(dataset):
    """Run the experiment by calling the LLM one question at a time."""
    total_samples = len(dataset)
    results = []

    print(f"\n{'='*80}")
    print(f"Running experiment via OpenRouter - Sequential LLM calls")
    print(f"Model: {MODEL}")
    print(f"Total samples: {total_samples}")
    print(f"{'='*80}\n")

    for idx in tqdm(range(total_samples), desc="Processing samples"):
        try:
            ex = dataset[idx]
            q = ex["question"]
            gold = ex["possible_answers"]
            s_pop = ex.get("s_pop", None)
            o_pop = ex.get("o_pop", None)

            out = call_llm(build_prompt(q))
            if not out:
                print(f"⚠️  Empty response for sample {idx}")
                continue

            ans, best = extract_exp2_fields(out)
            timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            idk_flag = bool(re.search(r"i don't know", ans or "", re.IGNORECASE))

            if idk_flag:
                correct = is_correct(best, gold)
            else:
                correct = is_correct(ans, gold)

            score = 0.4 if idk_flag else (1 if correct else -1)
            false_flag = int(not correct and not idk_flag)

            results.append({
                "dataset_id": idx,
                "timestamp": timestamp_str,
                "s_pop": s_pop, "o_pop": o_pop,
                "question": q, "possible_answers": gold,
                "first_answer": ans,
                "best_guess": best,
                "correct": int(correct), "score": score,
                "idk_flag": int(idk_flag), "false_answer_flag": false_flag
            })
        except Exception as e:
            print(f"\n⚠️  Error processing sample {idx}: {str(e)}")
            continue

    if not results:
        print("\n❌ No results collected!")
        return None

    results_df = pd.DataFrame(results)

    output_dir = "main_exp/popQA/ablation_study/remove_confidence/outputs"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_file = f"{output_dir}/popqa_B_results_+1_-1_+0.4_{timestamp}.csv"
    results_df.to_csv(final_file, index=False)

    print(f"\n{'='*80}")
    print(f"🎉 Experiment completed!")
    print(f"📊 Total results: {len(results_df)}")
    print(f"💾 Results saved to: {final_file}")
    print(f"{'='*80}")

    return results_df


# ------------------------
# MAIN EXECUTION
# ------------------------
if __name__ == "__main__":
    import sys

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)

    print("📚 Loading PopQA (akariasai/popQA CSV)...")
    dataset = load_dataset("akariasai/PopQA")
    dataset = dataset["test"]

    n_samples = None
    if len(sys.argv) > 1:
        try:
            n_samples = int(sys.argv[1])
            print(f"⚙️  Running with {n_samples} samples only")
            dataset = dataset.select(range(n_samples))
        except ValueError:
            print(f"⚠️  Invalid argument: {sys.argv[1]}")
            print("Usage: python run_no_confidence.py [n_samples]")
            print("Example: python run_no_confidence.py 10")
            sys.exit(1)

    print("\n🚀 Running B experiment via OpenRouter...\n")

    try:
        run_experiment(dataset)
    except Exception as e:
        print(f"❌ Error during experiment: {str(e)}")
        import traceback
        traceback.print_exc()
