#!/usr/bin/env python3
"""Recompute the derived columns of a TriviaQA reference CSV in place.

The model-produced columns (``first_answer``, ``first_confidence``,
``best_guess``, ``best_guess_confidence``) are trusted and left untouched. The
four derived columns are recomputed from them so they match the current grading
logic used by the model clients (main_exp/TriviaQA/models/*.py):

  * idk_flag           = the *first answer* is "I don't know" (regex, not the
                         whole raw output).
  * correct            = is_correct(best_guess if idk_flag else first_answer,
                         gold aliases).
  * score              = scheme_b reward: idk -> reward_abstain,
                         correct -> reward_correct, else -> reward_incorrect.
  * false_answer_flag  = int((not correct) and (not idk_flag)).

Usage:
    python baseline/recompute_reference_derived.py <csv> [<csv> ...] \
        [--reward-correct 1.0 --reward-incorrect -1.0 --reward-abstain 0.4]

Each input file is backed up to ``<csv>.bak`` before being overwritten.
"""
from __future__ import annotations

import argparse
import ast
import re
import shutil
import string

import numpy as np
import pandas as pd

IDK_RE = re.compile(r"i don't know", re.IGNORECASE)
_PUNCT = str.maketrans("", "", string.punctuation)


def normalize(text: str) -> str:
    return text.lower().translate(_PUNCT).strip()


def parse_gold(answer) -> list[str]:
    """The gold `answer` column stores a stringified list of aliases."""
    try:
        value = ast.literal_eval(answer) if isinstance(answer, str) else answer
    except (ValueError, SyntaxError):
        value = [str(answer)]
    if isinstance(value, list):
        return [str(x) for x in value]
    return [str(value)]


def is_correct(llm_answer, gold_answers: list[str]) -> bool:
    """Substring-containment grader, mirroring the model clients."""
    if not gold_answers:
        return False
    if all(isinstance(g, str) and len(g) == 1 for g in gold_answers):
        try:
            joined = "".join(gold_answers)
            parsed = ast.literal_eval(joined)
            gold_answers = [str(parsed)] if not isinstance(parsed, list) else [str(x) for x in parsed]
        except Exception:
            gold_answers = ["".join(gold_answers)]
    if llm_answer is None:
        return False
    norm_llm = normalize(str(llm_answer))
    for gold in gold_answers:
        norm_gold = normalize(str(gold))
        if norm_gold in norm_llm or norm_llm in norm_gold:
            return True
    return False


def _clean(cell):
    """Treat NaN/blank cells as a missing candidate answer (None)."""
    if cell is None:
        return None
    if isinstance(cell, float) and np.isnan(cell):
        return None
    return cell


def is_idk(first_answer) -> bool:
    cell = _clean(first_answer)
    if cell is None:
        return False
    text = str(cell)
    return bool(text) and bool(IDK_RE.search(text))


def recompute(df: pd.DataFrame, rc: float, ri: float, ra: float) -> pd.DataFrame:
    gold = df["answer"].apply(parse_gold)
    idk = df["first_answer"].apply(is_idk)

    def grade(i: int) -> int:
        cand = df["best_guess"].iloc[i] if idk.iloc[i] else df["first_answer"].iloc[i]
        return int(is_correct(_clean(cand), gold.iloc[i]))

    correct = pd.Series([grade(i) for i in range(len(df))], index=df.index)
    idk_int = idk.astype(int)
    score = np.where(idk_int == 1, ra, np.where(correct == 1, rc, ri))
    false_flag = ((correct == 0) & (idk_int == 0)).astype(int)

    out = df.copy()
    out["correct"] = correct.astype(int)
    out["score"] = score
    out["idk_flag"] = idk_int
    out["false_answer_flag"] = false_flag
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="+", help="Reference CSV file(s) to recompute in place")
    ap.add_argument("--reward-correct", type=float, default=1.0)
    ap.add_argument("--reward-incorrect", type=float, default=-1.0)
    ap.add_argument("--reward-abstain", type=float, default=0.4)
    ap.add_argument("--no-backup", action="store_true", help="Skip writing a .bak copy")
    args = ap.parse_args()

    for path in args.csv:
        df = pd.read_csv(path)
        before_idk = int((df["idk_flag"] == 1).sum()) if "idk_flag" in df else -1
        out = recompute(df, args.reward_correct, args.reward_incorrect, args.reward_abstain)
        if not args.no_backup:
            shutil.copy2(path, f"{path}.bak")
        out.to_csv(path, index=False)
        after_idk = int((out["idk_flag"] == 1).sum())
        print(f"✅ {path}: rows={len(out)}  idk_flag 1: {before_idk} -> {after_idk}  "
              f"correct1={int((out['correct']==1).sum())}  "
              f"false1={int((out['false_answer_flag']==1).sum())}")


if __name__ == "__main__":
    main()
