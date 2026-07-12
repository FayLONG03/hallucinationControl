#!/usr/bin/env python3
"""Recompute the derived columns of popQA result CSVs in place.

The model-produced columns (``first_answer``, ``first_confidence``,
``best_guess``, ``best_guess_confidence``) are trusted and left untouched. The
four derived columns are recomputed so they match the grading logic in
``main_exp/popQA/models/*.py``:

  * idk_flag          = the *first answer* is "I don't know" (not the raw output)
  * correct           = is_correct(best_guess if idk_flag else first_answer,
                        possible_answers)
  * score             = idk_flag -> the abstain reward (see resolve_idk_score);
                        correct -> reward_correct; else reward_incorrect

    The abstain reward is scheme-aware:
      B / B_norm / B_control : the 3rd reward in the filename. B_control's
                               gpt4omini names omit it and default to 0
                               (gpt5mini spells it out as `+0`).
      A / A_baseline         : no abstain reward -- an abstention pays
                               reward_incorrect.
  * false_answer_flag = int((not correct) and (not idk_flag))

Rewards and scheme come from the filename via ``eval.parse_filename``, so this
script and eval.py can never disagree.

Files without ``first_answer``/``best_guess`` (the A_baseline single-shot runs)
are skipped: they have no best-guess round, so there is nothing to relabel.

Usage:
    # preview only, writes nothing
    python main_exp/popQA/recompute_derived.py --dry-run <csv> ...

    # rewrite in place
    python main_exp/popQA/recompute_derived.py main_exp/popQA/outputs/popqa_gpt4omini/*.csv

    # rewrite, then run eval.py on each and print its metrics
    python main_exp/popQA/recompute_derived.py --eval main_exp/popQA/outputs/popqa_gpt4omini/*.csv

Each rewritten file is backed up to ``<csv>.bak`` the first time; an existing
backup is never overwritten.
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import os
import re
import shutil
import string
import subprocess
import sys

import numpy as np
import pandas as pd

_EVAL_PY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval.py")
# eval.py builds its plot output path relative to the repo root, so run it there.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_spec = importlib.util.spec_from_file_location("popqa_eval", _EVAL_PY)
_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_eval)
parse_filename = _eval.parse_filename

IDK_RE = re.compile(r"i don't know", re.IGNORECASE)
_PUNCT = str.maketrans("", "", string.punctuation)

REQUIRED = ("first_answer", "best_guess", "possible_answers")


def normalize(text: str) -> str:
    return text.lower().translate(_PUNCT).strip()


def is_correct(llm_answer, gold_answers) -> bool:
    """Mirrors the popQA client grader.

    ``gold_answers`` is popQA's stringified list (e.g. '["politician", ...]');
    iterating it yields single characters, which triggers the decode branch.
    """
    if all(isinstance(g, str) and len(g) == 1 for g in gold_answers):
        try:
            gold_answers = ast.literal_eval("".join(gold_answers))
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
    """NaN/blank cells mean the field failed to parse -> no candidate answer."""
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


def resolve_idk_score(scheme: str, ri: float, ra: float | None) -> float:
    """The reward paid for an abstention (idk_flag == 1).

    * B / B_norm / B_control pay an abstain reward. B_control's gpt4omini
      filenames omit it, but gpt5mini spells it out as `+0`, so it is 0 there.
    * A / A_baseline have no abstain reward: an abstention pays
      reward_incorrect, i.e. abstaining is worth no more than answering wrong.

    Scheme names vary by run: `B_control` (gpt4omini/gpt5mini) vs
    `scheme_b_control` (qwen), so the `scheme_` prefix is stripped first.
    """
    if ra is not None:
        return ra
    name = scheme.lower()
    if name.startswith("scheme_"):
        name = name[len("scheme_"):]
    if name.startswith("b_control"):
        return 0.0
    return ri


def recompute(df: pd.DataFrame, rc: float, ri: float, idk_score: float) -> pd.DataFrame:
    idk = df["first_answer"].apply(is_idk)

    def grade(i: int) -> int:
        cand = df["best_guess"].iloc[i] if idk.iloc[i] else df["first_answer"].iloc[i]
        return int(is_correct(_clean(cand), df["possible_answers"].iloc[i]))

    correct = pd.Series([grade(i) for i in range(len(df))], index=df.index)
    idk_int = idk.astype(int)

    score = np.where(idk_int == 1, idk_score, np.where(correct == 1, rc, ri))
    if np.all(score == score.astype(int)):
        score = score.astype(int)

    out = df.copy()
    out["correct"] = correct.astype(int)
    out["score"] = score
    out["idk_flag"] = idk_int
    out["false_answer_flag"] = ((correct == 0) & (idk_int == 0)).astype(int)
    return out


def run_eval(path: str) -> int:
    """Run eval.py on a freshly rewritten CSV and stream its metrics to stdout.

    matplotlib emits a `findfont` warning per glyph on machines without Arial;
    those are dropped so real errors stay visible.
    """
    proc = subprocess.run(
        [sys.executable, _EVAL_PY, "--result-file", os.path.abspath(path)],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )
    sys.stdout.write(proc.stdout)
    noise = [ln for ln in proc.stderr.splitlines() if "findfont" not in ln]
    if any(ln.strip() for ln in noise):
        sys.stderr.write("\n".join(noise) + "\n")
    if proc.returncode != 0:
        print(f"⚠️  eval.py exited with status {proc.returncode} for {os.path.basename(path)}")
    return proc.returncode


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="+", help="popQA result CSV file(s)")
    ap.add_argument("--dry-run", action="store_true", help="Report changes without writing")
    ap.add_argument("--no-backup", action="store_true", help="Skip writing a .bak copy")
    ap.add_argument(
        "--eval",
        action="store_true",
        help="After rewriting each file, run eval.py on it and print the metrics. "
             "Skipped files (e.g. A_baseline) are never evaluated.",
    )
    args = ap.parse_args()

    for path in args.csv:
        df = pd.read_csv(path)
        name = os.path.basename(path)

        missing = [c for c in REQUIRED if c not in df.columns]
        if missing:
            print(f"⏭️  {name}: skipped (no {'/'.join(missing)})")
            continue

        _model, scheme, rc, ri, ra = parse_filename(path)
        rc_f, ri_f = float(rc), float(ri)
        ra_f = float(ra) if ra is not None else None
        idk_score = resolve_idk_score(scheme, ri_f, ra_f)

        out = recompute(df, rc_f, ri_f, idk_score)

        changed = {
            c: int((df[c].values != out[c].values).sum())
            for c in ("idk_flag", "correct", "score", "false_answer_flag")
        }
        if ra_f is not None:
            source = "from filename"
        elif scheme.lower().startswith("b_control"):
            source = "B_control default"
        else:
            source = "= reward_incorrect"
        print(
            f"{'🔎' if args.dry_run else '✅'} {name}\n"
            f"    scheme={scheme} rewards=({rc_f:+g}, {ri_f:+g})  "
            f"abstain={idk_score:+g} ({source})  rows={len(out)}\n"
            f"    changed: idk={changed['idk_flag']} correct={changed['correct']} "
            f"score={changed['score']} false={changed['false_answer_flag']}"
        )

        if args.dry_run:
            continue
        backup = f"{path}.bak"
        # Never overwrite an existing backup: the recompute is idempotent, so a
        # second run would replace the pristine original with recomputed data.
        if not args.no_backup and not os.path.exists(backup):
            shutil.copy2(path, backup)
        out.to_csv(path, index=False)

        if args.eval:
            run_eval(path)


if __name__ == "__main__":
    main()
