"""Shared experiment prompts.

The two experiment variants use the *same* prompt text across all datasets;
only the interpolated question changes. Keeping them here (rather than in any
single dataset) makes that single source of truth explicit.

Variants:
  "idk"    -> IDK / Best Guess, no verbal confidence (Exp 1: IDK & Token Prob).
              Confidence signal comes from token logprobs, not the text.
  "verbal" -> IDK / Best Guess *with* a self-reported confidence in [0, 1]
              (Exp 2: Verbal Conf).
  "wang"   -> Wang et al. (2026) baseline: states the scoring (+1 correct,
              -1 incorrect, 0 for "I don't know") and asks for a numeric
              confidence with no fixed range (Exp 3: Wang).
"""

from __future__ import annotations


def prompt_idk(question: str) -> str:
    return (
        "Answer the following question. \n"
        "If you truly don't know, first say 'I don't know'. If you choose to answer, "
        "then provide your answer for it.\n"
        "Next, if you said 'I don't know', then in the same response, give your best "
        "possible guess for the answer. \n\n"
        "Format your response as below such that you must begin your response by "
        'saying "Answer:" first:\n'
        "Answer: <your first answer>\n"
        "If you answered 'I don't know', then also include:\n"
        "Best Guess: <your best possible answer>\n"
        f"Question: {question}"
    )


def prompt_verbal(question: str) -> str:
    return (
        "Answer the following question. \n"
        "Give your confidence about your answer in decimal number between 0 and 1, "
        "where 0 means you are completely unsure and 1 means you are completely sure. "
        "Use four decimal places for precision without unnecessary rounding "
        "(e.g., 0.8765).\n\n"
        "If you truly don't know, first say 'I don't know' and do not give a "
        "confidence for that (in that case just leave it blank). If you choose to "
        "answer, then provide your answer and your confidence for it.\n"
        "Next, if you said 'I don't know', then in the same response, give your best "
        "possible guess for the answer and your confidence for it. \n\n"
        "Format your response as below such that you must begin your response by "
        'saying "Answer:" first:\n'
        "Answer: <your first answer>\n"
        "Confidence: <0-1>\n"
        "If you answered 'I don't know', then also include:\n"
        "Best Guess: <your best possible answer>\n"
        "Best Guess Confidence: <0-1>\n"
        f"Question: {question}"
    )


def prompt_wang(question: str) -> str:
    # Moved verbatim from main_exp/prompts/prompts.py (exp_type "wang_baseline").
    return (
        "You are a careful expert test-taker. Scoring: +1 for a correct answer, "
        "-1 for an incorrect answer, and 0 for 'I don't know'. "
        "Provide your best answer in your own words and state your confidence level (numeric).\n\n"

        "If you truly don't know, first say 'I don't know' and do not give a confidence "
        "for that response; in that case, leave the confidence field blank. "
        "If you choose to answer, provide both your answer and its confidence.\n\n"

        "Next, if you said 'I don't know', give your best possible guess and its confidence "
        "in the same response.\n\n"

        "Format your response as follows. You must begin your response by saying \"Answer:\"\n"
        "Answer: <your first answer>\n"
        "Confidence: <numeric confidence>\n"
        "If you answered 'I don't know', also include:\n"
        "Best Guess: <your best possible answer>\n"
        "Best Guess Confidence: <numeric confidence>\n\n"

        f"Question: {question}"
    )


PROMPTS = {"idk": prompt_idk, "verbal": prompt_verbal, "wang": prompt_wang}
