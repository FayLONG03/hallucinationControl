"""Shared experiment prompts.

The two experiment variants use the *same* prompt text across all datasets;
only the interpolated question changes. Keeping them here (rather than in any
single dataset) makes that single source of truth explicit.

Variants:
  "idk"    -> IDK / Best Guess, no verbal confidence (Exp 1: IDK & Token Prob).
              Confidence signal comes from token logprobs, not the text.
  "verbal" -> IDK / Best Guess *with* a self-reported confidence in [0, 1]
              (Exp 2: Verbal Conf).
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


PROMPTS = {"idk": prompt_idk, "verbal": prompt_verbal}
