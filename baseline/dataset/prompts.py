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
  "wu_risk_informing"
           -> Wu et al. (2025) risk-informing prompt (Figure 2) + best guess
              after a refusal (Exp 4). Takes the reward values.
  "wu_chaining"
           -> Wu et al. (2025) prompt chaining (Figure 6 right), three calls
              per question (Exp 5). PROMPTS holds skill 1; skills 2 and 3 are
              built from the earlier outputs (see ``prompt_wu_skill2/3``).
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


# ---------------------------------------------------------------------------
# Wu et al. (2025), "Answer, Refuse, or Guess? Investigating Risk-Aware Decision
# Making in Language Models" (arXiv:2503.01332). Moved verbatim from
# main_exp/popQA/baseline_wu_risk_informing and baseline_wu_chaining. The
# multiple-choice parts are adapted for open-ended PopQA (no "Choices").
# ---------------------------------------------------------------------------

WU_FIGURE2_TEMPLATE = (
    "Answer the following question:\n"
    "\n"
    "Question: {question}\n"
    "\n"
    "The scoring criteria is as follows:\n"
    "If you answer correctly, you get {r_cor} point(s).\n"
    "If you answer incorrectly, you get {r_inc} point(s).\n"
    "If you refuse to answer, you get {r_ref} point(s).\n"
    "\n"
    "Your goal is to score as high as possible based on the above scoring criteria, "
    "and you can either choose to:\n"
    "(1) Answer the question with your answer\n"
    "(2) Refuse to answer the question by outputting a single letter N\n"
    "\n"
    "If you refused to answer, then in the same response, give your best possible guess for the answer.\n"
    "\n"
    "Provide your reasoning process first, then provide your final answer in the following format:\n"
    "ANSWER: $answer\n"
    "If you answered N, then also include:\n"
    "BEST GUESS: <your best possible answer>"
)


def prompt_wu_risk_informing(
    question: str, reward_correct: float, reward_incorrect: float, reward_abstain: float
) -> str:
    """Wu et al. Figure 2 (risk-informing), single call, no confidence."""
    return WU_FIGURE2_TEMPLATE.format(
        question=question,
        r_cor=f"{reward_correct:g}",
        r_inc=f"{reward_incorrect:g}",
        r_ref=f"{reward_abstain:g}",
    )


def prompt_wu_skill1(question: str) -> str:
    """Wu et al. chaining, skill 1: question answering."""
    return (
        "Answer the following question:\n"
        f"Question: {question}\n"
        "Provide your reasoning process first, then provide your final answer in the following format:\n"
        "ANSWER: <your answer>"
    )


def prompt_wu_skill2(question: str, predicted_answer: str) -> str:
    """Wu et al. chaining, skill 2: confidence estimation."""
    return (
        "Here is a question and your predicted answer.\n"
        f"Question: {question}\n"
        f"Your predicted answer: {predicted_answer}\n"
        "Please provide the confidence score of your predicted answer in the following format directly:\n"
        "CONFIDENCE: <a score between 0% and 100%>"
    )


def wu_scoring_criteria(
    reward_correct: float, reward_incorrect: float, reward_abstain: float
) -> str:
    return (
        f"If you answer correctly, you get {reward_correct:g} point(s).\n"
        f"If you answer incorrectly, you get {reward_incorrect:g} point(s).\n"
        f"If you refuse to answer, you get {reward_abstain:g} point(s)."
    )


def prompt_wu_skill3(question: str, predicted_answer: str, confidence: str, criteria: str) -> str:
    """Wu et al. chaining, skill 3: expected-value decision (Y answer / N refuse)."""
    return (
        "Here is a question, your predicted answer, and your confidence of the predicted answer.\n"
        f"Question: {question}\n"
        f"Your predicted answer: {predicted_answer}\n"
        f"Your confidence of the predicted answer: {confidence}\n"
        "The scoring criteria is as follows:\n"
        f"{criteria}\n"
        "Your goal is to score as high as possible based on the above scoring criteria, "
        "and you can either choose to:\n"
        "(1) Answer the question with your predicted answer by outputting a single letter Y\n"
        "(2) Refuse to answer the question by outputting a single letter N\n"
        "Provide your reasoning process first, then provide your final choice in the following format:\n"
        "CHOICE: <letter>"
    )


PROMPTS = {
    "idk": prompt_idk,
    "verbal": prompt_verbal,
    "wang": prompt_wang,
    # Needs the rewards: PopQADataset.build_prompt passes them.
    "wu_risk_informing": prompt_wu_risk_informing,
    # First call of the chain; run.py builds skills 2 and 3.
    "wu_chaining": prompt_wu_skill1,
}
