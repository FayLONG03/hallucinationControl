#!/usr/bin/env python3
"""Selective-prediction metrics: AUROC, AURC and AUARC with bootstrap CIs.

All three ask whether a per-row confidence score ranks correct answers above
wrong ones. Inputs are a score per row and ``correct`` (1 right / 0 wrong);
IDK rows count as not correct. Missing scores are set to 0.

- AUROC (higher is better): probability that a random correct row scores above
  a random wrong row; ties count 1/2 (Mann-Whitney U with average ranks).
- AURC (lower is better): sort rows by score, highest first; for each k = 1..N,
  risk(k) = wrong among the top k / k; AURC = mean of risk(k). Rows with tied
  scores are handled by the expected value over a random order within the tie
  (inside a tie group, errors accumulate at the group's error rate), so the
  result does not depend on row order.
- AUARC (higher is better): the same curve with accuracy instead of risk,
  i.e. AUARC = 1 - AURC.

95% CIs are percentile bootstrap intervals (1000 resamples of the rows,
``random.Random(0)``); ``*_ci_pm`` is half the interval width.

Score used per run type (see ``scores_from_rows``):
- ``icalm``:   ``first_confidence`` when answered, ``best_guess_confidence`` when IDK
- ``verbal``:  ``self_confidence``
- ``tokprob``: ``api_confidence_avg``
- ``idk``:     1 if answered, 0 if IDK (the abstention decision itself)

Library use: ``selective_metrics(scores, correct)``.
CLI use:     ``python main_exp/utils/selective_metrics.py <csv> <mode>``.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys

import numpy as np

MODES = ("icalm", "verbal", "tokprob", "idk")
MODE_COLUMN = {"verbal": "self_confidence", "tokprob": "api_confidence_avg"}


def _to_float(value) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def auroc(scores: np.ndarray, correct: np.ndarray) -> float:
    n1 = int(correct.sum())
    n0 = len(correct) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    ranks = starts[inverse] + (counts[inverse] + 1) / 2.0
    return float((ranks[correct == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def aurc(scores: np.ndarray, correct: np.ndarray) -> float:
    n = len(correct)
    if n == 0:
        return float("nan")
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    errors = np.bincount(inverse, weights=1 - correct, minlength=len(counts))
    # Highest score first.
    counts, errors = counts[::-1], errors[::-1]
    rate = errors / counts
    errors_before = np.concatenate(([0.0], np.cumsum(errors)[:-1]))
    group = np.repeat(np.arange(len(counts)), counts)
    group_start = np.concatenate(([0], np.cumsum(counts)[:-1]))
    k = np.arange(1, n + 1)
    step_in_group = k - group_start[group]
    cum_err = errors_before[group] + step_in_group * rate[group]
    return float(np.mean(cum_err / k))


def point_metrics(scores: np.ndarray, correct: np.ndarray) -> tuple[float, float, float]:
    r = aurc(scores, correct)
    return auroc(scores, correct), r, 1 - r


def selective_metrics(
    scores,
    correct,
    *,
    n_bootstrap: int = 1000,
    seed: int = 0,
    confidence_level: float = 0.95,
) -> dict:
    """AUROC / AURC / AUARC with percentile-bootstrap CIs.

    ``scores`` may contain None/NaN; those are set to 0 (ranked as least
    confident), matching how the earlier reported numbers were computed.
    """
    raw = [_to_float(s) for s in scores]
    missing = sum(v is None for v in raw)
    s = np.array([0.0 if v is None else v for v in raw], dtype=float)
    y = np.asarray(correct, dtype=float)
    n = len(y)

    if n and missing == n:
        # No usable scores at all (e.g. a model without token probabilities):
        # report NaN instead of the meaningless all-zero ranking.
        nan = float("nan")
        result = {"auroc": nan, "aurc": nan, "auarc": nan,
                  "selective_n": n, "selective_missing_scores": missing}
        for name in ("auroc", "aurc", "auarc"):
            result[f"{name}_ci_95"] = (nan, nan)
            result[f"{name}_ci_pm"] = nan
        return result

    a, r, ar = point_metrics(s, y)
    result = {
        "auroc": a,
        "aurc": r,
        "auarc": ar,
        "selective_n": n,
        "selective_missing_scores": missing,
    }

    if n_bootstrap and n:
        rng = random.Random(seed)
        draws = []
        for _ in range(n_bootstrap):
            idx = np.fromiter((rng.randrange(n) for _ in range(n)), dtype=np.int64, count=n)
            draws.append(point_metrics(s[idx], y[idx]))
        draws = np.array(draws)
        alpha = 1 - confidence_level
        lo_i = int(alpha / 2 * n_bootstrap)
        hi_i = int((1 - alpha / 2) * n_bootstrap) - 1
        for col, name in enumerate(("auroc", "aurc", "auarc")):
            v = np.sort(draws[:, col])  # NaNs (single-class resamples) sort last
            lo, hi = float(v[lo_i]), float(v[hi_i])
            result[f"{name}_ci_95"] = (lo, hi)
            result[f"{name}_ci_pm"] = (hi - lo) / 2
    return result


def scores_from_rows(rows: list[dict], mode: str) -> list[float | None]:
    """Pick the ranking score for each CSV row (dicts of strings or numbers)."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    out = []
    for row in rows:
        idk = int(float(row.get("idk_flag") or 0))
        if mode == "idk":
            out.append(float(1 - idk))
        elif mode == "icalm":
            col = "best_guess_confidence" if idk and "best_guess_confidence" in row else "first_confidence"
            out.append(_to_float(row.get(col)))
        else:
            out.append(_to_float(row.get(MODE_COLUMN[mode])))
    return out


def selective_metrics_from_df(df, mode: str, **kwargs) -> dict:
    """Convenience wrapper for pandas DataFrames (main_exp eval scripts)."""
    rows = df.to_dict("records")
    result = selective_metrics(
        scores_from_rows(rows, mode), [int(float(r["correct"])) for r in rows], **kwargs
    )
    result["selective_score"] = mode
    return result


def format_line(result: dict) -> str:
    parts = []
    for label, key in (("AUROC ↑", "auroc"), ("AURC ↓", "aurc"), ("AUARC ↑", "auarc")):
        pm = result.get(f"{key}_ci_pm")
        parts.append(f"{label} {result[key]:.3f}" + (f" ± {pm:.3f}" if pm is not None else ""))
    return " | ".join(parts)


def print_report(result: dict) -> None:
    print(f"\nSelective prediction (score: {result.get('selective_score', '?')}, "
          f"n={result['selective_n']}, missing scores set to 0: {result['selective_missing_scores']})")
    print(f"  {format_line(result)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("csv", help="Result CSV with `correct` and `idk_flag` columns")
    parser.add_argument("mode", choices=MODES, help="Which confidence score to rank by")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    csv.field_size_limit(sys.maxsize)
    with open(args.csv, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result = selective_metrics(
        scores_from_rows(rows, args.mode),
        [int(float(r["correct"])) for r in rows],
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    result["selective_score"] = args.mode
    print_report(result)


if __name__ == "__main__":
    main()
