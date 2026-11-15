"""Unbiased HumanEval pass@k and example-filtered n@k, never extrapolating k."""

import math
from collections import Counter, defaultdict

import numpy as np


def pass_at_k(n, c, k):
    if not (0 <= c <= n and 1 <= k <= n):
        raise ValueError(f"Require 0 <= correct <= samples and 1 <= k <= samples; got {n}, {c}, {k}")
    if n - c < k:
        return 1.0
    return -math.expm1(sum(math.log1p(-k / i) for i in range(n - c + 1, n + 1))) if c else 0.0


def bootstrap_mean(values, seed=0, draws=2000):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return None
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (draws, len(values)))].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95": np.quantile(means, [0.025, 0.975]).tolist(),
        "problems": len(values),
        "method": "percentile bootstrap over problems; small samples have low power",
    }


def summarize(records, ks=(1, 5), seed=0):
    groups = defaultdict(list)
    for row in records:
        groups[row["problem_id"]].append(row)
    metrics = {}
    for k in ks:
        if groups and all(len(rows) >= k for rows in groups.values()):
            vals = [
                pass_at_k(len(rows), sum(r["result"]["outcome"] == "passed" for r in rows), k)
                for rows in groups.values()
            ]
            metrics[f"pass@{k}"] = bootstrap_mean(vals, seed)
        else:
            metrics[f"pass@{k}"] = {"mean": None, "reason": "insufficient samples for at least one problem"}
    metrics["outcomes"] = dict(Counter(r["result"]["outcome"] for r in records))
    metrics["problems"] = len(groups)
    metrics["candidates"] = len(records)
    metrics["by_difficulty"] = {}
    for difficulty in sorted({r.get("difficulty", "unknown") for r in records}):
        selected = [rows for rows in groups.values() if rows[0].get("difficulty", "unknown") == difficulty]
        metrics["by_difficulty"][difficulty] = {
            f"pass@{k}": float(
                np.mean(
                    [
                        pass_at_k(len(rows), sum(r["result"]["outcome"] == "passed" for r in rows), k)
                        for rows in selected
                    ]
                )
            )
            if selected and all(len(r) >= k for r in selected)
            else None
            for k in ks
        }
    return metrics


def filtered_n_at_k(records, n):
    """Uniform n draws from example passers (fallback to all).

    Hidden outcomes are read only after the pool has been selected.
    """
    groups = defaultdict(list)
    for row in records:
        groups[row["problem_id"]].append(row)
    values = []
    for rows in groups.values():
        pool = [r for r in rows if r.get("example_result", {}).get("outcome") == "passed"] or rows
        values.append(
            pass_at_k(len(pool), sum(r["result"]["outcome"] == "passed" for r in pool), min(n, len(pool)))
        )
    return bootstrap_mean(values)
