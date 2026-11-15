"""Rebuild analysis and figures from measured artifacts, never inventing a score."""

import io
import tokenize
from collections import Counter
from pathlib import Path

from .io import read_json, read_jsonl, write_json


def normalized_lines(code):
    try:
        tokens = [
            t
            for t in tokenize.generate_tokens(io.StringIO(code).readline)
            if t.type not in (tokenize.COMMENT, tokenize.ENCODING)
        ]
        code = tokenize.untokenize(tokens)
    except (tokenize.TokenError, IndentationError):
        pass
    return ["".join(line.split()) for line in code.splitlines() if line.strip()]


def overlap_analysis(source_codes, target_codes):
    source = {line for code in source_codes for line in normalized_lines(code)}
    target = [normalized_lines(code) for code in target_codes]
    counts = Counter(line for lines in target for line in lines)
    rows = []
    for i, lines in enumerate(target):
        eligible = [line for line in lines if counts[line] <= 2]
        copied = [line for line in eligible if line in source]
        rows.append(
            {
                "target_index": i,
                "eligible_lines": len(eligible),
                "overlap_lines": len(copied),
                "fraction": len(copied) / len(eligible) if eligible else 0.0,
            }
        )
    return {
        "source_programs": len(source_codes),
        "target_programs": len(target_codes),
        "rows": rows,
        "method": "Strip comments and whitespace; exclude target lines occurring more than twice (paper §4.5).",
        "over_half_fraction": sum(r["fraction"] > 0.5 for r in rows) / len(rows) if rows else None,
    }


def build_report(output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    manifest = read_json(output / "manifest.json")
    cfg = manifest["identity"]["config"]
    metrics = {p.stem: read_json(p) for p in sorted((output / "metrics").glob("*.json"))}
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(10, 5))
    for stage in ["msp", "ntp", "sft", "critic4", "critic2", "repair", "coderl"]:
        path = output / "training" / f"{stage}.jsonl"
        if path.exists():
            rows = read_jsonl(path)
            ax.plot([r["step"] for r in rows], [r["loss"] for r in rows], label=stage, linewidth=1.4)
    ax.set(
        xlabel="Optimizer step (per stage)",
        ylabel="Recorded loss",
        title="CodeRL: measured optimization traces",
    )
    ax.legend(ncol=4)
    fig.tight_layout()
    fig.savefig(figures / "training.png")
    plt.close(fig)
    names = [k for k in metrics if not k.endswith("beam")]
    fig, ax = plt.subplots(figsize=(11, 5))
    labels = ["compile_error", "runtime_error", "failed", "passed"]
    colors = ["#e17055", "#f5b041", "#5dade2", "#28b463"]
    bottom = [0.0] * len(names)
    for label, color in zip(labels, colors):
        vals = [metrics[n]["outcomes"].get(label, 0) / max(1, metrics[n]["candidates"]) * 100 for n in names]
        ax.bar(names, vals, bottom=bottom, label=label, color=color)
        bottom = [a + b for a, b in zip(bottom, vals)]
    ax.set(
        ylabel="Generated candidates (%)",
        title="Hidden-test outcomes (MBPP: published assertions)",
        ylim=(0, 100),
    )
    ax.tick_params(axis="x", rotation=35, labelsize=8)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    fig.tight_layout()
    fig.savefig(figures / "outcomes.png")
    plt.close(fig)
    table = []
    for name, m in metrics.items():
        metric_key = "beam_pass@1" if "beam_pass@1" in m else "pass@1"
        one = m.get(metric_key, {}).get("mean")
        key5 = "beam_pass@5" if "beam_pass@5" in m else "pass@5"
        five = m.get(key5, {}).get("mean")
        table.append(
            f"| {name} | {m['problems']} | {m['candidates']} | {'—' if one is None else f'{one:.3%}'} | {'—' if five is None else f'{five:.3%}'} |"
        )
    losses = {}
    for p in sorted((output / "training").glob("*.jsonl")):
        rows = read_jsonl(p)
        if rows:
            losses[p.stem] = {"first": rows[0]["loss"], "last": rows[-1]["loss"], "steps": len(rows)}
    total_seconds = sum(s["seconds"] for s in manifest["stages"].values() if s["status"] == "complete")
    outcomes = Counter()
    for m in metrics.values():
        outcomes.update(m["outcomes"])
    policy = (
        read_jsonl(output / "training" / "coderl.jsonl")
        if (output / "training" / "coderl.jsonl").exists()
        else []
    )
    nonzero = sum(abs(r.get("rl_loss", 0)) > 1e-10 for r in policy)
    body = (
        f"""# Experiment report — {cfg.get("name", output.name)}

Scope: **{cfg.get("scope", "reduced-scale local experiment")}**. Status: `{manifest["status"]}`.
This is a measured local run, not a reproduction of the paper's published accuracy.

## Configuration and evidence

- Hardware: {manifest["environment"]["cpu"]}; {manifest["environment"]["machine"]}; device `{manifest["environment"]["device"]}`.
- Model: `{cfg["model"]}`; seed {cfg["seed"]}; source/target limits {cfg["max_source"]}/{cfg["max_target"]}.
- Measured stage time: {total_seconds:.1f} seconds. See [manifest](manifest.json) for configuration, versions, hashes and each stage's timing.
- [Events](events.jsonl), `training/`, `evaluations/`, `traces/`, `critic_scores.jsonl`, and `checkpoints/` contain original artifacts.
- MSP/NTP use this run's training reference corpus only. They exercise the objectives; they do not recreate CSN/GCPY pretraining.
- Each ablation starts from the same SFT checkpoint, uses the same frozen rollout bank and seed.

## Results

| Experiment | Problems | Candidates | pass@1 / beam top-1 | pass@5 / beam top-5 |
|---|---:|---:|---:|---:|
"""
        + "\n".join(table)
        + f"""

Nucleus scores use the unbiased finite-population pass@k estimator; beam scores use ranked top-k success.
Critic Sampling produces dependent candidates: its estimator is descriptive and lacks the IID guarantee of independent nucleus draws.
Values for k greater than the observed population are never extrapolated. MBPP includes its three assertions in the input, as specified by the paper.
`n@k` filters on public examples before inspecting final outcomes, with fallback to all candidates if none pass examples.

## Interpretation

- Total recorded outcomes across evaluation variants: `{dict(outcomes)}`.
- CodeRL steps with a nonzero policy-loss value: **{nonzero}/{len(policy)}**. Zero-advantage samples produce no policy gradient, by design; a decreasing CE loss alone does not demonstrate RL improvement.
- Small training sets, short optimization schedules, sequence truncation and a much smaller model limit task accuracy. No statistically supported improvement over the paper or over SFT is claimed.
- Bootstrap intervals in the metric JSON resample problems. With very few problems and all-zero/all-one outcomes they are degenerate and do not establish precision.
- Qualitative failure details and the actual generated programs are retained for every problem; example-test outcomes are separate from final-test outcomes.

![Optimization traces](figures/training.png)

![Test outcomes](figures/outcomes.png)

## Reproduction boundaries

This run verifies mechanisms at the recorded scale. It does not complete the 770M model's multi-week, 16-A100 pretraining, full 5,000-problem APPS evaluation at 1,000 candidates, 500-problem MBPP evaluation at 80/1,000 candidates, GPT-Neo/GPT-J model comparisons, or the CodeXGLUE appendix. See [paper coverage](../../docs/paper-coverage.md).
"""
    )
    (output / "analysis.md").write_text(body)
    write_json(
        output / "analysis.json",
        {
            "stage_seconds": total_seconds,
            "losses": losses,
            "nonzero_policy_steps": nonzero,
            "metrics": metrics,
        },
    )
    return output / "analysis.md"
