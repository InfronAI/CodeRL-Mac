"""Cross-run analyses derived only from saved records; optional held-out critic scoring."""

import argparse
import csv
from collections import Counter
from pathlib import Path

from coderl.data import load_problems
from coderl.execution import outcome_label, reward
from coderl.io import read_json, read_jsonl, write_json


def analyze(root, score_critic=False):
    manifest = read_json(root / "manifest.json")
    cfg = manifest["identity"]["config"]
    rollout = read_jsonl(root / "rollouts.jsonl")
    advantage = Counter(str(round(reward(r["result"]) - reward(r["baseline_result"]), 3)) for r in rollout)
    branches = Counter()
    for path in (root / "traces").glob("*.json"):
        for problem in read_json(path):
            for entry in problem["trace"]:
                branches[entry["stage"]] += 1
    history = read_jsonl(root / "training/coderl.jsonl")
    result = {
        "run": root.name,
        "status": manifest["status"],
        "model": cfg["model"],
        "device": manifest["environment"]["device"],
        "scope": cfg["scope"],
        "stage_seconds": sum(s["seconds"] for s in manifest["stages"].values() if s["status"] == "complete"),
        "stages_complete": sum(s["status"] == "complete" for s in manifest["stages"].values()),
        "rollout_outcomes": dict(Counter(r["result"]["outcome"] for r in rollout)),
        "relative_return_distribution": dict(advantage),
        "cs_branches": dict(branches),
        "nonzero_policy_steps": sum(abs(r["rl_loss"]) > 1e-10 for r in history),
        "policy_steps": len(history),
        "metrics": {p.stem: read_json(p) for p in sorted((root / "metrics").glob("*.json"))},
    }
    result["generation_lengths"] = {}
    for variant in ("sft", "coderl"):
        generated = read_jsonl(root / "evaluations" / f"{variant}.jsonl")
        eos = 1 if cfg["model"] == "tiny" else read_json(root / "checkpoints/sft/config.json")["eos_token_id"]
        lengths = [len(row["token_ids"]) for row in generated]
        result["generation_lengths"][variant] = {
            "candidates": len(generated),
            "configured_max_target": cfg["max_target"],
            "min": min(lengths),
            "max": max(lengths),
            "mean": sum(lengths) / len(lengths),
            "ended_with_eos": sum(row["token_ids"][-1] == eos for row in generated),
            "hit_limit_without_eos": sum(
                len(row["token_ids"]) >= cfg["max_target"] and row["token_ids"][-1] != eos
                for row in generated
            ),
        }
    if score_critic:
        from coderl.models import Critic
        from coderl.training import score_rows

        problems = {p.id: p for p in load_problems(cfg["test_data"], "test")}
        rows = []
        for row in read_jsonl(root / "evaluations/sft.jsonl"):
            rows.append({**row, "prompt": problems[row["problem_id"]].public().prompt()})
        critic, tok = Critic.from_pretrained(root / "checkpoints/critic4")
        # Scoring on CPU keeps the diagnostic independent of MPS memory pressure.
        scored = score_rows(critic, tok, rows, cfg)
        confusion = [[0] * 4 for _ in range(4)]
        for row in scored:
            confusion[outcome_label(row["result"])][row["critic_prediction"]] += 1
        result["heldout_critic"] = {
            "scope": "SFT-generated programs on the frozen test subset; diagnostic only, no model selection",
            "examples": len(scored),
            "accuracy": sum(row["critic_prediction"] == outcome_label(row["result"]) for row in scored)
            / len(scored),
            "confusion_true_rows_predicted_columns": confusion,
            "class_order": ["compile_error", "runtime_error", "failed", "passed"],
        }
    write_json(root / "diagnostics.json", result)
    details = ["# Diagnostic interpretation", "", f"Run: `{root.name}`; scope: {cfg['scope']}", ""]
    for variant, lengths in result["generation_lengths"].items():
        details.append(
            f"- `{variant}`: {lengths['ended_with_eos']}/{lengths['candidates']} candidates end in EOS; "
            f"{lengths['hit_limit_without_eos']}/{lengths['candidates']} reach the token limit without EOS. "
            f"Observed lengths: {lengths['min']}–{lengths['max']} tokens."
        )
    if "heldout_critic" in result:
        diagnostic = result["heldout_critic"]
        confusion = diagnostic["confusion_true_rows_predicted_columns"]
        populated = sum(sum(row) > 0 for row in confusion)
        predicted = sum(sum(confusion[i][j] for i in range(4)) > 0 for j in range(4))
        details += [
            f"- Held-out critic accuracy is {diagnostic['accuracy']:.1%}, across only {populated} observed "
            f"outcome classes and {predicted} predicted classes. A majority-class predictor can score highly "
            "here; this does not establish a useful four-class critic.",
        ]
    details += [
        f"- CodeRL has {result['nonzero_policy_steps']}/{result['policy_steps']} nonzero policy-loss steps. "
        "The fixed rollout bank contains relative returns " + str(dict(advantage)) + ".",
        "- End-of-sequence termination below the configured cap is not evidence of length truncation. "
        "The small-model benchmark failures cannot be attributed to the output cap when every sample emitted EOS.",
        "- Tiny CPU policy training can deteriorate despite correct loss mechanics; a single-seed, tiny-corpus "
        "run is a stability diagnostic. The MPS run's higher fixture pass rate is not a robust generalization result.",
        "",
        "## First candidate on the first problem, by method",
        "",
        "These examples are selected by position, not by success, and are separate from aggregate metrics.",
        "",
    ]
    for variant in ["sft", "coderl", "cs_repair1"]:
        row = read_jsonl(root / "evaluations" / f"{variant}.jsonl")[0]
        details += [
            f"### {variant}: {row['problem_id']}",
            "",
            f"Outcome: `{row['result']['outcome']}`. Error: `{row['result']['error']}`",
            "",
            "```python",
            row["code"].replace("```", "'''"),
            "```",
            "",
        ]
    (root / "diagnostic-analysis.md").write_text("\n".join(details))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--score-critic", action="store_true")
    args = parser.parse_args()
    results = [analyze(Path(path), args.score_critic) for path in args.runs]
    out = Path("experiments")
    with (out / "summary.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "run",
                "model",
                "device",
                "experiment",
                "problems",
                "candidates",
                "pass_at_1",
                "pass_at_5",
                "decoding",
            ]
        )
        for r in results:
            for name, m in r["metrics"].items():
                one = m.get("pass@1", m.get("beam_pass@1", {})).get("mean")
                five = m.get("pass@5", m.get("beam_pass@5", {})).get("mean")
                writer.writerow(
                    [
                        r["run"],
                        r["model"],
                        r["device"],
                        name,
                        m["problems"],
                        m["candidates"],
                        one,
                        five,
                        m["decoding"],
                    ]
                )
    table = []
    for r in results:
        sft = r["metrics"]["sft"]["pass@1"]["mean"]
        rl = r["metrics"]["coderl"]["pass@1"]["mean"]
        cs = r["metrics"]["cs_repair1"]["pass@1"]["mean"]
        table.append(
            f"| {r['run']} | {r['device']} | {r['stages_complete']} | {r['stage_seconds']:.1f} | {sft:.1%} | {rl:.1%} | {cs:.1%} | {r['nonzero_policy_steps']}/{r['policy_steps']} |"
        )
    body = (
        """# Measured experiment results

These are local reduced-scale experiments, not a reproduction of the published CodeRL scores.
All scores below are computed from saved candidate programs and actual test outcomes.

| Run | Device | Completed stages | Stage seconds | SFT pass@1 | CodeRL pass@1 | CS M=1 pass@1 | Nonzero RL steps |
|---|---|---:|---:|---:|---:|---:|---:|
"""
        + "\n".join(table)
        + """

## Evidence and interpretation

- `summary.csv` contains every evaluation variant. Each run has `analysis.md`, `diagnostics.json`, a hashed `manifest.json`, `source.zip`, original program JSONL, error records, learning curves and full CS traces.
- Tiny uses a 694,528-parameter randomly initialized T5, byte tokenizer and authored paraphrases of the same addition algorithm. Its scores are a mechanism demonstration, not evidence of benchmark generalization.
- CodeT5-small uses an official pinned 60,492,288-parameter pretrained model. Five APPS train problems and a 40-step warm start are drastically smaller than the paper's training setup. Six APPS test problems cover three difficulty labels but cannot estimate full-benchmark quality. MBPP has three test problems and five samples each.
- Policy loss is exactly zero when a sampled program and greedy program earn the same reward. The nonzero counts reveal whether the RL term actually contributed; CE decreasing alone is not an RL result.
- Comparisons within a run reuse a warm-start checkpoint, fixed rollout bank and random seed. MPS floating-point behavior and different devices can produce different generations. Cross-device tiny outcomes are not a model-quality comparison.
- Small-model CPU latency can beat MPS because GPU dispatch overhead dominates. Runtime totals include training, generation, code execution and checkpoint I/O; they are observational workflow timings, not a controlled hardware benchmark. Some runs overlapped with other local work.
- Reports retain unfavorable results, missing categories, failed generations and degenerate confidence intervals. No improvement claim is made from this small sample.
- The exact-list APPS evaluator can reject nonunique valid answers. Reference mismatches are retained in `reference-validation.json`; no post-evaluation cherry-picking of test IDs occurs.
- `dataset-overlap.json` applies the paper's line-normalization method to 101 reference programs from the five bundled APPS problems and all 974 MBPP programs. The source corpus is much smaller than the paper's; its result cannot establish benchmark-wide absence of contamination.

## Unfinished original experiments

The 770M pretraining budget, full APPS/MBPP generation counts, model/corpus/epoch sweeps, GPT-Neo/GPT-J comparisons and CodeXGLUE appendix are not completed. See [the detailed coverage audit](../docs/paper-coverage.md). Large checkpoints are saved locally and ignored by Git; source, configuration and pinned data support regenerating them.
"""
    )
    (out / "RESULTS.md").write_text(body)
    zh = (
        """# 本机实测实验结果

以下是实际执行的缩小规模实验，不是论文榜单的完整复现。分数来自已保存的候选程序和真实测试结果。

| 实验 | 设备 | 完成阶段 | 阶段总秒数 | SFT pass@1 | CodeRL pass@1 | CS M=1 pass@1 | 非零 RL 步数 |
|---|---|---:|---:|---:|---:|---:|---:|
"""
        + "\n".join(table)
        + """

- tiny 使用随机初始化的 694,528 参数 T5、字节 tokenizer 和相同加法算法的同义题目。其分数只说明机制可运行，不代表基准泛化能力。
- CodeT5-small 使用固定版本的官方 60M 权重；APPS 仅 5 道训练题、40 步监督预热、6 道测试题；MBPP 仅 3 道测试题、每题 5 个候选。数据与预算远小于原论文。
- 当采样程序与 greedy 基线奖励相同时，RL 梯度为零。表中的非零步数直接核验 RL 项是否参与优化，不能仅凭 CE 损失下降判断 RL 有效。
- 同轮消融从相同检查点、同一候选库和随机种子开始；MPS 数值行为可能产生不同生成轨迹。不能将 CPU 与 MPS 的 tiny 分数差异解释为模型优劣。
- 运行时间包含训练、生成、执行测试与保存检查点，部分运行与本机其他工作并发，并非受控硬件性能基准。微型模型可能因 GPU 调度开销而在 CPU 上更快。
- 失败结果、未出现类别、参考答案不匹配、退化置信区间均保留。没有从小样本推出显著改进或论文水平结论。
- APPS 部分题目允许多种正确输出，当前有序比较需要专用 checker 才能完整处理，具体情况见 `reference-validation.json`。
- 数据重复分析只覆盖 5 道 APPS 训练题的 101 份参考程序，不能作为完整 APPS 与 MBPP 无数据污染的证明。

各实验的 `analysis.md`、`diagnostics.json`、`manifest.json`、`source.zip`、程序 JSONL、CS 轨迹及图表提供完整证据；`summary.csv` 汇总全部评测变体。大模型预训练、完整 APPS/MBPP 采样、规模与语料消融、GPT-Neo/GPT-J 和 CodeXGLUE 仍未完成，详见[论文覆盖审计](../docs/paper-coverage.md)。
"""
    )
    (out / "RESULTS.zh-CN.md").write_text(zh)
    print(out / "RESULTS.md")


if __name__ == "__main__":
    main()
