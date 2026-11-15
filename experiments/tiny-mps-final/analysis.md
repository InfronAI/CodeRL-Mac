# Experiment report — offline-tiny

Scope: **Mechanism demonstration only: randomly initialized T5, byte tokenizer, authored addition paraphrases with the same target algorithm; no benchmark/generalization claim**. Status: `complete`.
This is a measured local run, not a reproduction of the paper's published accuracy.

## Configuration and evidence

- Hardware: Apple M1 Pro; arm64; device `mps`.
- Model: `tiny`; seed 42; source/target limits 192/64.
- Measured stage time: 115.7 seconds. See [manifest](manifest.json) for configuration, versions, hashes and each stage's timing.
- [Events](events.jsonl), `training/`, `evaluations/`, `traces/`, `critic_scores.jsonl`, and `checkpoints/` contain original artifacts.
- MSP/NTP use this run's training reference corpus only. They exercise the objectives; they do not recreate CSN/GCPY pretraining.
- Each ablation starts from the same SFT checkpoint, uses the same frozen rollout bank and seed.

## Results

| Experiment | Problems | Candidates | pass@1 / beam top-1 | pass@5 / beam top-5 |
|---|---:|---:|---:|---:|
| absolute_beam | 2 | 10 | 100.000% | 100.000% |
| ce_only_beam | 2 | 10 | 100.000% | 100.000% |
| ce_synthetic_beam | 2 | 10 | 100.000% | 100.000% |
| coderl | 2 | 10 | 70.000% | 100.000% |
| coderl_beam | 2 | 10 | 100.000% | 100.000% |
| cs_refine | 2 | 10 | 80.000% | 100.000% |
| cs_repair1 | 2 | 10 | 80.000% | 100.000% |
| cs_repair2 | 2 | 10 | 80.000% | 100.000% |
| cs_repair4 | 2 | 10 | 80.000% | 100.000% |
| distance_beam | 2 | 10 | 100.000% | 100.000% |
| no_critic_beam | 2 | 10 | 100.000% | 100.000% |
| rl_only_beam | 2 | 10 | 100.000% | 100.000% |
| sft | 2 | 10 | 30.000% | 100.000% |
| sft_beam | 2 | 10 | 0.000% | 100.000% |

Nucleus scores use the unbiased finite-population pass@k estimator; beam scores use ranked top-k success.
Critic Sampling produces dependent candidates: its estimator is descriptive and lacks the IID guarantee of independent nucleus draws.
Values for k greater than the observed population are never extrapolated. MBPP includes its three assertions in the input, as specified by the paper.
`n@k` filters on public examples before inspecting final outcomes, with fallback to all candidates if none pass examples.

## Interpretation

- Total recorded outcomes across evaluation variants: `{'passed': 58, 'compile_error': 82}`.
- CodeRL steps with a nonzero policy-loss value: **2/10**. Zero-advantage samples produce no policy gradient, by design; a decreasing CE loss alone does not demonstrate RL improvement.
- Small training sets, short optimization schedules, sequence truncation and a much smaller model limit task accuracy. No statistically supported improvement over the paper or over SFT is claimed.
- Bootstrap intervals in the metric JSON resample problems. With very few problems and all-zero/all-one outcomes they are degenerate and do not establish precision.
- Qualitative failure details and the actual generated programs are retained for every problem; example-test outcomes are separate from final-test outcomes.

![Optimization traces](figures/training.png)

![Test outcomes](figures/outcomes.png)

## Reproduction boundaries

This run verifies mechanisms at the recorded scale. It does not complete the 770M model's multi-week, 16-A100 pretraining, full 5,000-problem APPS evaluation at 1,000 candidates, 500-problem MBPP evaluation at 80/1,000 candidates, GPT-Neo/GPT-J model comparisons, or the CodeXGLUE appendix. See [paper coverage](../../docs/paper-coverage.md).
