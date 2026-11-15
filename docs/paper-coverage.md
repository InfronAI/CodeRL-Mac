# Paper-to-implementation audit

Source: [CodeRL, NeurIPS 2022, arXiv v3](https://arxiv.org/html/2207.01780v3), reviewed on 2026-11-14. The original upstream commit is `ec3c526a07e44f56f702f354cbad11187bdf25cc`.

**Implemented mechanisms and reproduced scientific results are different claims. This project does not claim 100% reproduction of the paper.** All run reports identify model, data, budget, hardware, actual metrics and missing experiments. The original paper used 16 A100 40 GB GPUs for about 21 days of pretraining; the available machine is a 16 GB M1 Pro. See the paper's §4.1 for the original resource budget.

| Paper item | Implementation | Local execution / limit |
|---|---|---|
| §3.2 masked span prediction | `pretraining.msp_pair`, `training.train_pretraining` | Actual updates on the local training reference corpus. 15% noise; short random spans. This corpus and span-length sampler differ from the original CSN/GCPY pretraining setup. |
| §3.2 NTP | `pretraining.ntp_pair` | Uniform pivot constrained to 10–90%; actual encoder/decoder updates. |
| §3.4 CE warm start | `training.train_ce` | Actual full-parameter FP32 updates; local steps and fixed learning rate are recorded. No 10-epoch full APPS training claim. |
| Eq. 4–7 rewards | `execution.REWARDS` | Compile −1; runtime −0.6; failed tests −0.3; all passed +1. Empty tests never pass. |
| Eq. 8 greedy baseline | `pipeline.collect_rollouts` | Greedy programs are generated and executed for every training problem. |
| Eq. 9 four-class critic | `models.Critic`, `training.train_critic` | Masked max-pool of decoder hidden states, then a linear head, as described in the paper. |
| Eq. 10/14 dense RL | `training.score_rows`, `training.train_rl` | Actual sampled IDs retained. Critic weight uses the observed outcome's probability, not always the passing class. Relative return is detached. CE + RL have equal weights. |
| Binary critic | `models.Critic(classes=2)` | Separate model and checkpoint; train on binary execution labels. |
| Eq. 13 repair model | `sampling.repair_prompt`, `pipeline.repair_training` | Separate seq2seq model trained on failed samples → reference code. Repair input reserves space for the description, buggy program and error. |
| Algorithm 1 CS | `sampling.critic_sampling` | Complete generation → public tests → top-M repair → prefix refinement. Exactly N outputs even when N is not divisible by M. Maximum one repair and one refine round, matching §4.1's practical bound. |
| Hidden/public separation | `data.PublicProblem` | CS cannot receive reference solutions or hidden tests. MBPP's assertions are intentionally public, following the paper. |
| Table 1 APPS | `metrics.summarize`, `filtered_n_at_k` | Actual stratified six-problem smoke evaluation. No full 5,000-problem, 1,000-candidate result. Difficulty strata are reported. |
| Table 2 return ablations A/B/C/D | `no_critic`, `absolute`, `distance`, `coderl` variants | Actual independent optimization from the same warm start and rollout bank. Beam top-1/top-5 evaluations. |
| Table 3 CodeT5 objective ablations | SFT / `ce_only` / `ce_synthetic` / `rl_only` / `coderl` | Actual optimization and beam evaluation. Synthetic CE mixes reference and generated programs. GPT-Neo branch is **not implemented** in the modern T5-only backend. |
| Table 4 CS ablations | no CS / refine / repair M=1,2,4 | All configurations executed locally; traces show which branches are reached on each problem. Scores at 200/1,000 candidates are **not reproduced**. |
| Table 5 / Fig. 6 scale and corpus comparisons | Model IDs and local checkpoints configurable | 60M and tiny backbones exercised. 220M/770M, CSN/GCPY corpus sweeps and per-epoch comparison are **not run**. The original corpus-processing/scheduler recipe is not fully implemented. |
| Fig. 5 competition tasks | Difficulty-stratified APPS metrics | Two competition problems in local smoke. GPT-J 6B comparison is **not implemented/run**. |
| Table 6 MBPP zero shot | `pipeline.evaluate` after APPS training | Actual evaluation without MBPP updates to the APPS model. Three test problems, five samples each; no claim of pass@80. |
| Fig. 7 overlap | `reporting.overlap_analysis` | All 974 MBPP programs vs 101 references from five bundled APPS train problems. Smaller source corpus; not the original full-data estimate. |
| Fig. 8 / Appendix C failures | `evaluations/*.jsonl`, outcome plots | Every program, execution category, error, example/final distinction retained. |
| Appendix B.1 CodeXGLUE | Original CodeT5 reference retained in upstream docs | Summarization, translation, refinement and CodeBLEU experiments are **not implemented/run** in this modernization. |
| Appendix B.2 MBPP finetuning | Separate `mbpp_finetuned` checkpoint | Actual updates on four MBPP training examples and evaluation on three test examples; no full 60-epoch/full-data result. |
| External baselines (GPT/Codex/AlphaCode) | None | Not retrained, queried, or relabeled as local measurements. |

## Deliberate implementation choices

- Use upstream Transformers 4.46.3 and PyTorch 2.9.1 instead of modifying the entire 2022 Transformers fork. The old fork remains under `transformers/` for source inspection and is **not installed** by the modern package.
- Paper text specifies pooling hidden states before a linear head. The released fork instead applies a two-layer head first and pools logits. `Critic(pooling="paper")` follows the paper; `pooling="legacy_logits"` implements the released order with padding correctly masked. Existing legacy checkpoints are not silently treated as new-format critics.
- Critic and actor use the same small architecture in the local profiles, sharing no trainable parameters after initialization. The full paper uses a smaller critic relative to the 770M actor.
- The offline tiny profile uses a randomly initialized T5 and a byte tokenizer. Its addition paraphrases share one target algorithm. Its positive execution results establish an executable mechanism demonstration, not generalization or benchmark quality.
- Losses average valid tokens (as in the released implementation). Padding and critic signals are masked. Gradients are accumulated explicitly and clipped; nonfinite gradients abort the stage.
- Training uses a fixed, saved synthetic rollout bank, matching the released offline training workflow. It is not an online PPO algorithm.
- The runner uses ordered token comparison with floating-point tolerance. It does not use the legacy evaluator's permissive unordered-token comparison, which can accept incorrect ordered answers. Problems with nonunique valid outputs need task-specific checkers. `experiments/reference-validation.json` identifies observed reference mismatches; they are not silently filtered out of the frozen sample.
- An APPS call-based singleton expected-output wrapper is supported. No-test and evaluator-infrastructure failures are distinguished from legitimate program failures.
- Generated Python executes in a subprocess. On macOS Seatbelt denies network access, process forks, home-directory reads and writes outside a temporary directory; CPU/time/output/file limits bound execution. This is a local research executor, not a hardened multi-tenant service. Linux uses process/resource limits only.
- Checkpoints support **stage-level resume**. Optimizer state within an interrupted stage is not resumed. Any change in source, configuration, data or saved artifacts is detected before reuse.

## What would finish the remaining paper reproduction

Implement and validate the causal-model backend and the exact large-corpus preparation/schedules; obtain full CSN/GCPY, APPS and CodeXGLUE datasets; run original-size model/corpus/epoch sweeps and the complete generation budgets; and compare full-dataset confidence intervals and metrics. These are outstanding work, not completed milestones. The local profiles are explicitly scoped evidence toward that work.

## Evaluator compatibility evidence

[`experiments/upstream-reexecution/summary.json`](../experiments/upstream-reexecution/summary.json) re-executes all 83 saved upstream synthetic programs through the new executor. It covers all four outcome classes. These are historical samples, not new generations and not a held-out model score. The 79/83 category matches and four disagreements are retained individually: Python script compilation versus the original function-wrapping scheme, load-time `NameError` classification, and first-failure short circuiting explain the observed differences. This establishes useful compatibility evidence but does not justify claiming the evaluators are identical.

The isolated worker currently uses Python's standard library and supplies typing annotations. It does not automatically inject the old evaluator's broad collection/math/NumPy imports, nor enable arbitrary third-party dependencies. Programs must import the standard-library modules they use. This is an additional evaluator-environment difference when comparing to the original benchmark.
