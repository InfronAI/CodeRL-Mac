# Measured experiment results

These are local reduced-scale experiments, not a reproduction of the published CodeRL scores.
All scores below are computed from saved candidate programs and actual test outcomes.

| Run | Device | Completed stages | Stage seconds | SFT pass@1 | CodeRL pass@1 | CS M=1 pass@1 | Nonzero RL steps |
|---|---|---:|---:|---:|---:|---:|---:|
| tiny-mps-final | mps | 29 | 115.7 | 30.0% | 70.0% | 80.0% | 2/10 |
| tiny-cpu-final | cpu | 29 | 32.9 | 20.0% | 0.0% | 0.0% | 8/10 |
| m1-codet5-small | mps | 33 | 917.8 | 0.0% | 0.0% | 0.0% | 4/10 |

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
