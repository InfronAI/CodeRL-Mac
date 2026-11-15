# Saved experiments

- [`RESULTS.md`](RESULTS.md) / [`RESULTS.zh-CN.md`](RESULTS.zh-CN.md): generated cross-run analysis.
- [`summary.csv`](summary.csv): every measured evaluation variant.
- `tiny-mps-final/`, `tiny-cpu-final/`, `m1-codet5-small/`: final runs using the same package source, independently recorded device/configuration/data hashes and local checkpoints.
- `upstream-reexecution/`: compatibility checks using 83 pre-existing upstream programs, explicitly distinguished from newly generated programs.
- `dataset-overlap.json`: the paper's line-overlap diagnostic on the available APPS corpus and full MBPP.
- `reference-validation.json`: actual execution of the first available reference program for each selected benchmark problem.
- `validation.json`, `test-results.txt`, `lint-results.txt`, `dependency-check.txt`, `resume-verification.log`: verification evidence.
- `tiny-mps-seed42/`: an earlier exploratory run, before the evaluator's call-based singleton-output fix and source snapshot support. It is retained for transparency and excluded from the final cross-run comparison. Its source hash deliberately differs from the final package.

Each final run saves JSON/JSONL rather than untrusted pickle for measurements. Its `source.zip` preserves the exact package version; its manifest hashes each completed stage's files. Reports are generated from those records. Checkpoints are available locally under `checkpoints/` and excluded from Git because of their size.
