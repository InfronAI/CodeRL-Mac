# Diagnostic interpretation

Run: `m1-codet5-small`; scope: Reduced-scale CodeT5-small mechanism/benchmark smoke experiment: 5 APPS train problems, 6 test problems (2 per difficulty), 3 MBPP test problems; not paper-scale reproduction

- `sft`: 30/30 candidates end in EOS; 0/30 reach the token limit without EOS. Observed lengths: 100–100 tokens.
- `coderl`: 30/30 candidates end in EOS; 0/30 reach the token limit without EOS. Observed lengths: 81–100 tokens.
- Held-out critic accuracy is 100.0%, across only 1 observed outcome classes and 1 predicted classes. A majority-class predictor can score highly here; this does not establish a useful four-class critic.
- CodeRL has 4/10 nonzero policy-loss steps. The fixed rollout bank contains relative returns {'0.0': 19, '0.4': 1, '-0.4': 5}.
- End-of-sequence termination below the configured cap is not evidence of length truncation. The small-model benchmark failures cannot be attributed to the output cap when every sample emitted EOS.
- Tiny CPU policy training can deteriorate despite correct loss mechanics; a single-seed, tiny-corpus run is a stability diagnostic. The MPS run's higher fixture pass rate is not a robust generalization result.

## First candidate on the first problem, by method

These examples are selected by position, not by success, and are separate from aggregate metrics.

### sft: apps/test/4000

Outcome: `compile_error`. Error: `SyntaxError: invalid syntax (<candidate>, line 4)`

```python
yi = 0


yi =if the path isyi = 0
yi = 0
yi = 0
yi = 0
yi == 1
yi = 0
yi == 0
yi = 0
yi = 0
yi = 1
yi = 0
y i = 0
yi =yi = 0
yi = 0
y i =
```

### coderl: apps/test/4000

Outcome: `failed`. Error: `WrongAnswer`

```python
yi = 0


yi = 0
yi = 0
yi = 0
yi = 0
yi = 0
yi = yi + 1
yi = yi + 1
yi = yi + 1
yi - 1
yi = yi + 1
yi = yi + 1
yi+1
yi + 1
yi += 1
yi + 1

```

### cs_repair1: apps/test/4000

Outcome: `compile_error`. Error: `SyntaxError: invalid syntax (<candidate>, line 1)`

```python
b, c = a, c, a, b, c.split(','), b, c.split(','), b, c.split('',').split('','):



























































```
