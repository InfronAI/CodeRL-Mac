# Diagnostic interpretation

Run: `tiny-mps-final`; scope: Mechanism demonstration only: randomly initialized T5, byte tokenizer, authored addition paraphrases with the same target algorithm; no benchmark/generalization claim

- `sft`: 10/10 candidates end in EOS; 0/10 reach the token limit without EOS. Observed lengths: 35–38 tokens.
- `coderl`: 10/10 candidates end in EOS; 0/10 reach the token limit without EOS. Observed lengths: 38–40 tokens.
- Held-out critic accuracy is 70.0%, across only 2 observed outcome classes and 1 predicted classes. A majority-class predictor can score highly here; this does not establish a useful four-class critic.
- CodeRL has 2/10 nonzero policy-loss steps. The fixed rollout bank contains relative returns {'0.0': 16, '2.0': 4}.
- End-of-sequence termination below the configured cap is not evidence of length truncation. The small-model benchmark failures cannot be attributed to the output cap when every sample emitted EOS.
- Tiny CPU policy training can deteriorate despite correct loss mechanics; a single-seed, tiny-corpus run is a stability diagnostic. The MPS run's higher fixture pass rate is not a robust generalization result.

## First candidate on the first problem, by method

These examples are selected by position, not by success, and are separate from aggregate metrics.

### sft: fixture/test/4

Outcome: `compile_error`. Error: `SyntaxError: '(' was never closed (<candidate>, line 1)`

```python
print(sum(map(int,input().split()))

```

### coderl: fixture/test/4

Outcome: `passed`. Error: ``

```python
print(sum(map(int,input().split())))

```

### cs_repair1: fixture/test/4

Outcome: `compile_error`. Error: `SyntaxError: unmatched ')' (<candidate>, line 1)`

```python
print(sum(map(int,input().split())))))

```
