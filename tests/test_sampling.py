import pytest

from coderl.data import Problem, PublicProblem
from coderl.sampling import critic_sampling


class Tokenizer:
    eos_token_id = 1


class Runner:
    def run(self, code, problem, tests):
        return {"outcome": "passed" if code == "good" else "failed", "error": "WrongAnswer"}


def test_repair_and_refine_preserve_budget(monkeypatch):
    calls = []

    def gen(model, tokenizer, prompt, count, **kwargs):
        calls.append((model, count, prompt, kwargs.get("prefix")))
        return [{"code": "good", "token_ids": [5, 6, 1]} for _ in range(count)]

    monkeypatch.setattr("coderl.sampling.generate", gen)
    monkeypatch.setattr(
        "coderl.sampling.critic_score",
        lambda *args: {"pass_probability": 0.7, "token_pass_probabilities": [0.8, 0.9, 0.7]},
    )
    cfg = {
        "samples": 5,
        "generation_batch": 2,
        "max_source": 50,
        "max_target": 10,
        "temperature": 0.6,
        "top_p": 0.95,
    }
    initial = [{"code": "bad", "token_ids": [5, 1]} for _ in range(5)]
    result, trace = critic_sampling(
        "actor",
        "critic",
        "repair",
        Tokenizer(),
        PublicProblem("x", "x"),
        Runner(),
        cfg,
        initial,
        repair_top_m=2,
    )
    assert len(result) == 5
    assert sum(count for model, count, _, _ in calls if model == "repair") == 5
    assert [t["stage"] for t in trace if t["stage"] in ("initial", "repair", "refine")] == [
        "initial",
        "repair",
        "refine",
    ]
    assert any(prefix for _, _, _, prefix in calls)


def test_reject_hidden_problem():
    with pytest.raises(TypeError):
        critic_sampling(None, None, None, None, Problem("x", "x", ["hidden"], {}), None, {})
