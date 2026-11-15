import math
import random

import pytest
import torch

from coderl.data import Problem, extract_examples
from coderl.metrics import filtered_n_at_k, pass_at_k, summarize
from coderl.models import Critic, encode_batch, generate, load_actor, policy_loss
from coderl.pretraining import msp_pair, ntp_pair
from coderl.sampling import prefix_length


@pytest.mark.parametrize("n,c,k", [(10, 3, 5), (5, 5, 1), (5, 0, 5), (1000, 3, 5)])
def test_passk_combinatorial(n, c, k):
    expected = 1 - (math.comb(n - c, k) / math.comb(n, k) if n - c >= k else 0)
    assert pass_at_k(n, c, k) == pytest.approx(expected)


def test_insufficient_k_and_filter():
    with pytest.raises(ValueError):
        pass_at_k(2, 1, 5)
    rows = [
        {"problem_id": "x", "result": {"outcome": "passed"}, "example_result": {"outcome": "failed"}},
        {"problem_id": "x", "result": {"outcome": "failed"}, "example_result": {"outcome": "passed"}},
    ]
    assert summarize(rows)["pass@5"]["mean"] is None
    assert filtered_n_at_k(rows, 1)["mean"] == 0  # cannot choose by hidden outcome


def test_policy_mask_sign_and_detachment():
    logits = torch.zeros(1, 3, 4, requires_grad=True)
    labels = torch.tensor([[1, 2, -100]])
    weights = torch.tensor([[1.0, -1.0, 1000.0]], requires_grad=True)
    loss = policy_loss(logits, labels, weights)
    loss.backward()
    assert logits.grad[0, 0, 1] < 0
    assert logits.grad[0, 1, 2] > 0
    assert not logits.grad[0, 2].any()
    assert weights.grad is None


def test_zero_advantage_gradient():
    logits = torch.randn(1, 2, 5, requires_grad=True)
    policy_loss(logits, torch.tensor([[1, 3]]), torch.zeros(1, 2)).backward()
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_critic_pooling_padding_and_save(tmp_path):
    torch.manual_seed(1)
    actor, tok = load_actor("tiny")
    critic = Critic(actor).eval()
    batch = encode_batch(tok, ["sum"], ["print(1)"])
    logits, tokens, mask = critic(**batch)
    padded = {**batch, "labels": torch.cat([batch["labels"], torch.full((1, 5), -100)], dim=1)}
    other, _, _ = critic(**padded)
    assert torch.allclose(logits, other, atol=1e-5)
    assert tokens.shape == (*mask.shape, 4)
    critic.save_pretrained(tmp_path, tok)
    loaded, _ = Critic.from_pretrained(tmp_path)
    loaded.eval()
    assert torch.allclose(logits, loaded(**batch)[0])


def test_generation_remainder_and_prefix():
    actor, tok = load_actor("tiny")
    prefix = tok.encode("x", add_special_tokens=False)
    rows = generate(actor, tok, "x", count=3, batch_size=2, max_target=4, prefix=prefix)
    assert len(rows) == 3
    assert all(r["token_ids"][0] == prefix[0] for r in rows)


@pytest.mark.parametrize(
    "values,expected", [([0.8, 0.9, 0.7], 2), ([0.8, 0.4, 0.9], 1), ([0.3, 0.7], 0), ([], 0)]
)
def test_prefix_selection(values, expected):
    assert prefix_length(values) == expected


def test_pretraining_objectives():
    rng = random.Random(4)
    ids = list(range(100))
    left, right = ntp_pair(ids, rng)
    assert 10 <= len(left) <= 90 and left + right == ids
    source, target = msp_pair(ids, rng, list(range(200, 300)))
    assert sum(x < 200 for x in target) == 15
    assert sum(x < 200 for x in source) == 85


def test_public_boundary_and_examples():
    p = Problem("x", "question", ["SECRET"], {"inputs": ["HIDDEN"], "outputs": ["SECRET"]})
    public = p.public()
    assert not hasattr(public, "solutions") and not hasattr(public, "tests")
    assert "SECRET" not in public.prompt()
    question = "-----Example-----\nInput\n1 2\nOutput\n3\n\n-----Note-----\nhello"
    assert extract_examples(question) == {"inputs": ["1 2\n"], "outputs": ["3\n"]}


def test_repair_prompt_truncation_keeps_feedback():
    from coderl.data import PublicProblem
    from coderl.sampling import repair_prompt

    _, tok = load_actor("tiny")
    prompt = repair_prompt(
        PublicProblem("x", "description " * 1000),
        "bad code " * 100,
        {"outcome": "failed", "error": "WrongAnswer"},
    )
    batch = encode_batch(tok, [prompt], max_source=384)
    text = tok.decode(batch["input_ids"][0].tolist())
    assert "BUGGY PROGRAM" in text and "TEST OUTCOME" in text and "WrongAnswer" in text
    assert batch["input_ids"].shape[1] <= 384


def test_stage_resume_verifies_artifacts(tmp_path):
    from coderl.io import write_json
    from coderl.pipeline import Experiment, config_from

    cfg = config_from("configs/tiny.json")
    cfg["device"] = "cpu"
    exp = Experiment(cfg, tmp_path / "run")
    path = exp.output / "example.json"

    def work():
        write_json(path, {"value": 1})
        return {"rows": 1}, [path]

    exp.stage("check", work)
    cached = Experiment(cfg, exp.output, resume=True)
    cached.stage("check", lambda: pytest.fail("completed stage must not rerun"))
    write_json(path, {"value": 2})
    with pytest.raises(ValueError, match="artifact missing/changed"):
        cached.stage("check", work)
    with pytest.raises(ValueError, match="identity changed"):
        Experiment({**cfg, "seed": 999}, exp.output, resume=True)
