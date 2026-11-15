"""Bounded Algorithm 1: example-only filtering, learned repair and prefix refinement."""

import torch

from .models import encode_batch, generate


def repair_prompt(problem, code, result):
    return (
        problem.prompt()
        + "\nBUGGY PROGRAM:\n"
        + code
        + "\nTEST OUTCOME:\n"
        + result["outcome"]
        + "\nERROR:\n"
        + result["error"]
        + "\nCORRECT PROGRAM:\n"
    )


def prefix_length(pass_probabilities):
    """§3.3.4: split at max passing score, chop at the first predicted failure."""
    if not pass_probabilities:
        return 0
    end = max(range(len(pass_probabilities)), key=pass_probabilities.__getitem__) + 1
    for i, p in enumerate(pass_probabilities[:end]):
        if p < 0.5:
            return i
    return end


@torch.inference_mode()
def critic_score(critic, tokenizer, problem, candidate, cfg):
    device = next(critic.parameters()).device
    critic.eval()
    batch = encode_batch(
        tokenizer,
        [problem.prompt()],
        [candidate["code"]],
        cfg["max_source"],
        cfg["max_target"],
        device,
        token_ids=[candidate["token_ids"]],
    )
    logits, token_logits, mask = critic(**batch)
    return {
        "pass_probability": float(logits.softmax(-1)[0, -1].cpu()),
        "token_pass_probabilities": token_logits.softmax(-1)[0, :, -1][mask[0]].cpu().tolist(),
    }


def sampling_kwargs(cfg):
    return dict(
        batch_size=cfg["generation_batch"],
        max_source=cfg["max_source"],
        max_target=cfg["max_target"],
        temperature=cfg["temperature"],
        top_p=cfg["top_p"],
    )


def critic_sampling(
    actor, critic, repair, tokenizer, problem, runner, cfg, initial=None, refine=True, repair_top_m=1
):
    """No Problem/hidden-test object is accepted. Always returns exactly N samples.

    One repair and one refinement round, matching the paper's practical limit.
    On exhausted repair, return failed candidates; never secretly use references.
    """
    from .data import PublicProblem

    if not isinstance(problem, PublicProblem):
        raise TypeError("Critic sampling accepts only PublicProblem (no hidden tests or reference programs)")
    n = cfg["samples"]
    candidates = (
        initial
        if initial is not None
        else generate(actor, tokenizer, problem.prompt(), n, **sampling_kwargs(cfg))
    )
    if len(candidates) != n:
        raise ValueError("Initial population must contain exactly samples candidates")
    trace = []

    def test_and_score(population, stage):
        tested = []
        for candidate in population:
            result = runner.run(candidate["code"], problem, problem.examples)
            score = critic_score(critic, tokenizer, problem, candidate, cfg)
            tested.append({**candidate, "example_result": result, **score})
        trace.append(
            {
                "stage": stage,
                "population": tested,
                "passed_examples": sum(c["example_result"]["outcome"] == "passed" for c in tested),
            }
        )
        return tested

    candidates = test_and_score(candidates, "initial")
    if all(c["example_result"]["outcome"] == "no_tests" for c in candidates):
        return candidates, trace + [{"stage": "skipped", "reason": "no public example tests"}]
    passed = [c for c in candidates if c["example_result"]["outcome"] == "passed"]
    if not passed and repair_top_m > 0:
        if repair is None:
            raise ValueError("Repair was requested without a trained repair model")
        selected = sorted(candidates, key=lambda c: c["pass_probability"], reverse=True)[
            : min(repair_top_m, n)
        ]
        repaired = []
        for i, candidate in enumerate(selected):
            count = n // len(selected) + int(i < n % len(selected))
            prompt = repair_prompt(problem, candidate["code"], candidate["example_result"])
            repaired.extend(generate(repair, tokenizer, prompt, count, **sampling_kwargs(cfg)))
        candidates = test_and_score(repaired, "repair")
        passed = [c for c in candidates if c["example_result"]["outcome"] == "passed"]
    if passed and refine:
        refined = []
        for i, candidate in enumerate(passed):
            ids = candidate["token_ids"]
            usable = ids[: ids.index(tokenizer.eos_token_id)] if tokenizer.eos_token_id in ids else ids
            length = min(
                prefix_length(candidate["token_pass_probabilities"][: len(usable)]),
                len(usable),
                cfg["max_target"] - 1,
            )
            count = n // len(passed) + int(i < n % len(passed))
            refined.extend(
                generate(
                    actor, tokenizer, problem.prompt(), count, prefix=usable[:length], **sampling_kwargs(cfg)
                )
            )
            trace.append({"stage": "prefix", "prefix_token_ids": usable[:length], "copies": count})
        candidates = test_and_score(refined, "refine")
    assert len(candidates) == n
    return candidates, trace
