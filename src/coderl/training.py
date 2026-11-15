"""Explicit, memory-bounded CE, critic and actor-critic optimization loops."""

import math
import random
import time

import torch
from torch.nn import functional as F

from .execution import outcome_label, reward
from .models import encode_batch, policy_loss
from .pretraining import msp_pair, ntp_pair


def optimizer_for(model, lr):
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05, foreach=False)


def update(model, optimizer):
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if not torch.isfinite(norm):
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError("Non-finite gradient; refusing to save invalid weights")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return float(norm.detach().cpu())


def _record(history, start, step, **values):
    row = {"step": step + 1, "seconds": time.monotonic() - start, **values}
    if any(isinstance(v, float) and not math.isfinite(v) for v in row.values()):
        raise FloatingPointError(f"Non-finite training metric: {row}")
    history.append(row)
    if step == 0 or (step + 1) % 10 == 0:
        print(row, flush=True)


def train_ce(model, tokenizer, pairs, cfg, steps=None):
    if not pairs:
        raise ValueError("No supervised training pairs")
    steps = cfg["sft_steps"] if steps is None else steps
    optimizer = optimizer_for(model, cfg["lr"])
    device = next(model.parameters()).device
    history, start = [], time.monotonic()
    rng = random.Random(cfg["seed"])
    model.train()
    for step in range(steps):
        total = 0.0
        for _ in range(cfg["grad_accum"]):
            batch_rows = [rng.choice(pairs) for _ in range(cfg["batch_size"])]
            batch = encode_batch(
                tokenizer,
                [p for p, _ in batch_rows],
                [c for _, c in batch_rows],
                cfg["max_source"],
                cfg["max_target"],
                device,
            )
            loss = model(**batch, use_cache=False).loss
            (loss / cfg["grad_accum"]).backward()
            total += float(loss.detach().cpu()) / cfg["grad_accum"]
        norm = update(model, optimizer)
        _record(history, start, step, loss=total, grad_norm=norm)
    return history


def train_pretraining(model, tokenizer, corpus, cfg, objective, steps):
    optimizer = optimizer_for(model, cfg["lr"])
    device = next(model.parameters()).device
    rng = random.Random(cfg["seed"])
    history, start = [], time.monotonic()
    sentinels = getattr(tokenizer, "sentinel_ids", None)
    if sentinels is None:
        sentinels = [tokenizer.convert_tokens_to_ids(f"<extra_id_{i}>") for i in range(100)]
        if len(set(sentinels)) != 100:
            raise ValueError("MSP requires the CodeT5 sentinel tokens")
    model.train()
    for step in range(steps):
        tokens = tokenizer.encode(rng.choice(corpus), add_special_tokens=False)
        tokens = tokens[: min(cfg["max_source"], cfg["max_target"]) - 2]
        source, target = ntp_pair(tokens, rng) if objective == "ntp" else msp_pair(tokens, rng, sentinels)
        source, target = source + [tokenizer.eos_token_id], target + [tokenizer.eos_token_id]
        x = torch.tensor([source], device=device)
        loss = model(
            input_ids=x,
            attention_mask=torch.ones_like(x),
            labels=torch.tensor([target], device=device),
            use_cache=False,
        ).loss
        loss.backward()
        norm = update(model, optimizer)
        _record(history, start, step, loss=float(loss.detach().cpu()), grad_norm=norm, objective=objective)
    return history


def train_critic(critic, tokenizer, rows, cfg):
    if not rows:
        raise ValueError("No executed critic training programs")
    optimizer = optimizer_for(critic, cfg["critic_lr"])
    device = next(critic.parameters()).device
    rng = random.Random(cfg["seed"])
    history, start = [], time.monotonic()
    critic.train()
    for step in range(cfg["critic_steps"]):
        batch_rows = [rng.choice(rows) for _ in range(cfg["batch_size"])]
        batch = encode_batch(
            tokenizer,
            [r["prompt"] for r in batch_rows],
            [r["code"] for r in batch_rows],
            cfg["max_source"],
            cfg["max_target"],
            device,
        )
        labels = [outcome_label(r["result"]) for r in batch_rows]
        if critic.classes == 2:
            labels = [int(x == 3) for x in labels]
        target = torch.tensor(labels, device=device)
        logits, _, _ = critic(**batch)
        loss = F.cross_entropy(logits, target)
        loss.backward()
        norm = update(critic, optimizer)
        _record(
            history,
            start,
            step,
            loss=float(loss.detach().cpu()),
            grad_norm=norm,
            accuracy=float(logits.argmax(-1).eq(target).float().mean().detach().cpu()),
        )
    return history


@torch.inference_mode()
def score_rows(critic, tokenizer, rows, cfg):
    critic.eval()
    device = next(critic.parameters()).device
    result = []
    for row in rows:
        batch = encode_batch(
            tokenizer,
            [row["prompt"]],
            [row["code"]],
            cfg["max_source"],
            cfg["max_target"],
            device,
            token_ids=[row["token_ids"]] if "token_ids" in row else None,
        )
        logits, token_logits, mask = critic(**batch)
        label = outcome_label(row["result"])
        if critic.classes == 2:
            label = int(label == 3)
        probs = token_logits.softmax(-1)[0, :, label]
        result.append(
            {
                **row,
                "critic_prediction": int(logits.argmax(-1).cpu()),
                "critic_probabilities": logits.softmax(-1)[0].cpu().tolist(),
                "token_values": probs[mask[0]].cpu().tolist(),
            }
        )
    return result


def train_rl(model, tokenizer, rows, references, cfg, variant="coderl"):
    """Eq. 10/14; preserve sampled token IDs; detach critic; one fixed rollout bank.

    variants: coderl, no_critic (A), absolute (B), distance (C), rl_only,
    ce_synthetic (CE on references plus samples), ce_only (no policy term).
    """
    optimizer = optimizer_for(model, cfg.get("rl_lr", cfg["lr"]))
    device = next(model.parameters()).device
    rng = random.Random(cfg["seed"])
    history, start = [], time.monotonic()
    model.train()
    for step in range(cfg["rl_steps"]):
        ce_value = rl_value = 0.0
        for _ in range(cfg["grad_accum"]):
            row = rng.choice(rows)
            if variant != "rl_only":
                code = (
                    row["code"]
                    if variant == "ce_synthetic" and rng.random() < 0.5
                    else rng.choice(references[row["problem_id"]])
                )
                batch = encode_batch(
                    tokenizer, [row["prompt"]], [code], cfg["max_source"], cfg["max_target"], device
                )
                ce = model(**batch, use_cache=False).loss
                (ce / cfg["grad_accum"]).backward()
                ce_value += float(ce.detach().cpu()) / cfg["grad_accum"]
            if variant not in ("ce_only", "ce_synthetic"):
                batch = encode_batch(
                    tokenizer,
                    [row["prompt"]],
                    [row["code"]],
                    cfg["max_source"],
                    cfg["max_target"],
                    device,
                    token_ids=[row["token_ids"]],
                )
                length = batch["labels"].shape[1]
                advantage = reward(row["result"]) - (
                    0 if variant == "absolute" else reward(row["baseline_result"])
                )
                if variant == "no_critic":
                    weights = torch.ones(length, device=device)
                elif variant == "distance":
                    weights = torch.linspace(1, 0, length, device=device)
                else:
                    weights = torch.tensor(row["token_values"][:length], device=device)
                returns = (weights * advantage).unsqueeze(0)
                logits = model(**batch, use_cache=False).logits
                rl = policy_loss(logits, batch["labels"], returns)
                (rl / cfg["grad_accum"]).backward()
                rl_value += float(rl.detach().cpu()) / cfg["grad_accum"]
        norm = update(model, optimizer)
        _record(
            history,
            start,
            step,
            loss=ce_value + rl_value,
            ce_loss=ce_value,
            rl_loss=rl_value,
            grad_norm=norm,
            variant=variant,
        )
    return history
