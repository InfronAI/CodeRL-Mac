"""CodeT5 actors and paper-style critics without patching Transformers."""

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn
from transformers import AutoTokenizer, T5Config, T5ForConditionalGeneration


class ByteTokenizer:
    """Offline byte tokenizer for mechanism tests, NOT the pretrained CodeT5 tokenizer."""

    pad_token_id = 0
    eos_token_id = 1
    vocab_size = 292
    sentinel_ids = list(range(259, 292))

    def encode(self, text, add_special_tokens=True):
        ids = [b + 3 for b in text.encode("utf-8")]
        return ids + [self.eos_token_id] if add_special_tokens else ids

    def decode(self, ids, skip_special_tokens=True, **kwargs):
        return bytes(i - 3 for i in ids if 3 <= i <= 258).decode("utf-8", errors="replace")

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "byte_tokenizer.json").write_text('{"version":1}\n')


def load_actor(name, revision=None):
    if name == "tiny":
        tokenizer = ByteTokenizer()
        config = T5Config(
            vocab_size=tokenizer.vocab_size,
            d_model=128,
            d_ff=256,
            d_kv=32,
            num_layers=2,
            num_decoder_layers=2,
            num_heads=4,
            dropout_rate=0.0,
            pad_token_id=0,
            eos_token_id=1,
            decoder_start_token_id=0,
        )
        return T5ForConditionalGeneration(config), tokenizer
    kwargs = {"revision": revision} if revision else {}
    tokenizer = (
        ByteTokenizer()
        if (Path(name) / "byte_tokenizer.json").exists()
        else AutoTokenizer.from_pretrained(name, **kwargs)
    )
    model = T5ForConditionalGeneration.from_pretrained(name, **kwargs)
    return model, tokenizer


def encode_batch(
    tokenizer, prompts, codes=None, max_source=600, max_target=512, device="cpu", token_ids=None
):
    sources = []
    for prompt in prompts:
        if "\nBUGGY PROGRAM:\n" in prompt and "\nTEST OUTCOME:\n" in prompt:
            # Reserve room for all repair signals even when APPS descriptions are long.
            description, rest = prompt.rsplit("\nBUGGY PROGRAM:\n", 1)
            buggy, feedback = rest.rsplit("\nTEST OUTCOME:\n", 1)
            feedback_ids = tokenizer.encode("\nTEST OUTCOME:\n" + feedback, add_special_tokens=False)
            feedback_ids = feedback_ids[: max(16, max_source // 6)]
            buggy_ids = tokenizer.encode("\nBUGGY PROGRAM:\n" + buggy, add_special_tokens=False)
            buggy_ids = buggy_ids[: max_source // 3]
            room = max_source - len(feedback_ids) - len(buggy_ids) - 1
            ids = tokenizer.encode(description, add_special_tokens=False)[: max(0, room)]
            ids += buggy_ids + feedback_ids + [tokenizer.eos_token_id]
            sources.append(ids[:max_source])
        else:
            sources.append(tokenizer.encode(prompt)[:max_source])
    width = max(map(len, sources))
    inputs = torch.full((len(sources), width), tokenizer.pad_token_id, dtype=torch.long)
    mask = torch.zeros_like(inputs)
    for i, ids in enumerate(sources):
        inputs[i, : len(ids)] = torch.tensor(ids)
        mask[i, : len(ids)] = 1
    batch = {"input_ids": inputs.to(device), "attention_mask": mask.to(device)}
    if codes is not None or token_ids is not None:
        targets = token_ids if token_ids is not None else [tokenizer.encode(c) for c in codes]
        targets = [t[:max_target] for t in targets]
        if any(not t for t in targets):
            raise ValueError("Empty target token sequence")
        labels = torch.full((len(targets), max(map(len, targets))), -100, dtype=torch.long)
        for i, ids in enumerate(targets):
            labels[i, : len(ids)] = torch.tensor(ids)
        batch["labels"] = labels.to(device)
    return batch


class Critic(nn.Module):
    def __init__(self, backbone, classes=4, pooling="paper"):
        super().__init__()
        if pooling not in ("paper", "legacy_logits"):
            raise ValueError(pooling)
        self.backbone = backbone
        self.classes = classes
        self.pooling = pooling
        d = backbone.config.d_model
        self.head = (
            nn.Linear(d, classes)
            if pooling == "paper"
            else nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Linear(128, classes))
        )

    def forward(self, input_ids, attention_mask, labels):
        mask = labels.ne(-100)
        encoder = self.backbone.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        decoder = self.backbone.decoder(
            input_ids=self.backbone._shift_right(labels),
            attention_mask=mask.long(),
            encoder_hidden_states=encoder.last_hidden_state,
            encoder_attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        hidden = decoder.last_hidden_state
        if self.pooling == "legacy_logits":
            hidden = hidden * self.backbone.config.d_model**-0.5
        token_logits = self.head(hidden)
        if self.pooling == "paper":
            pooled = hidden.masked_fill(~mask.unsqueeze(-1), torch.finfo(hidden.dtype).min).amax(dim=1)
            logits = self.head(pooled)
        else:
            logits = token_logits.masked_fill(~mask.unsqueeze(-1), torch.finfo(hidden.dtype).min).amax(dim=1)
        return logits, token_logits, mask

    def save_pretrained(self, path, tokenizer):
        path = Path(path)
        self.backbone.save_pretrained(path)
        tokenizer.save_pretrained(path)
        save_file(
            {k: v.detach().cpu().contiguous() for k, v in self.head.state_dict().items()},
            str(path / "critic_head.safetensors"),
        )
        (path / "critic.json").write_text(json.dumps({"classes": self.classes, "pooling": self.pooling}))

    @classmethod
    def from_pretrained(cls, path):
        info = json.loads((Path(path) / "critic.json").read_text())
        actor, tokenizer = load_actor(str(path))
        critic = cls(actor, **info)
        critic.head.load_state_dict(load_file(str(Path(path) / "critic_head.safetensors")))
        return critic, tokenizer


@torch.inference_mode()
def generate(
    model,
    tokenizer,
    prompt,
    count=1,
    batch_size=1,
    max_source=600,
    max_target=512,
    temperature=0.6,
    top_p=0.95,
    mode="sample",
    prefix=None,
):
    if count < 1 or batch_size < 1 or temperature <= 0 or not 0 < top_p <= 1:
        raise ValueError("Invalid generation parameters")
    model.eval()
    device = next(model.parameters()).device
    inputs = encode_batch(tokenizer, [prompt], max_source=max_source, device=device)
    result = []
    # A beam call must rank all count beams together; sampling is microbatched.
    chunk_size = count if mode == "beam" else batch_size
    while len(result) < count:
        n = min(chunk_size, count - len(result))
        kwargs = {"max_length": max_target + 1, "num_return_sequences": n, "use_cache": True}
        if mode == "sample":
            kwargs.update(do_sample=True, temperature=temperature, top_p=top_p, top_k=0)
        else:
            kwargs.update(do_sample=False, num_beams=n)
        if prefix:
            prefix = prefix[: max_target - 1]
            kwargs["decoder_input_ids"] = torch.tensor(
                [[model.config.decoder_start_token_id] + prefix], device=device
            )
        output = model.generate(**inputs, **kwargs)
        for seq in output.cpu().tolist():
            ids = seq[1:]
            if tokenizer.eos_token_id in ids:
                ids = ids[: ids.index(tokenizer.eos_token_id) + 1]
            result.append(
                {
                    "code": tokenizer.decode(
                        ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    ),
                    "token_ids": ids,
                }
            )
    return result


def policy_loss(logits, labels, returns):
    mask = labels.ne(-100)
    targets = labels.masked_fill(~mask, 0)
    logp = logits.log_softmax(-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return -(logp * returns.detach() * mask).sum() / mask.sum().clamp_min(1)
