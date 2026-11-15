"""Auditable end-to-end CodeRL experiment stages with immutable run identity."""

import hashlib
import json
import os
import time
import zipfile
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .data import load_problems
from .execution import Runner, outcome_label, reward
from .io import digest, read_json, read_jsonl, write_json, write_jsonl
from .metrics import filtered_n_at_k, summarize
from .models import Critic, generate, load_actor
from .runtime import device_for, environment, release_memory, seed_everything
from .sampling import critic_sampling, repair_prompt, sampling_kwargs
from .training import score_rows, train_ce, train_critic, train_pretraining, train_rl

DEFAULTS = {
    "model": "tiny",
    "revision": None,
    "device": "auto",
    "seed": 42,
    "threads": 4,
    "batch_size": 1,
    "grad_accum": 1,
    "lr": 0.001,
    "critic_lr": 0.001,
    "rl_lr": 0.0001,
    "max_source": 384,
    "max_target": 128,
    "generation_batch": 2,
    "samples": 5,
    "train_samples": 4,
    "temperature": 0.6,
    "top_p": 0.95,
    "msp_steps": 3,
    "ntp_steps": 3,
    "sft_steps": 100,
    "critic_steps": 30,
    "repair_steps": 30,
    "rl_steps": 20,
    "variants": ["coderl", "no_critic", "absolute", "distance", "ce_only", "ce_synthetic", "rl_only"],
    "cs_modes": ["refine", "repair1", "repair2", "repair4"],
    "test_timeout": 1.0,
    "program_timeout": 10.0,
    "mbpp_steps": 10,
    "mbpp_samples": 5,
    "max_solutions": 4,
}


def config_from(path):
    cfg = {**DEFAULTS, **read_json(path)}
    unknown = set(cfg) - set(DEFAULTS) - {"train_data", "test_data", "mbpp_data", "name", "scope"}
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    for key in [
        "batch_size",
        "grad_accum",
        "max_source",
        "max_target",
        "generation_batch",
        "samples",
        "train_samples",
    ]:
        if cfg[key] < 1:
            raise ValueError(f"{key} must be positive")
    return cfg


class Experiment:
    def __init__(self, cfg, output, resume=False):
        self.cfg = cfg
        self.output = Path(output)
        self.device = device_for(cfg["device"])
        self.runner = Runner(cfg["test_timeout"], cfg["program_timeout"])
        self.output.mkdir(parents=True, exist_ok=True)
        identity = {
            "config": cfg,
            "source": self.source_hash(),
            "data": {
                k: self.data_hash(cfg[k]) for k in ("train_data", "test_data", "mbpp_data") if cfg.get(k)
            },
        }
        manifest_path = self.output / "manifest.json"
        if manifest_path.exists():
            old = read_json(manifest_path)
            if not resume:
                raise FileExistsError(
                    f"Run already exists: {output}. Use --resume or a new output directory."
                )
            if old["identity"] != identity:
                raise ValueError(
                    "Run identity changed (source, data, or config). Use a new output directory."
                )
            self.manifest = old
        else:
            self.manifest = {
                "identity": identity,
                "environment": environment(self.device),
                "stages": {},
                "started_at": datetime.now(timezone.utc).isoformat(),
                "status": "running",
            }
            write_json(manifest_path, self.manifest)
            with zipfile.ZipFile(self.output / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
                for source in sorted(Path(__file__).parent.glob("*.py")):
                    archive.write(source, f"coderl/{source.name}")
        self.train = load_problems(cfg["train_data"], "train")
        self.test = load_problems(cfg["test_data"], "test")
        if {p.id for p in self.train} & {p.id for p in self.test}:
            raise ValueError("Train/test problem ID overlap")
        if {p.question for p in self.train} & {p.question for p in self.test}:
            raise ValueError("Train/test question overlap")
        seed_everything(cfg["seed"], cfg["threads"])

    @staticmethod
    def source_hash():
        paths = sorted(Path(__file__).parent.glob("*.py"))
        return hashlib.sha256("\n".join(p.name + ":" + digest(p) for p in paths).encode()).hexdigest()

    @staticmethod
    def data_hash(path):
        path = Path(path)
        if path.is_file():
            return digest(path)
        return hashlib.sha256(
            "\n".join(
                str(p.relative_to(path)) + ":" + digest(p) for p in sorted(path.rglob("*")) if p.is_file()
            ).encode()
        ).hexdigest()

    def log(self, event, **data):
        row = {"time": datetime.now(timezone.utc).isoformat(), "event": event, **data}
        print(json.dumps(row), flush=True)
        with (self.output / "events.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")

    def checkpoint(self, stage):
        return self.output / "checkpoints" / stage

    def stage(self, name, function):
        prior = self.manifest["stages"].get(name)
        if prior and prior["status"] == "complete":
            for relative, expected in prior["artifacts"].items():
                path = self.output / relative
                if not path.exists() or digest(path) != expected:
                    raise ValueError(f"Stage {name} artifact missing/changed: {relative}")
            self.log("stage_cached", stage=name)
            return
        seed_everything(self.cfg["seed"], self.cfg["threads"])
        start = time.monotonic()
        self.log("stage_start", stage=name)
        try:
            metadata, paths = function()
            files = []
            for path in paths:
                path = Path(path)
                files.extend(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else files.append(
                    path
                )
            artifacts = {str(p.relative_to(self.output)): digest(p) for p in files}
            self.manifest["stages"][name] = {
                "status": "complete",
                "seconds": time.monotonic() - start,
                "artifacts": artifacts,
                **metadata,
            }
            self.log("stage_complete", stage=name, seconds=time.monotonic() - start, **metadata)
        except BaseException as error:
            self.manifest["stages"][name] = {
                "status": "failed",
                "error": repr(error),
                "seconds": time.monotonic() - start,
            }
            self.manifest["status"] = "failed"
            self.log("stage_failed", stage=name, error=repr(error))
            raise
        finally:
            write_json(self.output / "manifest.json", self.manifest)
            release_memory()

    def actor(self, stage=None):
        model, tokenizer = load_actor(
            str(self.checkpoint(stage)) if stage else self.cfg["model"],
            None if stage else self.cfg.get("revision"),
        )
        model.to(self.device)
        return model, tokenizer

    def save_actor(self, stage, model, tokenizer, history):
        model.save_pretrained(self.checkpoint(stage))
        tokenizer.save_pretrained(self.checkpoint(stage))
        log = self.output / "training" / f"{stage}.jsonl"
        write_jsonl(log, history)
        return {
            "steps": len(history),
            "parameters": sum(p.numel() for p in model.parameters()),
            "last_loss": history[-1]["loss"] if history else None,
        }, [self.checkpoint(stage), log]

    def pretrain(self, stage, previous=None):
        model, tokenizer = self.actor(previous)
        corpus = [code for p in self.train for code in p.solutions]
        history = train_pretraining(model, tokenizer, corpus, self.cfg, stage, self.cfg[f"{stage}_steps"])
        return self.save_actor(stage, model, tokenizer, history)

    def sft(self):
        model, tokenizer = self.actor("ntp")
        pairs = [
            (p.public().prompt(), code)
            for p in self.train
            for code in p.solutions[: self.cfg["max_solutions"]]
        ]
        history = train_ce(model, tokenizer, pairs, self.cfg)
        return self.save_actor("sft", model, tokenizer, history)

    def collect_rollouts(self):
        actor, tokenizer = self.actor("sft")
        rows, baselines, references = [], [], []
        for p in self.train:
            public = p.public()
            baseline = generate(
                actor, tokenizer, public.prompt(), 1, mode="greedy", **sampling_kwargs(self.cfg)
            )[0]
            baseline_result = self.runner.run(baseline["code"], public, p.tests)
            baselines.append({"problem_id": p.id, **baseline, "result": baseline_result})
            for candidate in generate(
                actor, tokenizer, public.prompt(), self.cfg["train_samples"], **sampling_kwargs(self.cfg)
            ):
                result = self.runner.run(candidate["code"], public, p.tests)
                outcome_label(result)
                rows.append(
                    {
                        "problem_id": p.id,
                        "prompt": public.prompt(),
                        **candidate,
                        "result": result,
                        "baseline_result": baseline_result,
                        "origin": "actor_sample",
                    }
                )
            for code in p.solutions[: self.cfg["max_solutions"]]:
                result = self.runner.run(code, public, p.tests)
                references.append(
                    {
                        "problem_id": p.id,
                        "prompt": public.prompt(),
                        "code": code,
                        "result": result,
                        "origin": "reference",
                    }
                )
            self.log("rollout_problem", problem=p.id, samples=self.cfg["train_samples"])
        paths = [
            self.output / "rollouts.jsonl",
            self.output / "baselines.jsonl",
            self.output / "references.jsonl",
        ]
        for path, contents in zip(paths, [rows, baselines, references]):
            write_jsonl(path, contents)
        return {
            "samples": len(rows),
            "outcomes": dict(Counter(r["result"]["outcome"] for r in rows)),
            "nonzero_advantages": sum(reward(r["result"]) != reward(r["baseline_result"]) for r in rows),
            "references_passed": sum(r["result"]["outcome"] == "passed" for r in references),
        }, paths

    def critic_training(self, classes):
        stage = f"critic{classes}"
        actor, tokenizer = self.actor("sft")
        critic = Critic(actor, classes).to(self.device)
        rows = read_jsonl(self.output / "rollouts.jsonl") + read_jsonl(self.output / "references.jsonl")
        history = train_critic(critic, tokenizer, rows, self.cfg)
        critic.save_pretrained(self.checkpoint(stage), tokenizer)
        log = self.output / "training" / f"{stage}.jsonl"
        write_jsonl(log, history)
        return {"steps": len(history), "training_examples": len(rows), "classes": classes}, [
            log,
            self.checkpoint(stage),
        ]

    def score(self):
        critic, tokenizer = Critic.from_pretrained(self.checkpoint("critic4"))
        critic.to(self.device)
        rows = score_rows(critic, tokenizer, read_jsonl(self.output / "rollouts.jsonl"), self.cfg)
        path = self.output / "critic_scores.jsonl"
        write_jsonl(path, rows)
        return {
            "scored": len(rows),
            "training_accuracy": sum(r["critic_prediction"] == outcome_label(r["result"]) for r in rows)
            / len(rows),
        }, [path]

    def repair_training(self):
        actor, tokenizer = self.actor("sft")
        problems = {p.id: p for p in self.train}
        pairs = []
        for row in read_jsonl(self.output / "rollouts.jsonl"):
            if row["result"]["outcome"] != "passed":
                p = problems[row["problem_id"]]
                pairs.append((repair_prompt(p.public(), row["code"], row["result"]), p.solutions[0]))
        if not pairs:
            raise ValueError(
                "No failed rollouts to train repair model; increase train_samples or temperature"
            )
        history = train_ce(actor, tokenizer, pairs, self.cfg, self.cfg["repair_steps"])
        return self.save_actor("repair", actor, tokenizer, history)

    def rl_training(self, variant):
        model, tokenizer = self.actor("sft")
        rows = read_jsonl(self.output / "critic_scores.jsonl")
        history = train_rl(model, tokenizer, rows, {p.id: p.solutions for p in self.train}, self.cfg, variant)
        return self.save_actor(variant, model, tokenizer, history)

    def evaluate(self, variant, problems=None, tag=None, cs_mode=None, beam=False, cfg=None):
        cfg = cfg or self.cfg
        tag = tag or variant
        actor, tokenizer = self.actor(variant)
        critic = repair = None
        if cs_mode:
            critic, _ = Critic.from_pretrained(self.checkpoint("critic2"))
            critic.to(self.device)
            if cs_mode != "refine":
                repair, _ = self.actor("repair")
        rows, traces = [], []
        for p in problems if problems is not None else self.test:
            public = p.public()
            mode = "beam" if beam else "sample"
            candidates = generate(
                actor, tokenizer, public.prompt(), cfg["samples"], mode=mode, **sampling_kwargs(cfg)
            )
            if cs_mode:
                candidates, trace = critic_sampling(
                    actor,
                    critic,
                    repair,
                    tokenizer,
                    public,
                    self.runner,
                    cfg,
                    initial=candidates,
                    repair_top_m=0 if cs_mode == "refine" else int(cs_mode[-1]),
                )
                traces.append({"problem_id": p.id, "trace": trace})
            for i, candidate in enumerate(candidates):
                result = self.runner.run(candidate["code"], public, p.tests)
                outcome_label(result)
                example = candidate.get("example_result") or self.runner.run(
                    candidate["code"], public, p.examples
                )
                rows.append(
                    {
                        "problem_id": p.id,
                        "difficulty": p.difficulty,
                        "dataset": p.dataset,
                        "sample": i,
                        **candidate,
                        "example_result": example,
                        "result": result,
                    }
                )
            self.log(
                "evaluation_problem",
                variant=tag,
                problem=p.id,
                passed=sum(r["result"]["outcome"] == "passed" for r in rows if r["problem_id"] == p.id),
            )
        metrics = summarize(
            rows, ks=sorted(set([1, min(5, cfg["samples"]), cfg["samples"]])), seed=cfg["seed"]
        )
        if beam:
            # Beam outputs are ranked, not IID samples. Report top-k success, not the estimator.
            from collections import defaultdict

            groups = defaultdict(list)
            for row in rows:
                groups[row["problem_id"]].append(row)
            metrics = {k: v for k, v in metrics.items() if not k.startswith("pass@") and k != "by_difficulty"}
            for k in sorted(set([1, min(5, cfg["samples"])])):
                metrics[f"beam_pass@{k}"] = {
                    "mean": sum(
                        any(r["result"]["outcome"] == "passed" for r in group[:k])
                        for group in groups.values()
                    )
                    / len(groups)
                }
        else:
            for n in [1, 5]:
                metrics[f"{n}@{cfg['samples']}"] = filtered_n_at_k(rows, n)
        metrics["decoding"] = (
            "beam" if beam else "nucleus" + (" + critic sampling (dependent samples)" if cs_mode else "")
        )
        metrics["scope"] = cfg.get("scope", "local reduced-scale experiment")
        paths = [self.output / "evaluations" / f"{tag}.jsonl", self.output / "metrics" / f"{tag}.json"]
        write_jsonl(paths[0], rows)
        write_json(paths[1], metrics)
        if traces:
            paths.append(self.output / "traces" / f"{tag}.json")
            write_json(paths[-1], traces)
        return {
            "problems": metrics["problems"],
            "candidates": metrics["candidates"],
            "outcomes": metrics["outcomes"],
        }, paths

    def mbpp_finetune(self, train):
        model, tokenizer = self.actor()  # separate pretrained checkpoint, no APPS/MBPP test training
        pairs = [(p.public().prompt(), c) for p in train for c in p.solutions]
        history = train_ce(model, tokenizer, pairs, self.cfg, self.cfg["mbpp_steps"])
        return self.save_actor("mbpp_finetuned", model, tokenizer, history)

    @contextmanager
    def lock(self):
        path = self.output / ".lock"
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            raise RuntimeError(
                f"Run is locked: {path}. Inspect the recorded PID before removing a stale lock."
            ) from None
        try:
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            yield
        finally:
            path.unlink()

    def run(self, stop_after=None):
        with self.lock():
            steps = [
                ("msp", lambda: self.pretrain("msp")),
                ("ntp", lambda: self.pretrain("ntp", "msp")),
                ("sft", self.sft),
                ("rollouts", self.collect_rollouts),
                ("critic4", lambda: self.critic_training(4)),
                ("critic2", lambda: self.critic_training(2)),
                ("scores", self.score),
                ("repair", self.repair_training),
                ("eval_sft", lambda: self.evaluate("sft")),
                ("beam_sft", lambda: self.evaluate("sft", tag="sft_beam", beam=True)),
            ]
            for variant in self.cfg["variants"]:
                steps += [
                    (variant, lambda v=variant: self.rl_training(v)),
                    (f"beam_{variant}", lambda v=variant: self.evaluate(v, tag=f"{v}_beam", beam=True)),
                ]
            if "coderl" in self.cfg["variants"]:
                steps.append(("eval_coderl", lambda: self.evaluate("coderl")))
                for cs in self.cfg["cs_modes"]:
                    steps.append((f"cs_{cs}", lambda c=cs: self.evaluate("coderl", tag=f"cs_{c}", cs_mode=c)))
            if self.cfg.get("mbpp_data"):
                train = load_problems(self.cfg["mbpp_data"], "train")
                test = load_problems(self.cfg["mbpp_data"], "test")
                mbpp_cfg = {**self.cfg, "samples": self.cfg["mbpp_samples"], "temperature": 1.2}
                for variant in ["sft", "coderl"]:
                    if variant == "sft" or variant in self.cfg["variants"]:
                        steps.append(
                            (
                                f"mbpp_zero_{variant}",
                                lambda v=variant: self.evaluate(
                                    v, problems=test, tag=f"mbpp_zero_{v}", cfg=mbpp_cfg
                                ),
                            )
                        )
                steps.extend(
                    [
                        ("mbpp_finetune", lambda: self.mbpp_finetune(train)),
                        (
                            "mbpp_eval_finetuned",
                            lambda: self.evaluate(
                                "mbpp_finetuned", problems=test, tag="mbpp_finetuned", cfg=mbpp_cfg
                            ),
                        ),
                    ]
                )
            if stop_after and stop_after not in {s[0] for s in steps}:
                raise ValueError(f"Unknown stage {stop_after}; choices: {[s[0] for s in steps]}")
            for name, fn in steps:
                self.stage(name, fn)
                if name == stop_after:
                    self.manifest["status"] = "stopped_at_requested_stage"
                    break
            else:
                self.manifest["status"] = "complete"
            write_json(self.output / "manifest.json", self.manifest)
            self.log("run_finished", status=self.manifest["status"])
