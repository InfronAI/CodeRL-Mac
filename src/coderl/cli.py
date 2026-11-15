import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(prog="coderl", description="CodeRL on Apple Silicon / CPU / CUDA")
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="Check device, dependencies and isolated execution")
    doctor.add_argument("--device", default="auto", choices=["auto", "mps", "cpu", "cuda"])
    doctor.add_argument("--output")
    run = commands.add_parser("run", help="Execute the full stage graph and save all artifacts")
    run.add_argument("--config", default="configs/tiny.json")
    run.add_argument("--output", required=True)
    run.add_argument("--device", choices=["auto", "mps", "cpu", "cuda"])
    run.add_argument("--seed", type=int)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--stop-after", help="Checkpoint after a named stage; resume continues the graph")
    report = commands.add_parser("report", help="Rebuild measured analysis and charts")
    report.add_argument("output")
    gen = commands.add_parser("generate", help="Generate programs from a saved actor")
    gen.add_argument("--model", required=True)
    gen.add_argument("--prompt", required=True, help="UTF-8 file containing the problem prompt")
    gen.add_argument("--output", required=True)
    gen.add_argument("--samples", type=int, default=5)
    gen.add_argument("--problem-id", default="custom", help="ID matching the evaluation dataset")
    gen.add_argument("--max-target", type=int, default=512)
    gen.add_argument("--device", default="auto")
    gen.add_argument("--seed", type=int, default=42)
    evaluate = commands.add_parser("evaluate", help="Execute saved candidate JSONL and compute pass@k")
    evaluate.add_argument("--data", required=True)
    evaluate.add_argument("--candidates", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--split", default="test")
    args = parser.parse_args()
    if args.command == "doctor":
        import torch

        from .data import PublicProblem
        from .execution import Runner
        from .io import write_json
        from .runtime import device_for, environment

        device = device_for(args.device)
        x = torch.tensor([2.0], device=device, requires_grad=True)
        x.square().sum().backward()
        info = environment(device)
        info["backward_test"] = float(x.grad.cpu()) == 4.0
        info["executor_test"] = Runner().run(
            "print(int(input()) + 1)",
            PublicProblem("doctor", "add one"),
            {"inputs": ["2\n"], "outputs": ["3\n"]},
        )
        print(json.dumps(info, indent=2))
        if args.output:
            write_json(args.output, info)
        if not info["backward_test"] or info["executor_test"]["outcome"] != "passed":
            raise SystemExit(1)
    elif args.command == "run":
        from .pipeline import Experiment, config_from
        from .reporting import build_report

        cfg = config_from(args.config)
        for key in ["device", "seed"]:
            if getattr(args, key) is not None:
                cfg[key] = getattr(args, key)
        experiment = Experiment(cfg, args.output, args.resume)
        experiment.run(args.stop_after)
        if not args.stop_after:
            print(build_report(args.output))
    elif args.command == "report":
        from .reporting import build_report

        print(build_report(args.output))
    elif args.command == "generate":
        from .io import write_jsonl
        from .models import generate, load_actor
        from .runtime import device_for, seed_everything

        seed_everything(args.seed)
        actor, tokenizer = load_actor(args.model)
        actor.to(device_for(args.device))
        rows = generate(
            actor, tokenizer, Path(args.prompt).read_text(), args.samples, max_target=args.max_target
        )
        rows = [{"problem_id": args.problem_id, **row} for row in rows]
        write_jsonl(args.output, rows)
    else:
        from .data import load_problems
        from .execution import Runner, outcome_label
        from .io import read_jsonl, write_json, write_jsonl
        from .metrics import summarize

        problems = {p.id: p for p in load_problems(args.data, args.split)}
        rows = read_jsonl(args.candidates)
        runner = Runner()
        for row in rows:
            problem = problems[row["problem_id"]]
            row["result"] = runner.run(row["code"], problem.public(), problem.tests)
            outcome_label(row["result"])
        output = Path(args.output)
        write_jsonl(output / "evaluated.jsonl", rows)
        write_json(output / "metrics.json", summarize(rows))
