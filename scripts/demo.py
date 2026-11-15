"""Live inference from a real saved checkpoint; no reference solutions are used."""

import argparse
from pathlib import Path

from coderl.data import load_problems
from coderl.execution import Runner
from coderl.io import read_json, write_json
from coderl.models import Critic, generate, load_actor
from coderl.runtime import device_for, seed_everything
from coderl.sampling import critic_sampling, sampling_kwargs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="experiments/tiny-mps-final")
    parser.add_argument("--output", default="docs/assets/demo-session.json")
    args = parser.parse_args()
    root = Path(args.run)
    manifest = read_json(root / "manifest.json")
    cfg = manifest["identity"]["config"]
    cfg["samples"] = 5
    cfg["generation_batch"] = 5
    seed_everything(cfg["seed"])
    device = device_for(cfg["device"])
    print("CodeRL | live local inference", flush=True)
    print(f"Device: {device} | model: {cfg['model']} | seed: {cfg['seed']}", flush=True)
    print("Scope: " + cfg["scope"], flush=True)
    problem = load_problems(cfg["test_data"], "test")[0]
    print("\nPROBLEM: " + problem.question.strip(), flush=True)
    actor, tok = load_actor(str(root / "checkpoints/coderl"))
    actor.to(device)
    critic, _ = Critic.from_pretrained(root / "checkpoints/critic2")
    critic.to(device)
    repair, _ = load_actor(str(root / "checkpoints/repair"))
    repair.to(device)
    runner = Runner()
    print("\nGenerating a greedy program from the trained actor ...", flush=True)
    candidate = generate(actor, tok, problem.public().prompt(), 1, mode="greedy", **sampling_kwargs(cfg))[0]
    print(candidate["code"], flush=True)
    result = runner.run(candidate["code"], problem.public(), problem.tests)
    print(f"Final tests: {result['outcome']} ({result['passed']}/{result['total']})", flush=True)
    print("\nSampling five candidates and running the learned critic/repair pipeline ...", flush=True)
    candidates, trace = critic_sampling(actor, critic, repair, tok, problem.public(), runner, cfg)
    for event in trace:
        if "population" in event:
            print(
                f"  {event['stage']}: {event['passed_examples']}/{len(event['population'])} pass public examples",
                flush=True,
            )
        elif event["stage"] == "prefix":
            print(
                f"  prefix: {len(event['prefix_token_ids'])} tokens, {event['copies']} continuation(s)",
                flush=True,
            )
    final = []
    for i, c in enumerate(candidates):
        tested = runner.run(c["code"], problem.public(), problem.tests)
        final.append({**c, "result": tested})
        print(f"Candidate {i + 1}: {tested['outcome']} | {c['code'][:80]!r}", flush=True)
    print("\nAll candidate programs, test errors and critic traces are saved.", flush=True)
    print("The tiny addition fixture demonstrates the workflow; it is not an APPS score.", flush=True)
    write_json(
        args.output,
        {
            "run": str(root),
            "model": cfg["model"],
            "scope": cfg["scope"],
            "device": str(device),
            "problem_id": problem.id,
            "greedy": {**candidate, "result": result},
            "cs_trace": trace,
            "candidates": final,
        },
    )
    print("Session artifact: " + args.output, flush=True)


if __name__ == "__main__":
    main()
