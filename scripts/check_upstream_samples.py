"""Re-execute upstream saved programs; distinguish them from newly sampled programs."""

from collections import Counter
from pathlib import Path

from coderl.data import load_apps
from coderl.execution import Runner
from coderl.io import read_json, write_json, write_jsonl

runner = Runner(timeout=2, program_timeout=20)
rows = []
for problem in load_apps("data/APPS/train"):
    folder = Path("data/APPS/train") / problem.id.rsplit("/", 1)[-1]
    for index, sample in enumerate(read_json(folder / "gen_solutions.json")):
        original = sample["result"]
        if original is True:
            label = "passed"
        elif original is False:
            label = "failed"
        else:
            label = {-2: "compile_error", -1: "runtime_error"}[original]
        result = runner.run(sample["code"], problem.public(), problem.tests)
        rows.append(
            {
                "problem_id": problem.id,
                "index": index,
                "code": sample["code"],
                "origin": "upstream saved sample; not newly generated",
                "upstream_outcome": label,
                "result": result,
            }
        )
    print(problem.id, "complete", flush=True)
write_jsonl("experiments/upstream-reexecution/programs.jsonl", rows)
write_json(
    "experiments/upstream-reexecution/summary.json",
    {
        "scope": "Re-execution of saved Salesforce/CodeRL samples; not new model inference and not held-out accuracy",
        "programs": len(rows),
        "outcomes": dict(Counter(r["result"]["outcome"] for r in rows)),
        "agreement": sum(r["upstream_outcome"] == r["result"]["outcome"] for r in rows) / len(rows),
        "disagreements": [
            {
                "problem_id": r["problem_id"],
                "index": r["index"],
                "upstream": r["upstream_outcome"],
                "current": r["result"]["outcome"],
                "error": r["result"]["error"],
            }
            for r in rows
            if r["upstream_outcome"] != r["result"]["outcome"]
        ],
    },
)
