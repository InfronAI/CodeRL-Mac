"""Freeze bounded public benchmark subsets with source hashes and IDs."""

import argparse
import json
import urllib.request
from pathlib import Path

from coderl.data import Problem, extract_examples, load_apps, mbpp_records
from coderl.io import digest, read_json, write_json, write_jsonl


def fetch(url, path):
    with urllib.request.urlopen(url, timeout=90) as r:
        path.write_bytes(r.read())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--from-cache", action="store_true", help="Use previously downloaded /private/tmp/coderl-* evidence"
    )
    args = parser.parse_args()
    root = Path("data/benchmarks")
    root.mkdir(parents=True, exist_ok=True)
    urls = {}
    tests = []
    for difficulty, config, count in [
        ("introductory", "introductory", 2),
        ("interview", "all", 2),
        ("competition", "competition", 2),
    ]:
        name = "rows" if config == "all" else config
        path = Path(f"/private/tmp/coderl-apps-{name}.json")
        url = f"https://datasets-server.huggingface.co/rows?dataset=codeparrot%2Fapps&config={config}&split=test&offset=0&length={6 if config == 'all' else count}"
        if not args.from_cache:
            fetch(url, path)
        raw = read_json(path)
        urls[difficulty] = {
            "url": url,
            "raw_sha256": digest(path),
            "selection": "first two rows in the indicated test configuration",
        }
        for entry in raw["rows"][:count]:
            r = entry["row"]
            io = json.loads(r["input_output"])
            pid = f"{r['problem_id']:04d}"
            examples = Path("data/APPS_test_example_tests") / pid / "example_input_output.json"
            example = read_json(examples) if examples.exists() else extract_examples(r["question"])
            tests.append(
                Problem(
                    id=f"apps/test/{pid}",
                    question=r["question"],
                    solutions=json.loads(r["solutions"]) if r["solutions"] else [],
                    tests=io,
                    examples=example,
                    mode="call" if io.get("fn_name") else "stdio",
                    fn_name=io.get("fn_name"),
                    starter=r.get("starter_code") or "",
                    difficulty=r["difficulty"],
                    split="test",
                    dataset="APPS",
                )
            )
    write_jsonl(root / "apps_test.jsonl", [p.asdict() for p in tests])
    train = load_apps("data/APPS/train")
    write_jsonl(root / "apps_train.jsonl", [p.asdict() for p in train])
    mbpp_path = Path("/private/tmp/coderl-mbpp.jsonl")
    commit_path = Path("/private/tmp/coderl-mbpp-commit.json")
    if not args.from_cache:
        fetch("https://api.github.com/repos/google-research/google-research/commits/master", commit_path)
    sha = read_json(commit_path)["sha"]
    url = f"https://raw.githubusercontent.com/google-research/google-research/{sha}/mbpp/mbpp.jsonl"
    pinned = Path("/private/tmp/coderl-mbpp-pinned.jsonl")
    fetch(url, pinned)
    if args.from_cache and digest(mbpp_path) != digest(pinned):
        raise RuntimeError("Moving-branch MBPP differs from pinned source; rerun without --from-cache")
    rows = mbpp_records([json.loads(x) for x in pinned.read_text().splitlines() if x.strip()])
    write_jsonl(root / "mbpp_all.jsonl", [p.asdict() for p in rows])
    subset = [p for p in rows if p.id in {f"mbpp/{i}" for i in [11, 12, 13, 601, 602, 603, 604]}]
    write_jsonl(root / "mbpp_subset.jsonl", [p.asdict() for p in subset])
    write_json(
        root / "provenance.json",
        {
            "APPS_test": urls,
            "APPS_train": {
                "source": "bundled Salesforce/CodeRL repository",
                "commit": "ec3c526",
                "ids": [p.id for p in train],
            },
            "MBPP": {
                "url": url,
                "revision": sha,
                "raw_sha256": digest(pinned),
                "selection": "test IDs 11–13; train IDs 601–604; no accuracy-based selection",
            },
            "normalized_sha256": {p.name: digest(p) for p in root.glob("*.jsonl")},
            "note": "APPS rows service is not revision-addressable; frozen normalized files and raw response digests are the evidence. Test references are never passed to inference.",
        },
    )
    print([(p.id, p.difficulty, len(p.examples.get("inputs", []))) for p in tests])


if __name__ == "__main__":
    main()
