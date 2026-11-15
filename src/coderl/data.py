"""APPS, MBPP and portable records; inference receives no references/hidden tests."""

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .io import read_json, read_jsonl


@dataclass(frozen=True)
class PublicProblem:
    id: str
    question: str
    mode: str = "stdio"
    starter: str = ""
    fn_name: str | None = None
    examples: dict = field(default_factory=dict)
    setup: str = ""

    def prompt(self):
        if self.mode == "assert":
            tests = "\n".join(self.examples.get("assertions", []))
            return self.question + "\nYour code should satisfy these tests:\n" + tests + "\n"
        style = "Call-Based" if self.mode == "call" else "Standard Input"
        return f"\nQUESTION:\n{self.question}\n{self.starter}\nUse {style} format\nANSWER:\n"


@dataclass
class Problem:
    id: str
    question: str
    solutions: list[str]
    tests: dict
    examples: dict = field(default_factory=dict)
    mode: str = "stdio"
    fn_name: str | None = None
    starter: str = ""
    difficulty: str = "unknown"
    split: str = "train"
    dataset: str = "fixture"
    setup: str = ""

    def public(self):
        return PublicProblem(
            self.id, self.question, self.mode, self.starter, self.fn_name, self.examples, self.setup
        )

    def asdict(self):
        return asdict(self)


def extract_examples(question):
    """Conservative APPS stdin examples; never borrow hidden input/output pairs."""
    pairs = re.findall(
        r"(?:^|\n)[ \t]*(?:Sample\s+)?Input(?:[ \t]*\d+)?[ \t]*:?[ \t]*\n(.*?)"
        r"\n[ \t]*(?:Sample\s+)?Output(?:[ \t]*\d+)?[ \t]*:?[ \t]*\n(.*?)"
        r"(?=\n\s*(?:[-]{2,}|(?:Sample\s+)?Input|Note\b|Explanation\b)|\Z)",
        question,
        re.IGNORECASE | re.DOTALL,
    )
    return {"inputs": [a.strip() + "\n" for a, _ in pairs], "outputs": [b.strip() + "\n" for _, b in pairs]}


def load_apps(root, split="train", limit=None, examples_root=None):
    result = []
    for folder in sorted(Path(root).iterdir()):
        if not folder.is_dir() or not (folder / "question.txt").exists():
            continue
        question = (folder / "question.txt").read_text()
        tests = read_json(folder / "input_output.json")
        fn = tests.get("fn_name")
        examples_file = folder / "example_input_output.json"
        if not examples_file.exists() and examples_root:
            examples_file = Path(examples_root) / folder.name / "example_input_output.json"
        examples = (
            read_json(examples_file)
            if examples_file.exists()
            else extract_examples(question)
            if not fn
            else {}
        )
        meta = read_json(folder / "metadata.json") if (folder / "metadata.json").exists() else {}
        result.append(
            Problem(
                id=f"apps/{split}/{folder.name}",
                question=question,
                solutions=read_json(folder / "solutions.json")
                if (folder / "solutions.json").exists()
                else [],
                tests=tests,
                examples=examples,
                mode="call" if fn else "stdio",
                fn_name=fn,
                starter=(folder / "starter_code.py").read_text()
                if (folder / "starter_code.py").exists()
                else "",
                difficulty=meta.get("difficulty", "unknown"),
                split=split,
                dataset="APPS",
            )
        )
        if limit is not None and len(result) >= limit:
            break
    if not result:
        raise ValueError(f"No APPS problems found at {root}")
    return result


def load_problems(path, split=None, limit=None):
    path = Path(path)
    if path.is_dir():
        return load_apps(path, split or "train", limit)
    rows = read_jsonl(path) if path.suffix == ".jsonl" else read_json(path)
    result = [Problem(**row) for row in rows if split is None or row.get("split") == split]
    if not result:
        raise ValueError(f"No problems for split {split!r} in {path}")
    ids = [p.id for p in result]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate problem IDs")
    return result[:limit] if limit is not None else result


def mbpp_records(rows):
    result = []
    for r in rows:
        i = r["task_id"]
        split = "prompt" if i <= 10 else "test" if i <= 510 else "validation" if i <= 600 else "train"
        tests = {"assertions": r["test_list"]}
        result.append(
            Problem(
                id=f"mbpp/{i}",
                question=r["text"],
                solutions=[r["code"]],
                tests=tests,
                examples=tests,
                mode="assert",
                split=split,
                dataset="MBPP",
                setup=r.get("test_setup_code", ""),
            )
        )
    return result
