"""Author a tiny, explicitly non-benchmark fixture for offline workflow testing."""

from pathlib import Path

from coderl.data import Problem
from coderl.io import write_jsonl

code = "print(sum(map(int,input().split())))\n"
questions = [
    "Read two integers from standard input and print their sum.",
    "Given two space-separated integers, output their total.",
    "Add the two integers on the input line. Print the answer.",
    "Compute the sum of two integers read from stdin.",
    "Read a pair of numbers and write the result of adding them.",
    "Input contains two integers. Output a single integer: their sum.",
]
rows = []
for i, q in enumerate(questions):
    split = "train" if i < 4 else "test"
    cases = (
        [("2 3\n", "5\n"), ("-4 7\n", "3\n")]
        if split == "train"
        else [("100 -3\n", "97\n"), ("0 0\n", "0\n"), ("-5 -8\n", "-13\n")]
    )
    p = Problem(
        id=f"fixture/{split}/{i}",
        question=q + "\nInput\n1 2\nOutput\n3\n",
        solutions=[code],
        tests={"inputs": [a for a, b in cases], "outputs": [b for a, b in cases]},
        examples={"inputs": ["1 2\n"], "outputs": ["3\n"]},
        split=split,
        dataset="authored addition fixture",
        difficulty="fixture",
    )
    rows.append(p.asdict())
Path("data/fixtures").mkdir(exist_ok=True)
write_jsonl("data/fixtures/addition.jsonl", rows)
