import sys

import pytest

from coderl.data import PublicProblem
from coderl.execution import Runner, outcome_label, reward


@pytest.mark.parametrize(
    "code,outcome,value",
    [
        ("def f(:", "compile_error", -1),
        ("raise ValueError('broken')", "runtime_error", -0.6),
        ("print(0)", "failed", -0.3),
        ("print(int(input())+1)", "passed", 1),
        ("while True: pass", "runtime_error", -0.6),
        ("import sys; print(3); sys.exit(0)", "passed", 1),
        ("import os; os._exit(0)", "runtime_error", -0.6),
        ("print('x'*100000)", "runtime_error", -0.6),
    ],
)
def test_real_execution(code, outcome, value):
    result = Runner(timeout=0.15).run(code, PublicProblem("x", "x"), {"inputs": ["2\n"], "outputs": ["3\n"]})
    assert result["outcome"] == outcome
    assert reward(result) == value


def test_call_solution_and_typing():
    p = PublicProblem("x", "sum", mode="call", fn_name="solve")
    code = "class Solution:\n def solve(self, xs: List[int]):\n  return sum(xs)"
    assert Runner().run(code, p, {"inputs": [[[1, 2]], [[-2, 5]]], "outputs": [3, 3]})["passed"] == 2


def test_mbpp_assertions():
    p = PublicProblem("x", "inc", mode="assert")
    tests = {"assertions": ["assert inc(2)==3", "assert inc(-1)==0"]}
    assert Runner().run("def inc(n): return n+1", p, tests)["outcome"] == "passed"
    assert Runner().run("def inc(n): return n", p, tests)["outcome"] == "failed"


def test_no_tests_not_success():
    result = Runner().run("print(1)", PublicProblem("x", "x"), {})
    assert result["outcome"] == "no_tests"
    with pytest.raises(ValueError):
        outcome_label(result)


def test_bad_dataset_fails_closed():
    with pytest.raises(RuntimeError, match="length mismatch"):
        Runner().run("print(1)", PublicProblem("x", "x"), {"inputs": ["x"], "outputs": []})


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS Seatbelt contract")
@pytest.mark.parametrize(
    "code",
    [
        "import socket; socket.create_connection(('1.1.1.1',80),timeout=.2); print(3)",
        "open('/Users/apple/coderl-forbidden','w').write('x'); print(3)",
        "import os; os.fork(); print(3)",
    ],
)
def test_sandbox_blocks_side_effects(code):
    assert (
        Runner().run(code, PublicProblem("x", "x"), {"inputs": ["2"], "outputs": ["3"]})["outcome"]
        == "runtime_error"
    )


def test_apps_scalar_wrapper_and_call_state():
    p = PublicProblem("x", "inc", mode="call", fn_name="f")
    code = "counter=0\ndef f(n):\n global counter\n counter+=1\n return counter+n"
    result = Runner().run(code, p, {"inputs": [[2], [2]], "outputs": [[3], [4]]})
    assert result["outcome"] == "passed"
