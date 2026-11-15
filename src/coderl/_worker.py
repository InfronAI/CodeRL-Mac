"""Stdlib-only execution worker. Invoked in an isolated interpreter, never imported by training."""

import contextlib
import io
import json
import math
import resource
import signal
import sys
import traceback


class BoundedIO(io.StringIO):
    def write(self, text):
        if self.tell() + len(text) > 65536:
            raise RuntimeError("output limit exceeded")
        return super().write(text)


def equal(actual, expected):
    if isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        return len(actual) == len(expected) and all(equal(a, b) for a, b in zip(actual, expected))
    if isinstance(actual, dict) and isinstance(expected, dict):
        return {str(k): v for k, v in actual.items()} == {str(k): v for k, v in expected.items()}
    if isinstance(actual, (float, int)) and isinstance(expected, (float, int)):
        return math.isclose(actual, expected, rel_tol=1e-5, abs_tol=1e-8)
    return actual == expected


def stdout_equal(actual, expected):
    if isinstance(expected, list):
        expected = "\n".join(str(x) for x in expected)
    a, b = actual.split(), str(expected).split()
    if a == b:
        return True
    if len(a) != len(b):
        return False
    try:
        return all(math.isclose(float(x), float(y), rel_tol=1e-5, abs_tol=1e-8) for x, y in zip(a, b))
    except ValueError:
        return False


def alarm(_signum, _frame):
    raise TimeoutError("test exceeded wall-time limit")


def execute(payload):
    code = payload["code"]
    tests = payload["tests"]
    mode = payload["mode"]
    total = len(tests.get("assertions", [])) if mode == "assert" else len(tests.get("inputs", []))
    if total == 0:
        return {"outcome": "no_tests", "passed": 0, "total": 0, "error": "No tests available"}
    if mode != "assert" and total != len(tests.get("outputs", [])):
        return {
            "outcome": "harness_error",
            "passed": 0,
            "total": total,
            "error": "Input/output length mismatch",
        }
    try:
        compiled = compile(code, "<candidate>", "exec")
    except (SyntaxError, ValueError, OverflowError) as error:
        return {
            "outcome": "compile_error",
            "passed": 0,
            "total": total,
            "error": f"{type(error).__name__}: {error}"[:2000],
        }
    signal.signal(signal.SIGALRM, alarm)
    passed = 0
    namespace = {"__name__": "__main__"}
    initialized = False
    fn = None
    for i in range(total):
        if mode == "stdio":
            namespace = {"__name__": "__main__"}
        # APPS call-based references commonly omit typing imports.
        exec("from typing import *", namespace)
        output = BoundedIO()
        old_stdin = sys.stdin
        try:
            signal.setitimer(signal.ITIMER_REAL, payload["timeout"])
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(BoundedIO()):
                if mode == "stdio":
                    inp = tests["inputs"][i]
                    if isinstance(inp, list):
                        inp = "\n".join(str(x) for x in inp)
                    sys.stdin = io.TextIOWrapper(io.BytesIO(str(inp).encode()))
                    try:
                        exec(compiled, namespace)
                    except SystemExit as error:
                        if error.code not in (0, None):
                            raise
                    success = stdout_equal(output.getvalue(), tests["outputs"][i])
                else:
                    if not initialized:
                        exec(payload.get("setup", ""), namespace)
                        exec(compiled, namespace)
                        initialized = True
                    if mode == "call":
                        fn_name = payload["fn_name"]
                        if fn is None:
                            fn = (
                                getattr(namespace["Solution"](), fn_name)
                                if "Solution" in namespace
                                else namespace[fn_name]
                            )
                        args = tests["inputs"][i]
                        if not isinstance(args, list):
                            args = [args]
                        actual = fn(*args)
                        expected = tests["outputs"][i]
                        success = equal(actual, expected)
                        # APPS can serialize a scalar return as a singleton list.
                        if isinstance(expected, list) and len(expected) == 1:
                            success = success or equal(actual, expected[0])
                    else:
                        exec(tests["assertions"][i], namespace)
                        success = True
            if not success:
                return {"outcome": "failed", "passed": passed, "total": total, "error": "WrongAnswer"}
            passed += 1
        except AssertionError:
            return {"outcome": "failed", "passed": passed, "total": total, "error": "AssertionError"}
        except BaseException as error:
            return {
                "outcome": "runtime_error",
                "passed": passed,
                "total": total,
                "error": f"{type(error).__name__}: {error}"[:2000],
            }
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            sys.stdin = old_stdin
    return {"outcome": "passed", "passed": passed, "total": total, "error": ""}


if __name__ == "__main__":
    payload = json.load(sys.stdin)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    cpu = max(2, math.ceil(payload["budget"]))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
    # RLIMIT_AS does not work reliably with macOS's shared mappings.
    if sys.platform != "darwin":
        resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))
    try:
        result = execute(payload)
    except BaseException:
        result = {
            "outcome": "harness_error",
            "error": traceback.format_exc()[-2000:],
            "passed": 0,
            "total": 0,
        }
    sys.stdout.write(json.dumps(result))
