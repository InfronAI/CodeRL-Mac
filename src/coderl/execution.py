"""Bounded program testing, with an OS sandbox on macOS and process-group cleanup."""

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

OUTCOMES = ("compile_error", "runtime_error", "failed", "passed")
REWARDS = (-1.0, -0.6, -0.3, 1.0)


def outcome_label(result):
    try:
        return OUTCOMES.index(result["outcome"])
    except ValueError as error:
        raise ValueError(
            f"Cannot train/evaluate on infrastructure or missing-test result: {result}"
        ) from error


def reward(result):
    return REWARDS[outcome_label(result)]


def _sandbox_profile(directory, worker):
    def q(s):
        return json.dumps(str(s))

    return f"""(version 1)
(allow default)
(deny network*)
(deny process-fork)
(deny file-write*)
(deny file-read* (subpath "/Users") (subpath "/Volumes") (subpath "/System/Volumes/Data/Users"))
(allow file-read* (subpath {q(directory)}) (literal {q(worker)}))
(allow file-write* (subpath {q(directory)}) (literal "/dev/null"))
"""


class Runner:
    def __init__(self, timeout=2.0, program_timeout=20.0):
        if timeout <= 0 or program_timeout <= 0:
            raise ValueError("Timeouts must be positive")
        self.timeout = timeout
        self.program_timeout = program_timeout
        self.sandbox = shutil.which("sandbox-exec") if sys.platform == "darwin" else None
        if sys.platform == "darwin" and not self.sandbox:
            raise RuntimeError("macOS sandbox-exec is required for generated code execution")
        from .data import PublicProblem

        probe = self.run(
            "print(42)", PublicProblem("executor-probe", ""), {"inputs": [""], "outputs": ["42"]}
        )
        if probe["outcome"] != "passed":
            raise RuntimeError(f"Execution environment failed its known-good preflight: {probe}")

    def run(self, code, problem, tests):
        started = time.monotonic()
        worker = Path(__file__).with_name("_worker.py").resolve()
        count = len(tests.get("assertions", [])) if problem.mode == "assert" else len(tests.get("inputs", []))
        budget = min(self.program_timeout, self.timeout * max(1, count))
        payload = {
            "code": code,
            "tests": tests,
            "mode": problem.mode,
            "fn_name": problem.fn_name,
            "setup": problem.setup,
            "timeout": self.timeout,
            "budget": budget,
        }
        with tempfile.TemporaryDirectory(prefix="coderl-") as tmp:
            tmp = str(Path(tmp).resolve())
            cmd = [str(Path(sys._base_executable).resolve()), "-I", "-S", str(worker)]
            if self.sandbox:
                cmd = [self.sandbox, "-p", _sandbox_profile(tmp, worker)] + cmd
            env = {"PATH": os.defpath, "HOME": tmp, "TMPDIR": tmp, "LANG": "en_US.UTF-8"}
            with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
                process = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=stdout,
                    stderr=stderr,
                    cwd=tmp,
                    env=env,
                    start_new_session=True,
                )
                try:
                    process.communicate(json.dumps(payload).encode(), timeout=budget + 2)
                    stdout.seek(0)
                    stderr.seek(0)
                    raw = stdout.read(65536)
                    err = stderr.read(4096).decode(errors="replace")
                    if process.returncode != 0:
                        infra = "sandbox" in err.lower() or "can't open file" in err
                        result = {
                            "outcome": "harness_error" if infra else "runtime_error",
                            "passed": 0,
                            "total": count,
                            "error": err or f"WorkerExit({process.returncode})",
                        }
                    else:
                        try:
                            result = json.loads(raw)
                        except (ValueError, UnicodeDecodeError):
                            result = {
                                "outcome": "runtime_error",
                                "passed": 0,
                                "total": count,
                                "error": "Worker terminated without a valid test result",
                            }
                except subprocess.TimeoutExpired:
                    result = {
                        "outcome": "runtime_error",
                        "passed": 0,
                        "total": count,
                        "error": "ProgramTimeout",
                    }
                finally:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
        result["seconds"] = round(time.monotonic() - started, 6)
        result["isolation"] = (
            "macOS seatbelt + resource limits"
            if self.sandbox
            else "process + resource limits (not a security sandbox)"
        )
        if result["outcome"] == "harness_error":
            raise RuntimeError(result["error"])
        return result
