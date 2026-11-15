"""Validate recorded files, media, links and model-run provenance without rerunning training."""

import argparse
import re
import subprocess
import zipfile
from pathlib import Path

from coderl.io import digest, read_json, write_json
from coderl.pipeline import Experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    args = parser.parse_args()
    result = {"runs": {}, "readme_links": {}, "media": {}}
    for name in args.runs:
        root = Path(name)
        m = read_json(root / "manifest.json")
        mismatches = []
        checked = 0
        for stage, info in m["stages"].items():
            if info["status"] != "complete":
                mismatches.append(stage + ": incomplete")
            for path, expected in info.get("artifacts", {}).items():
                checked += 1
                if not (root / path).exists() or digest(root / path) != expected:
                    mismatches.append(path)
        source_ok = m["identity"]["source"] == Experiment.source_hash()
        with zipfile.ZipFile(root / "source.zip") as z:
            bad = z.testzip()
        result["runs"][name] = {
            "status": m["status"],
            "checked_files": checked,
            "mismatches": mismatches,
            "source_matches_current": source_ok,
            "source_zip_error": bad,
        }
        if mismatches or not source_ok or bad or m["status"] != "complete":
            raise RuntimeError(result["runs"][name])
    for file in ["README.md", "README.zh-CN.md"]:
        text = Path(file).read_text()
        missing = []
        checked = 0
        for link in re.findall(r"\]\(([^)]+)\)", text):
            if "://" in link or link.startswith("#"):
                continue
            checked += 1
            if not Path(link.split("#")[0]).exists():
                missing.append(link)
        result["readme_links"][file] = {"checked": checked, "missing": missing}
        if missing:
            raise RuntimeError(missing)
    probe = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_name,width,height,pix_fmt",
            "-show_entries",
            "format=duration,size",
            "-of",
            "json",
            "docs/assets/demo.mp4",
        ],
        text=True,
    )
    import json

    result["media"] = json.loads(probe)
    for stream in result["media"]["streams"]:
        assert stream["codec_name"] == "h264" and stream["pix_fmt"] == "yuv420p"
    assert float(result["media"]["format"]["duration"]) > 0
    session = read_json("docs/assets/demo-session.json")
    result["demo_greedy_outcome"] = session["greedy"]["result"]["outcome"]
    result["media_sha256"] = digest("docs/assets/demo.mp4")
    write_json("experiments/validation.json", result)
    print(
        "Validated",
        sum(x["checked_files"] for x in result["runs"].values()),
        "artifact hashes and all README/media references",
    )


if __name__ == "__main__":
    main()
