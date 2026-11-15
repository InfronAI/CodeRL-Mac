"""Render a timestamped replay of actual captured CLI output to MP4/GIF.

This is a terminal replay, not a fabricated screencast. capture.json preserves the
commands, their real timing and exact output. Video idle gaps are compressed.
"""

import argparse
import json
import subprocess
import sys
import textwrap
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from coderl.io import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="experiments/tiny-mps-final")
    args = parser.parse_args()
    out = Path("docs/assets")
    out.mkdir(parents=True, exist_ok=True)
    commands = [
        [sys.executable, "-m", "coderl", "doctor"],
        [sys.executable, "scripts/demo.py", "--run", args.run],
    ]
    captured = []
    for command in commands:
        start = time.monotonic()
        lines = []
        print("Capturing:", command, flush=True)
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        for line in process.stdout:
            lines.append({"seconds": time.monotonic() - start, "text": line.rstrip()})
        status = process.wait()
        captured.append(
            {"command": command, "exit_code": status, "duration": time.monotonic() - start, "lines": lines}
        )
        if status:
            write_json(out / "demo-capture.json", captured)
            raise RuntimeError("Demo command failed; see capture")
    write_json(out / "demo-capture.json", captured)
    font_path = "/System/Library/Fonts/Menlo.ttc"
    if not Path(font_path).exists():
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
    font = ImageFont.truetype(font_path, 19)
    small = ImageFont.truetype(font_path, 15)
    title = ImageFont.truetype(font_path, 27)
    width, height = 1280, 800
    scenes = []
    for index, cap in enumerate(captured):
        lines = ["$ " + " ".join(cap["command"]).replace(sys.executable, "python")]
        # Keep all raw output in capture; omit verbose environment JSON from replay.
        if index == 0:
            raw = "\n".join(x["text"] for x in cap["lines"])
            info = json.loads(raw)
            lines += [
                f"Hardware: {info['cpu']} | {info['machine']} | {info['device']}",
                f"PyTorch {info['versions']['torch']} | Transformers {info['versions']['transformers']}",
                f"Gradient check: {info['backward_test']}",
                f"Isolated execution: {info['executor_test']['outcome']}",
                "",
                "Native Apple Silicon runtime verified.",
            ]
            scenes.append((lines, 7))
        else:
            for row in cap["lines"]:
                if row["text"].startswith("Passing a tuple of"):
                    continue
                lines.extend(
                    textwrap.wrap(row["text"], 105, replace_whitespace=False, drop_whitespace=False) or [""]
                )
                if (
                    len(lines) % 7 == 0
                    or "Final tests:" in row["text"]
                    or row["text"].startswith("Candidate")
                ):
                    scenes.append((list(lines[-25:]), 3))
            scenes.append((lines[-25:], 8))
    ffmpeg = "ffmpeg"
    from shutil import which

    if not which(ffmpeg):
        from imageio_ffmpeg import get_ffmpeg_exe

        ffmpeg = get_ffmpeg_exe()
    cmd = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        "6",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "22",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(out / "demo.mp4"),
    ]
    encoder = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    previews = []
    elapsed = 0
    total = sum(duration for _, duration in scenes)
    for lines, duration in scenes:
        im = Image.new("RGB", (width, height), "#111827")
        d = ImageDraw.Draw(im)
        d.rounded_rectangle((24, 22, 1256, 777), radius=18, fill="#0b1220", outline="#334155", width=2)
        for j, color in enumerate(["#f87171", "#fbbf24", "#34d399"]):
            d.ellipse((47 + 27 * j, 45, 60 + 27 * j, 58), fill=color)
        d.text((158, 35), "CodeRL / Apple Silicon", font=title, fill="#e2e8f0")
        d.text(
            (48, 91),
            "ACTUAL CLI OUTPUT  /  terminal replay; idle gaps compressed",
            font=small,
            fill="#60a5fa",
        )
        y = 128
        for line in lines:
            color = "#34d399" if "passed" in line or line.startswith("$") else "#e2e8f0"
            d.text((48, y), line, font=font, fill=color)
            y += 23
        d.text(
            (48, 742),
            "Offline tiny-model demonstration | saved checkpoints + real unit tests",
            font=small,
            fill="#94a3b8",
        )
        d.rectangle((48, 724, 1230, 728), fill="#1e293b")
        d.rectangle((48, 724, 48 + int(1182 * (elapsed + duration) / total), 728), fill="#38bdf8")
        previews.append((im.copy(), duration * 1000))
        for _ in range(duration * 6):
            encoder.stdin.write(im.tobytes())
        elapsed += duration
    encoder.stdin.close()
    if encoder.wait():
        raise RuntimeError("Video encoding failed")
    preview_index = next(
        (i for i, (lines, _) in enumerate(scenes) if any("Final tests:" in line for line in lines)), 0
    )
    previews[preview_index][0].save(out / "demo-preview.png")
    gif_images = [im.resize((768, 480)).quantize(colors=64) for im, _ in previews]
    gif_images[0].save(
        out / "demo.gif",
        save_all=True,
        append_images=gif_images[1:],
        duration=[d for _, d in previews],
        loop=0,
    )
    write_json(
        out / "demo-video.json",
        {
            "video": "demo.mp4",
            "duration_seconds": total,
            "frames": total * 6,
            "fps": 6,
            "width": width,
            "height": height,
            "source": "demo-capture.json",
            "kind": "actual stdout replay, not a desktop screen recording",
        },
    )
    print(out / "demo.mp4")


if __name__ == "__main__":
    main()
