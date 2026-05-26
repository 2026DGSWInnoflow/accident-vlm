import json
import subprocess
import sys
from pathlib import Path

from typer.testing import CliRunner

from accident_vlm.cli import app
import accident_vlm.cli as cli
from accident_vlm.schemas.preprocessing import PipelineContext, VideoMetadata


def test_cli_version_exits_successfully() -> None:
    result = CliRunner().invoke(app, ["--version"])

    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_cli_import_defers_heavy_command_modules() -> None:
    script = """
import sys
import accident_vlm.cli
for name in (
    "accident_vlm.benchmark",
    "accident_vlm.evaluation",
    "accident_vlm.modules.vlm_composer",
    "accident_vlm.pipeline",
    "rich",
):
    print(name, name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "True" not in result.stdout


def test_cli_analyze_writes_pre_vlm_context(monkeypatch, tmp_path: Path) -> None:
    video_path = tmp_path / "sample.mp4"
    output_path = tmp_path / "nested" / "context.json"
    metadata = VideoMetadata(
        duration_sec=2.0,
        fps=30.0,
        resolution="1920x1080",
        frame_count=60,
        has_audio=False,
    )
    calls = {}

    def fake_analyze_video_pre_vlm(video_path: Path, config):
        calls["video_path"] = video_path
        calls["config"] = config
        return PipelineContext(
            video_path=str(video_path),
            video_metadata=metadata,
            evidence_package={
                "frames": [],
                "overlays": [],
                "crops": [],
                "precomputed_facts": {"metadata": metadata.model_dump()},
            },
        )

    monkeypatch.setattr("accident_vlm.cli.analyze_video_pre_vlm", fake_analyze_video_pre_vlm)

    result = CliRunner().invoke(app, ["analyze", str(video_path), str(output_path)])

    assert result.exit_code == 0
    assert calls["video_path"] == video_path
    assert calls["config"].ocr_backend == "auto"

    output = json.loads(output_path.read_text(encoding="utf-8"))
    assert output["video_path"] == str(video_path)
    assert output["video_metadata"]["fps"] == 30.0
    assert output["evidence_package"]["overlays"] == []
    assert output["evidence_package"]["crops"] == []
    assert output["evidence_package"]["precomputed_facts"]["metadata"]["fps"] == 30.0
    assert str(output_path) in result.output


def test_cli_fast_analyze_path_writes_context_without_typer_dispatch(monkeypatch, tmp_path: Path) -> None:
    video_path = tmp_path / "sample.mp4"
    output_path = tmp_path / "context.json"
    metadata = VideoMetadata(
        duration_sec=1.0,
        fps=10.0,
        resolution="640x360",
        frame_count=10,
        has_audio=False,
    )
    calls = {}

    def fake_analyze_video_pre_vlm(video_path: Path, config):
        calls["video_path"] = video_path
        calls["config"] = config
        return PipelineContext(
            video_path=str(video_path),
            video_metadata=metadata,
        )

    monkeypatch.setattr("accident_vlm.cli.analyze_video_pre_vlm", fake_analyze_video_pre_vlm)

    handled = cli._try_fast_analyze(
        [
            "accident-vlm",
            "analyze",
            str(video_path),
            str(output_path),
            "--ocr-backend",
            "none",
            "--detector",
            "none",
            "--output-dir",
            str(tmp_path / "outputs"),
        ]
    )

    assert handled is True
    assert calls["video_path"] == video_path
    assert calls["config"].ocr_backend == "none"
    assert calls["config"].object_detector_backend == "none"
    assert calls["config"].output_dir == tmp_path / "outputs"
    assert json.loads(output_path.read_text(encoding="utf-8"))["video_path"] == str(video_path)


def test_cli_module_fast_analyze_exits_before_typer_import(tmp_path: Path) -> None:
    video_path = tmp_path / "sample.mp4"
    output_path = tmp_path / "context.json"
    script = f"""
import cv2
import json
import numpy as np
import runpy
import sys

video_path = {str(video_path)!r}
writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), 5, (32, 24))
for _ in range(3):
    writer.write(np.zeros((24, 32, 3), dtype=np.uint8))
writer.release()

sys.argv = [
    "accident-vlm",
    "analyze",
    video_path,
    {str(output_path)!r},
    "--ocr-backend",
    "none",
    "--detector",
    "none",
]
try:
    runpy.run_module("accident_vlm.cli", run_name="__main__")
except SystemExit:
    pass
print(json.dumps({{"typer_loaded": "typer" in sys.modules}}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(result.stdout.strip().splitlines()[-1]) == {"typer_loaded": False}
    assert output_path.exists()
