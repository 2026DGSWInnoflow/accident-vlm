import json
from pathlib import Path

from typer.testing import CliRunner

from accident_vlm.cli import app
from accident_vlm.schemas.preprocessing import PipelineContext, VideoMetadata


def test_cli_version_exits_successfully() -> None:
    result = CliRunner().invoke(app, ["--version"])

    assert result.exit_code == 0
    assert "0.1.0" in result.output


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
