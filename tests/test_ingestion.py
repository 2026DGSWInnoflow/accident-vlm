import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from accident_vlm.modules.ingestion import IngestionError, parse_ffprobe_streams, probe_video


def test_parse_ffprobe_streams_video_and_audio():
    data = {
        "streams": [
            {
                "codec_type": "video",
                "width": 1920,
                "height": 1080,
                "avg_frame_rate": "30/1",
                "nb_frames": "300",
                "duration": "10.0",
            },
            {"codec_type": "audio"},
        ]
    }

    metadata = parse_ffprobe_streams(data)

    assert metadata.duration_sec == 10.0
    assert metadata.fps == 30
    assert metadata.resolution == "1920x1080"
    assert metadata.frame_count == 300
    assert metadata.has_audio is True


def test_parse_ffprobe_streams_raises_when_no_video_stream():
    data = {"streams": [{"codec_type": "audio"}]}

    with pytest.raises(IngestionError, match="No video stream"):
        parse_ffprobe_streams(data)


def test_parse_ffprobe_streams_no_audio_sets_has_audio_false():
    data = {
        "streams": [
            {
                "codec_type": "video",
                "width": 1280,
                "height": 720,
                "avg_frame_rate": "24/1",
                "nb_frames": "48",
                "duration": "2.0",
            }
        ]
    }

    metadata = parse_ffprobe_streams(data)

    assert metadata.has_audio is False


def test_parse_ffprobe_streams_uses_format_duration_when_stream_duration_invalid():
    data = {
        "streams": [
            {
                "codec_type": "video",
                "width": 1920,
                "height": 1080,
                "avg_frame_rate": "30/1",
                "nb_frames": "90",
                "duration": "N/A",
            }
        ],
        "format": {"duration": "3.0"},
    }

    metadata = parse_ffprobe_streams(data)

    assert metadata.duration_sec == 3.0
    assert metadata.frame_count == 90


def test_parse_ffprobe_streams_uses_r_frame_rate_when_avg_frame_rate_missing():
    data = {
        "streams": [
            {
                "codec_type": "video",
                "width": 1920,
                "height": 1080,
                "r_frame_rate": "60/1",
                "nb_frames": "120",
                "duration": "2.0",
            }
        ]
    }

    metadata = parse_ffprobe_streams(data)

    assert metadata.fps == 60


def test_parse_ffprobe_streams_allows_missing_duration_when_nb_frames_valid():
    data = {
        "streams": [
            {
                "codec_type": "video",
                "width": 1920,
                "height": 1080,
                "avg_frame_rate": "30/1",
                "nb_frames": "300",
                "duration": "N/A",
            }
        ]
    }

    metadata = parse_ffprobe_streams(data)

    assert metadata.duration_sec == 0.0
    assert metadata.frame_count == 300


def test_parse_ffprobe_streams_nb_frames_na_falls_back_to_duration_times_fps():
    data = {
        "streams": [
            {
                "codec_type": "video",
                "width": 640,
                "height": 480,
                "avg_frame_rate": "30000/1001",
                "nb_frames": "N/A",
                "duration": "10.0",
            }
        ]
    }

    metadata = parse_ffprobe_streams(data)

    assert metadata.frame_count == 300


@pytest.mark.parametrize("frame_rate", [None, "N/A", "0/0", "0/1", "-30/1", "not-a-rate"])
def test_parse_ffprobe_streams_invalid_frame_rate_raises(frame_rate):
    data = {
        "streams": [
            {
                "codec_type": "video",
                "width": 1920,
                "height": 1080,
                "avg_frame_rate": frame_rate,
                "duration": "1.0",
            }
        ]
    }

    with pytest.raises(IngestionError, match="frame rate"):
        parse_ffprobe_streams(data)


@pytest.mark.parametrize("field_name", ["width", "height"])
def test_parse_ffprobe_streams_missing_dimension_raises(field_name):
    video_stream = {
        "codec_type": "video",
        "width": 1920,
        "height": 1080,
        "avg_frame_rate": "30/1",
        "nb_frames": "30",
        "duration": "1.0",
    }
    del video_stream[field_name]
    data = {"streams": [video_stream]}

    with pytest.raises(IngestionError, match=field_name):
        parse_ffprobe_streams(data)


def test_probe_video_requests_format_metadata_from_ffprobe():
    ffprobe_output = """
    {
      "streams": [
        {
          "codec_type": "video",
          "width": 1920,
          "height": 1080,
          "avg_frame_rate": "30/1",
          "nb_frames": "300",
          "duration": "10.0"
        }
      ],
      "format": {"duration": "10.0"}
    }
    """
    completed = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout=ffprobe_output,
        stderr="",
    )

    with patch("accident_vlm.modules.ingestion.subprocess.run", return_value=completed) as run:
        probe_video(Path("sample.mp4"))

    command = run.call_args.args[0]
    assert "-show_streams" in command
    assert "-show_format" in command


def test_probe_video_caches_missing_ffprobe_between_calls(monkeypatch):
    calls = []
    metadata = object()

    def fake_run(*args, **kwargs):
        calls.append(args)
        raise FileNotFoundError

    monkeypatch.setattr("accident_vlm.modules.ingestion._FFPROBE_AVAILABLE", None)
    monkeypatch.setattr("accident_vlm.modules.ingestion.subprocess.run", fake_run)
    monkeypatch.setattr("accident_vlm.modules.ingestion.probe_video_with_opencv", lambda path: metadata)

    assert probe_video(Path("first.mp4")) is metadata
    assert probe_video(Path("second.mp4")) is metadata
    assert len(calls) == 1
