import json

from accident_vlm.benchmark import (
    BenchmarkOptions,
    _multipart_body,
    discover_videos,
    inspect_result_quality,
    summarize_benchmark_items,
)


def test_discover_videos_sorts_and_applies_limit(tmp_path):
    (tmp_path / "b.mp4").write_bytes(b"b")
    (tmp_path / "a.mov").write_bytes(b"a")
    (tmp_path / "ignore.txt").write_text("x")

    videos = discover_videos(tmp_path, sample_limit=1)

    assert [video.name for video in videos] == ["a.mov"]


def test_inspect_result_quality_measures_required_fields():
    quality = inspect_result_quality(
        {
            "objective_summary": "차량과 보행자가 확인됨",
            "scene_type": {"value": "일반도로"},
            "traffic_control": {"signal": {"value": "적색"}},
            "actors": [{"id": "A"}],
            "timeline": [{"time": "00:01"}],
            "collision": {"impact_type": "보행자 충돌"},
            "uncertainties": ["야간"],
        }
    )

    assert quality["has_objective_summary"] is True
    assert quality["scene_type_known"] is True
    assert quality["traffic_signal_known"] is True
    assert quality["actor_count"] == 1
    assert quality["timeline_count"] == 1
    assert quality["collision_known"] is True


def test_summarize_benchmark_items_reports_rates():
    summary = summarize_benchmark_items(
        [
            {
                "status": "succeeded",
                "elapsed_sec": 10,
                "quality": {
                    "has_objective_summary": True,
                    "scene_type_known": True,
                    "traffic_signal_known": False,
                    "actor_count": 1,
                    "timeline_count": 1,
                    "collision_known": False,
                },
                "label_evaluation": {"checks": {"accident_object_match": True}},
            },
            {
                "status": "failed",
                "elapsed_sec": 20,
            },
        ]
    )

    assert summary["sample_count"] == 2
    assert summary["success_rate"] == 0.5
    assert summary["avg_elapsed_sec"] == 15
    assert summary["field_fill_rates"]["actors_present"] == 1.0
    assert summary["accident_object_match_rate"] == 1.0


def test_multipart_body_contains_fields_and_file(tmp_path):
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"video-bytes")

    body, content_type = _multipart_body({"mode": "full"}, "file", video)

    assert content_type.startswith("multipart/form-data; boundary=")
    assert b'name="mode"' in body
    assert b"full" in body
    assert b'filename="sample.mp4"' in body
    assert b"video-bytes" in body


def test_benchmark_options_defaults_are_json_serializable(tmp_path):
    options = BenchmarkOptions(api_base_url="http://localhost", video_root=tmp_path)

    assert json.dumps({"api_base_url": options.api_base_url, "sample_limit": options.sample_limit})
