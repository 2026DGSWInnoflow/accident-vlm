import json

from accident_vlm.benchmark import (
    BenchmarkOptions,
    _multipart_body,
    compute_preprocessing_metrics,
    discover_videos,
    inspect_result_quality,
    load_benchmark_manifest,
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
            "insurance_claim_fields": {
                "accident_datetime": {"value": "2026-05-21 12:00"},
                "road_shape": {"value": "교차로"},
                "damage_parts": [{"actor_id": "A", "part": "전면"}],
            },
            "accident_type_candidates": {
                "vehicle_to_pedestrian": {"status": "observed", "evidence": ["slot_1"]}
            },
            "uncertainties": ["야간"],
        }
    )

    assert quality["has_objective_summary"] is True
    assert quality["scene_type_known"] is True
    assert quality["traffic_signal_known"] is True
    assert quality["actor_count"] == 1
    assert quality["timeline_count"] == 1
    assert quality["collision_known"] is True
    assert quality["insurance_fields_known_count"] == 3
    assert quality["accident_type_candidate_known"] is True


def test_inspect_result_quality_handles_list_values_inside_collision():
    quality = inspect_result_quality(
        {
            "objective_summary": "요약",
            "scene_type": {"value": "확인불가"},
            "collision": {"source": ["chunk_1"], "evidence": [], "occurred": ["likely"]},
        }
    )

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
                    "insurance_fields_known_count": 2,
                    "insurance_fields_total": 8,
                    "accident_type_candidate_known": True,
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
    assert summary["field_fill_rates"]["insurance_fields_avg_fill_rate"] == 0.25
    assert summary["field_fill_rates"]["accident_type_candidates_present"] == 1.0
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
    assert options.verify_tls is True


def test_load_benchmark_manifest_validates_required_labels(tmp_path):
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"video")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "accident_vlm.benchmark_manifest.v1",
                "items": [
                    {
                        "id": "sample",
                        "video_path": str(video),
                        "labels": {
                            "accident_type": "차대차",
                            "actors": ["ego_vehicle", "vehicle_1"],
                            "collision_time": "00:03.000",
                            "traffic_signal": "적색",
                            "lane_count": 3,
                            "ocr": {"speed_kmh": 42},
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    loaded = load_benchmark_manifest(manifest)

    assert loaded["items"][0]["id"] == "sample"
    assert loaded["items"][0]["labels"]["lane_count"] == 3


def test_compute_preprocessing_metrics_reports_recall_accuracy_and_uncertainty():
    metrics = compute_preprocessing_metrics(
        precomputed={
            "event_candidates": [{"time": "00:03.100", "event_score": 90}],
            "tracks": [{"track_id": "ego_vehicle"}, {"track_id": "vehicle_1", "track_quality": {"fragmentation_score": 12}}],
            "traffic_control": {"signal": {"value": "적색"}},
            "road_geometry": {"visible_lane_count": {"value": 3}},
            "ocr_summary": {"speed": {"numeric_kmh": 43}},
        },
        final_output={
            "insurance_claim_fields": {
                "accident_datetime": {"value": "확인불가"},
                "road_shape": {"value": "교차로"},
            },
            "uncertainties": ["장소 확인불가"],
        },
        labels={
            "collision_time": "00:03.000",
            "actors": ["ego_vehicle", "vehicle_1"],
            "traffic_signal": "적색",
            "lane_count": 3,
            "ocr": {"speed_kmh": 42},
        },
    )

    assert metrics["collision_recall_at_3"] == 1.0
    assert metrics["actor_recall"] == 1.0
    assert metrics["actor_miss_rate"] == 0.0
    assert metrics["actor_false_positive_rate"] == 0.0
    assert metrics["track_fragmentation_rate"] == 0.06
    assert metrics["signal_accuracy"] == 1.0
    assert metrics["lane_count_accuracy"] == 1.0
    assert metrics["ocr_speed_accuracy"] == 1.0
    assert metrics["tracking_continuity"] > 0.8
    assert metrics["insurance_known_unknown_appropriateness"] == 1.0
