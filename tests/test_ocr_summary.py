import cv2
import numpy as np
import sys
import types

from accident_vlm.modules.ocr import (
    create_ocr_backend,
    extract_ocr_observations,
    parse_overlay_text,
    summarize_ocr_observations,
)
from accident_vlm.schemas.preprocessing import SelectedFrame


class RecordingOcrBackend:
    name = "recording"

    def __init__(self) -> None:
        self.paths = []

    def read_text(self, image_path):
        self.paths.append(str(image_path))
        return [
            {
                "text": "2026.05.19 08:31:22 47 km/h GPS 37.123456,127.654321",
                "confidence": 0.91,
                "bbox": [0, 0, 10, 10],
                "source": self.name,
                "image_path": str(image_path),
            }
        ]


class FieldAwareOcrBackend:
    name = "field_aware"

    def __init__(self) -> None:
        self.calls = []

    def read_text(self, image_path, field_hint=None):
        self.calls.append((str(image_path), field_hint))
        text_by_field = {
            "datetime": "2026/05/19 08:31:22",
            "speed": "SPD 47 km/h",
            "gps": "GPS 37.123456, 127.654321",
        }
        text = text_by_field.get(field_hint, "")
        if not text:
            return []
        return [
            {
                "text": text,
                "confidence": 0.93,
                "bbox": [1, 2, 20, 12],
                "source": self.name,
                "image_path": str(image_path),
            }
        ]


class EmptyOcrBackend:
    name = "empty"

    def read_text(self, image_path, field_hint=None):
        return []


def test_create_easyocr_backend_defaults_to_cpu_to_avoid_vlm_gpu_oom(monkeypatch) -> None:
    captured = {}

    class FakeReader:
        def __init__(self, languages, gpu):
            captured["languages"] = languages
            captured["gpu"] = gpu

    monkeypatch.setitem(sys.modules, "easyocr", types.SimpleNamespace(Reader=FakeReader))
    monkeypatch.delenv("ACCIDENT_VLM_EASYOCR_GPU", raising=False)
    create_ocr_backend.cache_clear()

    backend = create_ocr_backend("easyocr")

    assert backend.name == "easyocr"
    assert captured == {"languages": ["ko", "en"], "gpu": False}


def test_parse_overlay_text_supports_dashcam_datetime_speed_and_gps_variants() -> None:
    parsed = parse_overlay_text(
        "2026년 05월 19일 08:31:22 SPEED 47km/h GPS N37.123456 E127.654321"
    )

    assert parsed["datetime"] == "2026-05-19 08:31:22"
    assert parsed["speed_kmh"] == 47.0
    assert parsed["gps"] == {"lat": 37.123456, "lon": 127.654321}


def test_extract_ocr_observations_reads_overlay_rois_and_maps_bbox_to_frame(tmp_path) -> None:
    frame_path = tmp_path / "frame.jpg"
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    cv2.imwrite(str(frame_path), image)
    backend = RecordingOcrBackend()

    observations = extract_ocr_observations(
        [
            SelectedFrame(
                id="frame_000001",
                time="00:00.033",
                frame_index=1,
                purpose="regular_context",
                path=str(frame_path),
            )
        ],
        backend,
        roi_output_dir=tmp_path / "ocr_rois",
    )

    assert len(backend.paths) >= 5
    assert {observation["roi_name"] for observation in observations} >= {
        "full_frame",
        "bottom_band",
        "top_band",
    }
    bottom_observation = next(
        observation for observation in observations if observation["roi_name"] == "bottom_band"
    )
    assert bottom_observation["bbox"][1] >= 70
    assert bottom_observation["parsed"]["speed_kmh"] == 47.0


def test_extract_ocr_observations_runs_field_specific_passes_and_records_normalized_text(tmp_path) -> None:
    frame_path = tmp_path / "frame.jpg"
    image = np.zeros((120, 240, 3), dtype=np.uint8)
    cv2.putText(image, "2026/05/19 08:31:22", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(image, "SPD 47 km/h GPS 37.123456,127.654321", (5, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    cv2.imwrite(str(frame_path), image)
    backend = FieldAwareOcrBackend()

    observations = extract_ocr_observations(
        [
            SelectedFrame(
                id="frame_000010",
                time="00:00.333",
                frame_index=10,
                purpose="event_scan_impact_candidate",
                path=str(frame_path),
            )
        ],
        backend,
        roi_output_dir=tmp_path / "ocr_rois",
    )

    called_fields = {field for _, field in backend.calls}
    assert {"datetime", "speed", "gps"} <= called_fields
    assert any(observation["target_field"] == "datetime" for observation in observations)
    assert any(observation["target_field"] == "speed" for observation in observations)
    assert any(observation["target_field"] == "gps" for observation in observations)
    assert all("normalized_text" in observation for observation in observations if observation["status"] == "observed")
    assert all("crop_path" in observation for observation in observations)


def test_extract_ocr_observations_records_failure_cases_for_empty_field_pass(tmp_path) -> None:
    frame_path = tmp_path / "frame.jpg"
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    cv2.imwrite(str(frame_path), image)

    observations = extract_ocr_observations(
        [
            SelectedFrame(
                id="frame_000001",
                time="00:00.033",
                frame_index=1,
                purpose="regular_context",
                path=str(frame_path),
            )
        ],
        EmptyOcrBackend(),
        roi_output_dir=tmp_path / "ocr_rois",
    )

    failures = [observation for observation in observations if observation["status"] == "failed"]
    assert failures
    assert {"datetime", "speed", "gps"} & {failure["target_field"] for failure in failures}
    assert all(failure["failure_reason"] == "no_text_detected" for failure in failures)


def test_summarize_ocr_observations_votes_datetime_and_medians_speed_gps() -> None:
    summary = summarize_ocr_observations(
        [
            {
                "frame_id": "frame_000001",
                "confidence": 0.95,
                "parsed": {
                    "datetime": "2026-05-19 12:00:01",
                    "speed_kmh": 48.0,
                    "gps": {"lat": 37.1, "lon": 127.1},
                },
            },
            {
                "frame_id": "frame_000002",
                "confidence": 0.90,
                "parsed": {
                    "datetime": "2026-05-19 12:00:01",
                    "speed_kmh": 49.0,
                    "gps": {"lat": 37.2, "lon": 127.2},
                },
            },
            {
                "frame_id": "frame_000003",
                "confidence": 0.80,
                "parsed": {
                    "datetime": "2026-05-19 12:00:02",
                    "speed_kmh": 50.0,
                    "gps": {"lat": 37.3, "lon": 127.3},
                },
            },
        ]
    )

    assert summary["datetime"]["value"] == "2026-05-19 12:00:01"
    assert summary["datetime"]["confidence"] == "medium"
    assert summary["speed"]["numeric_kmh"] == 49.0
    assert summary["speed"]["range_kmh"] == [48.0, 50.0]
    assert summary["speed"]["confidence"] == "high"
    assert summary["speed"]["final_confidence_score"] > 0.8
    assert summary["speed"]["temporal_consistency"] == "stable"
    assert summary["gps"]["value"] == {"lat": 37.2, "lon": 127.2}
    assert summary["gps"]["sample_count"] == 3
    assert summary["field_votes"]["speed"]["observed_count"] == 3


def test_summarize_ocr_observations_returns_unknowns_when_no_parsed_values() -> None:
    summary = summarize_ocr_observations(
        [{"frame_id": "frame_000001", "confidence": 0.0, "parsed": {}}]
    )

    assert summary["datetime"]["status"] == "unknown"
    assert summary["speed"]["numeric_kmh"] is None
    assert summary["gps"]["value"] is None
    assert summary["observation_count"] == 1


def test_summarize_ocr_observations_removes_speed_outliers_and_records_candidates() -> None:
    summary = summarize_ocr_observations(
        [
            {"frame_id": "f1", "confidence": 0.9, "parsed": {"speed_kmh": 48.0}},
            {"frame_id": "f2", "confidence": 0.9, "parsed": {"speed_kmh": 49.0}},
            {"frame_id": "f3", "confidence": 0.9, "parsed": {"speed_kmh": 240.0}},
        ]
    )

    assert summary["speed"]["numeric_kmh"] == 48.5
    assert summary["speed"]["range_kmh"] == [48.0, 49.0]
    assert summary["speed"]["rejected_outliers"] == [{"frame_id": "f3", "value": 240.0}]
    assert summary["speed"]["candidates"] == [
        {"frame_id": "f1", "value": 48.0, "confidence": 0.9},
        {"frame_id": "f2", "value": 49.0, "confidence": 0.9},
        {"frame_id": "f3", "value": 240.0, "confidence": 0.9},
    ]


def test_summarize_ocr_observations_rejects_implausible_temporal_speed_jump() -> None:
    summary = summarize_ocr_observations(
        [
            {"frame_id": "f1", "confidence": 0.9, "parsed": {"speed_kmh": 40.0}},
            {"frame_id": "f2", "confidence": 0.9, "parsed": {"speed_kmh": 42.0}},
            {"frame_id": "f3", "confidence": 0.9, "parsed": {"speed_kmh": 88.0}},
            {"frame_id": "f4", "confidence": 0.9, "parsed": {"speed_kmh": 90.0}},
        ]
    )

    assert summary["speed"]["numeric_kmh"] == 41.0
    assert {
        "frame_id": "f3",
        "value": 88.0,
        "reason": "temporal_jump",
        "previous_value": 42.0,
    } in summary["speed"]["rejected_outliers"]
