import cv2
import numpy as np

from accident_vlm.modules.ocr import (
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
    assert summary["gps"]["value"] == {"lat": 37.2, "lon": 127.2}
    assert summary["gps"]["sample_count"] == 3


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
