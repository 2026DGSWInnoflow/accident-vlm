from accident_vlm.modules.ocr import summarize_ocr_observations


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
