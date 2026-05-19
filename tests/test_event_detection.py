from accident_vlm.modules.event_detection import detect_event_candidates


def test_detect_event_candidates_adds_distance_drop_and_sudden_stop_events() -> None:
    tracks = [
        {
            "track_id": "T1",
            "movement_candidate": "직진",
            "positions": [
                {"time": "00:01.000", "frame_id": "f1", "bbox": [10, 10, 50, 50]},
                {"time": "00:02.000", "frame_id": "f2", "bbox": [30, 30, 110, 110]},
                {"time": "00:03.000", "frame_id": "f3", "bbox": [35, 35, 120, 120]},
            ],
        }
    ]
    speed_and_distance = {
        "speed_estimates": [
            {
                "actor_id": "ego_vehicle",
                "numeric_kmh": 45,
                "method": "ocr_overlay",
                "evidence": ["f1"],
            },
            {
                "actor_id": "ego_vehicle",
                "numeric_kmh": 8,
                "method": "ocr_overlay",
                "evidence": ["f3"],
            },
        ]
    }

    events = detect_event_candidates(tracks, speed_and_distance)

    event_types = {event["event_type"] for event in events}
    assert "급접근" in event_types
    assert "급감속" in event_types


def test_detect_event_candidates_adds_camera_shake_collision_signal() -> None:
    events = detect_event_candidates(
        tracks=[],
        speed_and_distance=None,
        input_quality={
            "camera_shake_score": {
                "value": 28.0,
                "time": "00:04.000",
                "evidence": ["frame_000120", "frame_000121"],
            }
        },
    )

    assert events == [
        {
            "time": "00:04.000",
            "event_type": "충격후보",
            "actors": [],
            "confidence": "low",
            "signals": ["camera_shake"],
            "evidence": ["frame_000120", "frame_000121"],
        }
    ]
