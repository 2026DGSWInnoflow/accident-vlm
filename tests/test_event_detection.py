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
    assert "급접근후보" in event_types
    assert "급감속" in event_types
    assert all("event_score" in event for event in events)
    assert any(event.get("supporting_signals") for event in events)


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

    assert events[0]["event_type"] == "충격후보"
    assert events[0]["event_score"] > 0


def test_detect_event_candidates_outputs_multi_signal_collision_candidate() -> None:
    tracks = [
        {
            "track_id": "T1",
            "movement_candidate": "직진",
            "positions": [
                {"time": "00:01.000", "frame_id": "frame_000030", "bbox": [10, 10, 50, 50]},
                {"time": "00:02.000", "frame_id": "frame_000060", "bbox": [15, 15, 65, 65]},
            ],
        },
        {
            "track_id": "T2",
            "movement_candidate": "접근",
            "positions": [
                {"time": "00:01.000", "frame_id": "frame_000030", "bbox": [80, 80, 120, 120]},
                {"time": "00:02.000", "frame_id": "frame_000060", "bbox": [20, 20, 70, 70]},
            ],
        },
    ]

    events = detect_event_candidates(
        tracks,
        speed_and_distance={"relative_motion": [{"actor_id": "T2", "relative_speed_trend": "접근"}]},
        input_quality={
            "camera_shake_score": {
                "value": 26,
                "ego_motion_compensated_value": 0.12,
                "time": "00:02.000",
                "evidence": ["frame_000030", "frame_000060"],
            }
        },
    )

    collision = next(event for event in events if event["event_type"] == "접촉")
    assert collision["candidate_class"] == "direct_contact_candidate"
    assert collision["supporting_signals"]["bbox_iou"] > 0
    assert "bbox_iou_change_rate" in collision["supporting_signals"]
    assert "camera_shake" in collision["signals"]
    assert collision["contradicting_signals"] == []


def test_detect_event_candidates_keeps_non_contact_candidate() -> None:
    events = detect_event_candidates(
        tracks=[
            {
                "track_id": "T1",
                "movement_candidate": "직진",
                "positions": [
                    {"time": "00:01.000", "frame_id": "frame_000030", "bbox": [0, 0, 20, 20]},
                    {"time": "00:02.000", "frame_id": "frame_000060", "bbox": [0, 0, 30, 30]},
                ],
            }
        ],
        speed_and_distance={"relative_motion": [{"actor_id": "T1", "relative_speed_trend": "접근"}]},
    )

    assert any(event["event_type"] == "비접촉후보" for event in events)


def test_detect_event_candidates_preserves_event_scan_optical_flow_signal() -> None:
    events = detect_event_candidates(
        tracks=[],
        event_scan_candidates=[
            {
                "time": "00:03.000",
                "confidence": "medium",
                "event_score": 74,
                "supporting_signals": {
                    "optical_flow_peak": 2.1,
                    "optical_flow_mean": 0.8,
                    "histogram_change": 16.0,
                },
                "evidence": ["frame_000087", "frame_000090"],
            }
        ],
    )

    flow_event = next(event for event in events if event["event_type"] == "광류급변후보")
    assert flow_event["candidate_class"] == "motion_peak_candidate"
    assert flow_event["supporting_signals"]["optical_flow_peak"] == 2.1


def test_detect_event_candidates_adds_post_event_motion_state_candidates() -> None:
    events = detect_event_candidates(
        [
            {
                "track_id": "P1",
                "type": "보행자",
                "movement_candidate": "직진",
                "positions": [
                    {"time": "00:01.000", "frame_id": "frame_000030", "bbox": [0, 0, 20, 80]},
                    {"time": "00:02.000", "frame_id": "frame_000060", "bbox": [40, 0, 60, 80]},
                    {"time": "00:03.000", "frame_id": "frame_000090", "bbox": [40, 40, 60, 120]},
                    {"time": "00:04.000", "frame_id": "frame_000120", "bbox": [42, 42, 62, 122]},
                    {"time": "00:05.000", "frame_id": "frame_000150", "bbox": [42, 80, 102, 110]},
                ],
            }
        ]
    )

    event_types = {event["event_type"] for event in events}
    assert "사고후정지후보" in event_types
    assert "방향변화후보" in event_types
    assert "낙상후보" in event_types
