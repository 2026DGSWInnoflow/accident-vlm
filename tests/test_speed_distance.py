from accident_vlm.modules.speed_distance import choose_speed_estimate, estimate_speed_and_distance


def test_choose_ocr_speed_before_geometry_speed() -> None:
    estimates = [
        {
            "actor_id": "vehicle_1",
            "value": "42 km/h",
            "numeric_kmh": 42,
            "method": "bev_tracking_estimate",
            "confidence": "medium",
        },
        {
            "actor_id": "vehicle_1",
            "value": "47 km/h",
            "numeric_kmh": 47,
            "method": "ocr_overlay",
            "confidence": "high",
        },
        {
            "actor_id": "vehicle_2",
            "value": "60 km/h",
            "method": "metadata",
            "confidence": "high",
        },
    ]

    selected = choose_speed_estimate(estimates, actor_id="vehicle_1")

    assert selected["method"] == "ocr_overlay"
    assert selected["numeric_kmh"] == 47


def test_choose_unknown_when_no_supported_speed() -> None:
    assert choose_speed_estimate([], actor_id="vehicle_1") == {
        "actor_id": "vehicle_1",
        "value": "모름",
        "numeric_kmh": None,
        "range_kmh": None,
        "method": "not_available",
        "confidence": "unknown",
    }


def test_unsupported_only_method_returns_unknown_fallback() -> None:
    assert choose_speed_estimate(
        [
            {
                "actor_id": "vehicle_1",
                "value": "fast",
                "method": "guessed_from_context",
                "confidence": "low",
            }
        ],
        actor_id="vehicle_1",
    ) == {
        "actor_id": "vehicle_1",
        "value": "모름",
        "numeric_kmh": None,
        "range_kmh": None,
        "method": "not_available",
        "confidence": "unknown",
    }


def test_mixed_unsupported_and_supported_selects_supported() -> None:
    selected = choose_speed_estimate(
        [
            {
                "actor_id": "vehicle_1",
                "value": "fast",
                "method": "guessed_from_context",
                "confidence": "low",
            },
            {
                "actor_id": "vehicle_1",
                "value": "38 km/h",
                "numeric_kmh": 38,
                "method": "relative_motion_only",
                "confidence": "low",
            },
        ],
        actor_id="vehicle_1",
    )

    assert selected["method"] == "relative_motion_only"
    assert selected["numeric_kmh"] == 38


def test_malformed_entries_are_ignored_while_valid_later_estimate_is_selected() -> None:
    selected = choose_speed_estimate(
        [
            None,
            "not a dict",
            ["actor_id", "vehicle_1"],
            {
                "actor_id": "vehicle_1",
                "value": "31 km/h",
                "numeric_kmh": 31,
                "method": "bev_tracking_estimate",
                "confidence": "medium",
            },
        ],
        actor_id="vehicle_1",
    )

    assert selected["method"] == "bev_tracking_estimate"
    assert selected["numeric_kmh"] == 31


def test_missing_method_is_ignored() -> None:
    assert choose_speed_estimate(
        [
            {
                "actor_id": "vehicle_1",
                "value": "44 km/h",
                "numeric_kmh": 44,
                "confidence": "medium",
            }
        ],
        actor_id="vehicle_1",
    ) == {
        "actor_id": "vehicle_1",
        "value": "모름",
        "numeric_kmh": None,
        "range_kmh": None,
        "method": "not_available",
        "confidence": "unknown",
    }


def test_tie_breaks_metadata_before_gps_and_obd_regardless_of_input_order() -> None:
    selected = choose_speed_estimate(
        [
            {
                "actor_id": "vehicle_1",
                "value": "52 km/h",
                "numeric_kmh": 52,
                "method": "obd",
                "confidence": "high",
            },
            {
                "actor_id": "vehicle_1",
                "value": "50 km/h",
                "numeric_kmh": 50,
                "method": "gps",
                "confidence": "high",
            },
            {
                "actor_id": "vehicle_1",
                "value": "48 km/h",
                "numeric_kmh": 48,
                "method": "metadata",
                "confidence": "high",
            },
        ],
        actor_id="vehicle_1",
    )

    assert selected["method"] == "metadata"
    assert selected["numeric_kmh"] == 48


def test_extra_fields_are_preserved_on_selected_estimate() -> None:
    selected = choose_speed_estimate(
        [
            {
                "actor_id": "vehicle_1",
                "value": "55 km/h",
                "numeric_kmh": 55,
                "method": "ocr_overlay",
                "confidence": "high",
                "source_frame_id": "frame_000123",
                "evidence": {"bbox": [10, 20, 30, 40]},
            }
        ],
        actor_id="vehicle_1",
    )

    assert selected == {
        "actor_id": "vehicle_1",
        "value": "55 km/h",
        "numeric_kmh": 55,
        "method": "ocr_overlay",
        "confidence": "high",
        "source_frame_id": "frame_000123",
        "evidence": {"bbox": [10, 20, 30, 40]},
    }


def test_estimate_speed_uses_ocr_summary_before_raw_observations() -> None:
    result = estimate_speed_and_distance(
        ocr_observations=[
            {
                "frame_id": "frame_000001",
                "confidence": 0.91,
                "parsed": {"speed_kmh": 51.0},
            }
        ],
        tracks=[],
        road_geometry={},
        ocr_summary={
            "speed": {
                "value": "48km/h",
                "numeric_kmh": 48.0,
                "range_kmh": [47.0, 49.0],
                "confidence": "high",
                "evidence": ["frame_000010", "frame_000020", "frame_000030"],
            }
        },
    )

    selected = result["speed_estimates"][0]
    assert selected["method"] == "ocr_overlay"
    assert selected["source"] == "ocr_summary"
    assert selected["numeric_kmh"] == 48.0
    assert selected["range_kmh"] == [47.0, 49.0]
