from accident_vlm.modules.scene import classify_scene_candidates


def test_classify_scene_candidates_uses_road_marking_candidates_for_rag_context() -> None:
    candidates = classify_scene_candidates(
        selected_frames=[],
        road_geometry={
            "visible_lane_count": {"value": 2, "confidence": "medium", "evidence": ["frame_1"]},
            "road_marking_candidates": [
                {"type": "crosswalk", "frame_id": "frame_1", "confidence": "medium"},
                {"type": "stop_line", "frame_id": "frame_1", "confidence": "medium"},
                {"type": "centerline_yellow", "frame_id": "frame_1", "confidence": "medium"},
            ],
        },
        traffic_control={},
    )

    values = {candidate["value"] for candidate in candidates}
    assert {"횡단보도", "정지선", "중앙선"} <= values
    assert any("scenario_keywords" in candidate for candidate in candidates)
