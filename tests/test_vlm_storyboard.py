from accident_vlm.modules.vlm_storyboard import build_vlm_storyboard


def test_build_vlm_storyboard_orders_temporal_frames_and_keeps_details_linked():
    evidence_package = {
        "evidence_images": [
            {
                "id": "impact",
                "path": "/tmp/impact.jpg",
                "frame_id": "frame_003",
                "time": "00:03.000",
                "purpose": "impact_candidate",
                "importance_score": 99,
            },
            {
                "id": "pre",
                "path": "/tmp/pre.jpg",
                "frame_id": "frame_001",
                "time": "00:01.000",
                "purpose": "pre_impact",
                "importance_score": 80,
            },
            {
                "id": "post",
                "path": "/tmp/post.jpg",
                "frame_id": "frame_004",
                "time": "00:04.000",
                "purpose": "post_impact",
                "importance_score": 79,
            },
            {
                "id": "crop_person",
                "path": "/tmp/crop.jpg",
                "frame_id": "frame_003",
                "time": "00:03.000",
                "purpose": "actor_crop",
                "track_id": "T2",
                "type": "보행자",
                "importance_score": 95,
            },
        ],
        "precomputed_facts": {
            "tracks": [
                {
                    "track_id": "T1",
                    "type": "승용차",
                    "positions": [{"frame_id": "frame_003"}],
                },
                {
                    "track_id": "T2",
                    "type": "보행자",
                    "positions": [{"frame_id": "frame_003"}],
                },
            ],
            "event_candidates": [{"time": "00:03.000", "event_type": "proximity", "event_score": 88}],
        },
    }

    storyboard = build_vlm_storyboard(evidence_package, max_items=4)

    assert [item["id"] for item in storyboard] == ["pre", "impact", "crop_person", "post"]
    assert [item["slot"] for item in storyboard] == [1, 2, 3, 4]
    crop = storyboard[2]
    assert crop["phase"] == "detail"
    assert crop["linked_actor_ids"] == ["T2"]
    assert crop["precomputed_hints"]["visible_actor_candidates"] == ["승용차", "보행자"]
    assert crop["precomputed_hints"]["nearby_event_candidates"][0]["event_type"] == "proximity"


def test_build_vlm_storyboard_uses_phase_quota_without_overfitting_to_pedestrians():
    evidence_package = {
        "evidence_images": [
            {
                "id": f"detail_{index}",
                "path": f"/tmp/detail_{index}.jpg",
                "frame_id": "frame_003",
                "time": "00:03.000",
                "purpose": "actor_crop",
                "track_id": f"M{index}",
                "type": "오토바이",
                "importance_score": 100 - index,
            }
            for index in range(10)
        ]
        + [
            {
                "id": "scene",
                "path": "/tmp/scene.jpg",
                "frame_id": "frame_001",
                "time": "00:01.000",
                "purpose": "regular_context",
                "importance_score": 40,
            },
            {
                "id": "impact",
                "path": "/tmp/impact.jpg",
                "frame_id": "frame_003",
                "time": "00:03.000",
                "purpose": "impact_candidate",
                "importance_score": 90,
            },
            {
                "id": "post",
                "path": "/tmp/post.jpg",
                "frame_id": "frame_004",
                "time": "00:04.000",
                "purpose": "post_impact",
                "importance_score": 88,
            },
        ],
        "precomputed_facts": {"tracks": [], "event_candidates": []},
    }

    storyboard = build_vlm_storyboard(evidence_package, max_items=6)
    phases = [item["phase"] for item in storyboard]

    assert "scene_context" in phases
    assert "impact_candidate" in phases
    assert "post_event" in phases
    assert phases.count("detail") <= 3
    assert any(item["precomputed_hints"].get("actor_type_hint") == "오토바이" for item in storyboard)


def test_build_vlm_storyboard_preserves_insurance_context_against_actor_crops():
    evidence_package = {
        "evidence_images": [
            {
                "id": f"actor_{index}",
                "path": f"/tmp/actor_{index}.jpg",
                "frame_id": "frame_003",
                "time": "00:03.000",
                "purpose": "actor_crop",
                "importance_score": 100 - index,
            }
            for index in range(8)
        ]
        + [
            {
                "id": "signal",
                "path": "/tmp/signal.jpg",
                "frame_id": "frame_002",
                "time": "00:02.000",
                "purpose": "traffic_light_crop",
                "importance_score": 20,
            },
            {
                "id": "lane",
                "path": "/tmp/lane.jpg",
                "frame_id": "frame_001",
                "time": "00:01.000",
                "purpose": "lane_segmentation_overlay",
                "importance_score": 10,
            },
        ],
        "precomputed_facts": {},
    }

    storyboard = build_vlm_storyboard(evidence_package, max_items=5)

    insurance_items = [item for item in storyboard if item["phase"] == "insurance_context"]
    assert {item["id"] for item in insurance_items} == {"lane", "signal"}
    assert all("보험" in item["role"] for item in insurance_items)


def test_build_vlm_storyboard_does_not_front_load_untimed_signal_crops():
    evidence_package = {
        "evidence_images": [
            {
                "id": f"signal_{index}",
                "path": f"/tmp/signal_{index}.jpg",
                "purpose": "traffic_light_crop",
                "importance_score": 100 - index,
            }
            for index in range(12)
        ]
        + [
            {
                "id": "scene",
                "path": "/tmp/scene.jpg",
                "frame_id": "frame_001",
                "time": "00:01.000",
                "purpose": "event_window_context",
                "importance_score": 70,
            },
            {
                "id": "pre",
                "path": "/tmp/pre.jpg",
                "frame_id": "frame_002",
                "time": "00:02.000",
                "purpose": "pre_impact",
                "importance_score": 80,
            },
            {
                "id": "impact",
                "path": "/tmp/impact.jpg",
                "frame_id": "frame_003",
                "time": "00:03.000",
                "purpose": "impact_candidate",
                "importance_score": 90,
            },
            {
                "id": "post",
                "path": "/tmp/post.jpg",
                "frame_id": "frame_004",
                "time": "00:04.000",
                "purpose": "post_impact",
                "importance_score": 75,
            },
        ],
        "precomputed_facts": {},
    }

    storyboard = build_vlm_storyboard(evidence_package, max_items=10)

    assert [item["id"] for item in storyboard[:4]] == ["scene", "pre", "impact", "post"]
    assert sum(item["purpose"] == "traffic_light_crop" for item in storyboard) <= 3
