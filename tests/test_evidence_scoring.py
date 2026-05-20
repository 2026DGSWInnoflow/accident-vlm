from accident_vlm.modules.evidence_scoring import rank_evidence_images, score_evidence_image, summarize_evidence_images


def test_score_evidence_image_prioritizes_localized_signal_crop() -> None:
    scored = score_evidence_image(
        {
            "id": "signal_1",
            "purpose": "traffic_light_crop",
            "source": "traffic_control",
            "frame_id": "frame_000010",
            "bbox": [1, 2, 3, 4],
        }
    )

    assert scored["importance_score"] == 100
    assert "localized" in scored["rank_reason"]


def test_rank_evidence_images_orders_by_importance() -> None:
    ranked = rank_evidence_images(
        [
            {"id": "regular", "purpose": "regular_context", "source": "selected_frame"},
            {"id": "actor", "purpose": "actor_crop", "source": "visual_evidence"},
            {"id": "signal", "purpose": "traffic_light_crop", "source": "traffic_control"},
        ]
    )

    assert [item["id"] for item in ranked] == ["signal", "actor", "regular"]


def test_summarize_evidence_images_counts_sources_and_top_evidence() -> None:
    summary = summarize_evidence_images(
        [
            {"id": "signal", "purpose": "traffic_light_crop", "source": "traffic_control"},
            {"id": "frame", "purpose": "regular_context", "source": "selected_frame"},
        ]
    )

    assert summary["total_images"] == 2
    assert summary["purpose_counts"]["traffic_light_crop"] == 1
    assert summary["source_counts"]["selected_frame"] == 1
    assert summary["top_evidence"][0]["id"] == "signal"
