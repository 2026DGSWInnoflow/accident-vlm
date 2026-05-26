import cv2
import numpy as np

from accident_vlm.modules.evidence_scoring import (
    assess_evidence_image_quality,
    rank_evidence_images,
    score_evidence_image,
    summarize_evidence_images,
)


def test_score_evidence_image_adds_quality_metrics_and_penalty(tmp_path):
    image_path = tmp_path / "dark_blurry.jpg"
    image = np.full((50, 80, 3), 8, dtype=np.uint8)
    cv2.imwrite(str(image_path), image)

    scored = score_evidence_image(
        {
            "id": "frame_dark",
            "path": str(image_path),
            "purpose": "regular_context",
            "source": "selected_frame",
        }
    )

    assert scored["evidence_quality"]["brightness"] == "dark"
    assert scored["evidence_quality"]["analysis_reliability"] in {"low", "medium"}
    assert scored["quality_confidence"] in {"low", "medium"}
    assert "quality_penalty" in scored["rank_reason"]


def test_rank_evidence_images_keeps_quality_metrics(tmp_path):
    bright_path = tmp_path / "bright.jpg"
    cv2.imwrite(str(bright_path), np.full((50, 80, 3), 180, dtype=np.uint8))

    ranked = rank_evidence_images(
        [
            {
                "id": "signal",
                "path": str(bright_path),
                "purpose": "traffic_light_crop",
                "source": "traffic_control",
            }
        ]
    )

    assert ranked[0]["evidence_quality"]["brightness"] == "normal"


def test_assess_evidence_image_quality_avoids_numpy_percentile(monkeypatch, tmp_path):
    image_path = tmp_path / "contrast.jpg"
    image = np.zeros((50, 80, 3), dtype=np.uint8)
    image[:, 40:] = 200
    cv2.imwrite(str(image_path), image)
    monkeypatch.setattr(
        "accident_vlm.modules.evidence_scoring.np.percentile",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("uint8 contrast should use histogram percentiles")
        ),
    )

    quality = assess_evidence_image_quality(image_path)

    assert quality["contrast_score"] > 0


def test_summarize_evidence_images_reuses_existing_scores(monkeypatch):
    def fail_score(record):
        raise AssertionError("pre-ranked records should not be scored again")

    monkeypatch.setattr("accident_vlm.modules.evidence_scoring.score_evidence_image", fail_score)

    summary = summarize_evidence_images(
        [
            {
                "id": "low",
                "purpose": "regular_context",
                "source": "selected_frame",
                "importance_score": 10,
            },
            {
                "id": "high",
                "purpose": "actor_crop",
                "source": "visual_evidence",
                "importance_score": 90,
            },
        ]
    )

    assert [item["id"] for item in summary["top_evidence"]] == ["high", "low"]
