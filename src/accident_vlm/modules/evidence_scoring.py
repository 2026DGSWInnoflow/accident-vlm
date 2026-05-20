from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Any


PURPOSE_BASE_SCORES = {
    "traffic_light_crop": 96,
    "sign_crop": 94,
    "event_segment": 92,
    "impact_candidate": 90,
    "actor_crop": 88,
    "pre_impact": 86,
    "tracking_overlay": 84,
    "track_overlay": 84,
    "post_impact": 80,
    "bev_overlay": 78,
    "lane_segmentation_overlay": 74,
    "motion_keyframe": 70,
    "event_window_context": 68,
    "pre_context": 66,
    "regular_context": 58,
    "failure_case": 52,
}

SOURCE_BONUS = {
    "traffic_control": 8,
    "visual_evidence": 6,
    "tracking_overlay": 6,
    "actor_crop": 6,
    "road_geometry": 4,
    "selected_frame": 2,
    "ocr_roi": -8,
}


def score_evidence_image(record: dict[str, Any]) -> dict[str, Any]:
    scored = deepcopy(record)
    purpose = str(scored.get("purpose") or "")
    source = str(scored.get("source") or "")
    score = PURPOSE_BASE_SCORES.get(purpose, 50) + SOURCE_BONUS.get(source, 0)
    reasons = [f"purpose:{purpose or 'unknown'}", f"source:{source or 'unknown'}"]
    if scored.get("frame_id"):
        score += 2
        reasons.append("frame-linked")
    if scored.get("track_id"):
        score += 4
        reasons.append("actor-linked")
    if scored.get("bbox"):
        score += 3
        reasons.append("localized")
    scored["importance_score"] = max(0, min(100, int(score)))
    scored["rank_reason"] = reasons
    return scored


def rank_evidence_images(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        [score_evidence_image(record) for record in records],
        key=lambda record: (
            -int(record.get("importance_score", 0)),
            str(record.get("frame_id") or ""),
            str(record.get("id") or ""),
        ),
    )


def summarize_evidence_images(records: list[dict[str, Any]]) -> dict[str, Any]:
    purpose_counter = Counter(str(record.get("purpose") or "unknown") for record in records)
    source_counter = Counter(str(record.get("source") or "unknown") for record in records)
    top_records = rank_evidence_images(records)[:10]
    return {
        "total_images": len(records),
        "purpose_counts": dict(purpose_counter),
        "source_counts": dict(source_counter),
        "top_evidence": [
            {
                "id": record.get("id"),
                "frame_id": record.get("frame_id"),
                "purpose": record.get("purpose"),
                "importance_score": record.get("importance_score"),
            }
            for record in top_records
        ],
    }
