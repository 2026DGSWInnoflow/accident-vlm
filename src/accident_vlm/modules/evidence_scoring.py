from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from accident_vlm.modules.image_metrics import gray_percentile_range
from accident_vlm.modules.image_io import read_cv_image


PURPOSE_BASE_SCORES = {
    "traffic_light_crop": 96,
    "sign_crop": 94,
    "event_segment": 92,
    "event_candidate_overlay": 92,
    "event_scan_impact_candidate": 91,
    "impact_candidate": 90,
    "actor_crop": 88,
    "pre_impact": 86,
    "event_scan_pre_impact": 86,
    "tracking_overlay": 84,
    "track_overlay": 84,
    "post_impact": 80,
    "event_scan_post_impact": 80,
    "bev_overlay": 78,
    "lane_segmentation_overlay": 74,
    "motion_keyframe": 70,
    "event_window_context": 68,
    "pre_context": 66,
    "event_scan_pre_context": 66,
    "frame_selection_contact_sheet": 64,
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
    quality = assess_evidence_image_quality(scored.get("path"))
    if quality:
        scored["evidence_quality"] = quality
        scored["quality_confidence"] = quality["analysis_reliability"]
        if quality["analysis_reliability"] == "low":
            score -= 12
            reasons.append("quality_penalty")
        elif quality["analysis_reliability"] == "medium":
            score -= 4
            reasons.append("quality_caution")
    scored["importance_score"] = max(0, min(100, int(score)))
    scored["rank_reason"] = reasons
    return scored


def assess_evidence_image_quality(path: Any) -> dict[str, Any]:
    if not path:
        return {}
    image_path = Path(str(path))
    if not image_path.exists():
        return {}
    image = read_cv_image(image_path)
    if image is None:
        return {}
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    metric_gray = _metric_gray(gray)
    blur_score = float(cv2.Laplacian(metric_gray, cv2.CV_64F).var())
    brightness_score = float(metric_gray.mean())
    noise_score = float(metric_gray.std())
    glare_ratio = float((metric_gray >= 245).mean())
    contrast_score = gray_percentile_range(metric_gray)
    blur = "high" if blur_score < 80 else "medium" if blur_score < 300 else "low"
    brightness = "dark" if brightness_score < 65 else "overexposed" if brightness_score > 205 else "normal"
    noise = "high" if noise_score > 75 else "medium" if noise_score > 35 else "low"
    weak_factors = [
        blur == "high",
        brightness != "normal",
        noise == "high",
        glare_ratio > 0.08,
        contrast_score < 35,
    ]
    reliability = "low" if sum(weak_factors) >= 2 else "medium" if any(weak_factors) else "high"
    return {
        "blur": blur,
        "brightness": brightness,
        "night_noise": noise,
        "analysis_reliability": reliability,
        "blur_score": round(blur_score, 3),
        "brightness_score": round(brightness_score, 3),
        "noise_score": round(noise_score, 3),
        "glare_ratio": round(glare_ratio, 5),
        "contrast_score": round(contrast_score, 3),
    }


def _metric_gray(gray, max_side: int = 96):
    height, width = gray.shape[:2]
    longest_side = max(height, width)
    if longest_side <= max_side:
        return gray
    scale = max_side / longest_side
    return cv2.resize(
        gray,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )


def rank_evidence_images(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        [score_evidence_image(record) for record in records],
        key=_evidence_sort_key,
    )


def summarize_evidence_images(records: list[dict[str, Any]]) -> dict[str, Any]:
    purpose_counter = Counter(str(record.get("purpose") or "unknown") for record in records)
    source_counter = Counter(str(record.get("source") or "unknown") for record in records)
    quality_counter = Counter(
        str(record.get("evidence_quality", {}).get("analysis_reliability") or record.get("quality_confidence") or "unknown")
        for record in records
    )
    top_records = (
        sorted((deepcopy(record) for record in records), key=_evidence_sort_key)[:10]
        if all("importance_score" in record for record in records)
        else rank_evidence_images(records)[:10]
    )
    return {
        "total_images": len(records),
        "purpose_counts": dict(purpose_counter),
        "source_counts": dict(source_counter),
        "quality_counts": dict(quality_counter),
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


def _evidence_sort_key(record: dict[str, Any]) -> tuple[int, str, str]:
    return (
        -int(record.get("importance_score", 0)),
        str(record.get("frame_id") or ""),
        str(record.get("id") or ""),
    )
