from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from accident_vlm.schemas.preprocessing import SelectedFrame


def _classify_signal_color(image: np.ndarray) -> tuple[str, float]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    red_mask_1 = cv2.inRange(hsv, (0, 80, 80), (10, 255, 255))
    red_mask_2 = cv2.inRange(hsv, (170, 80, 80), (180, 255, 255))
    yellow_mask = cv2.inRange(hsv, (18, 80, 80), (38, 255, 255))
    green_mask = cv2.inRange(hsv, (40, 60, 60), (90, 255, 255))
    scores = {
        "적색": float((red_mask_1 | red_mask_2).mean()),
        "황색": float(yellow_mask.mean()),
        "녹색": float(green_mask.mean()),
    }
    color, score = max(scores.items(), key=lambda item: item[1])
    if score < 4.0:
        return "확인불가", score
    return color, score


def analyze_traffic_control(
    selected_frames: list[SelectedFrame],
    ocr_observations: list[dict],
    output_dir: Path | None = None,
) -> dict:
    signal_votes: list[tuple[str, float, str, dict | None]] = []
    signal_crops: list[dict] = []
    failure_cases: list[dict] = []
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "failure_cases").mkdir(parents=True, exist_ok=True)

    for frame in selected_frames[:20]:
        if not frame.path:
            continue
        image = cv2.imread(frame.path)
        if image is None:
            continue
        height, width = image.shape[:2]
        for index, candidate in enumerate(_detect_signal_candidates(image)):
            x1, y1, x2, y2 = candidate["bbox"]
            crop = image[y1:y2, x1:x2]
            color, score = _classify_signal_color(crop)
            if color == "확인불가":
                continue
            crop_record = None
            if output_dir:
                crop_path = output_dir / f"{frame.id}_signal_{index:02d}.jpg"
                cv2.imwrite(str(crop_path), crop)
                crop_record = {
                    "id": f"signal_{frame.id}_{index:02d}",
                    "path": str(crop_path),
                    "frame_id": frame.id,
                    "bbox": candidate["bbox"],
                    "color": color,
                    "score": score,
                    "purpose": "traffic_light_crop",
                }
                signal_crops.append(crop_record)
            signal_votes.append((color, score, frame.id, crop_record))

    if signal_votes:
        color_scores: dict[str, float] = {}
        evidence: dict[str, list[str]] = {}
        crops_by_color: dict[str, list[dict]] = {}
        vote_counts: Counter[str] = Counter()
        for color, score, frame_id, crop_record in signal_votes:
            color_scores[color] = color_scores.get(color, 0.0) + score
            vote_counts[color] += 1
            evidence.setdefault(color, []).append(frame_id)
            if crop_record:
                crops_by_color.setdefault(color, []).append(crop_record)
        selected_color, confidence, vote_diagnostics = _choose_signal_vote(color_scores, vote_counts)
        signal = {
            "value": selected_color,
            "visible": True,
            "method": "traffic_light_hsv_crop_temporal_vote",
            "confidence": confidence,
            "evidence": evidence[selected_color],
            "crops": crops_by_color.get(selected_color, []),
            "vote_diagnostics": vote_diagnostics,
            "note": "HSV crop 후보 기반이므로 작은 신호등/역광에서는 confidence를 낮게 유지",
        }
    else:
        signal = {
            "value": "확인불가",
            "visible": False,
            "method": "traffic_light_hsv_crop_temporal_vote",
            "confidence": "unknown",
            "evidence": [],
            "crops": signal_crops,
        }
        if output_dir and selected_frames:
            first_frame = next((frame for frame in selected_frames if frame.path), None)
            if first_frame:
                image = cv2.imread(first_frame.path)
                if image is not None:
                    height = image.shape[0]
                    failure_path = output_dir / "failure_cases" / f"{first_frame.id}_traffic_light_not_detected.jpg"
                    cv2.imwrite(str(failure_path), image[: int(height * 0.45), :])
                    failure_cases.append(
                        {
                            "id": f"failure_{first_frame.id}_traffic_light",
                            "path": str(failure_path),
                            "frame_id": first_frame.id,
                            "reason": "traffic_light_not_detected",
                            "purpose": "failure_case",
                        }
                    )

    signs = []
    for observation in ocr_observations:
        text = observation.get("text", "")
        speed_limit_match = re.search(r"(?:제한속도|속도제한|speed\s*limit)?\s*(30|40|50|60|70|80|90|100|110)", text, re.IGNORECASE)
        if speed_limit_match:
            signs.append(
                {
                    "value": f"제한속도 {speed_limit_match.group(1)}",
                    "numeric_kmh": int(speed_limit_match.group(1)),
                    "raw_text": text,
                    "confidence": observation.get("confidence", 0.0),
                    "source": observation.get("source", "ocr"),
                    "evidence": [observation.get("frame_id")],
                }
            )
        if "STOP" in text.upper() or "정지" in text:
            signs.append(
                {
                    "value": "일시정지",
                    "raw_text": text,
                    "confidence": observation.get("confidence", 0.0),
                    "source": observation.get("source", "ocr"),
                    "evidence": [observation.get("frame_id")],
                }
            )

    sign_votes = _vote_signs(signs)
    return {
        "signal": signal,
        "signs": signs,
        "sign_votes": sign_votes,
        "failure_cases": failure_cases,
        "crosswalk": {"visible": False, "confidence": "unknown", "method": "not_implemented"},
    }


def _vote_signs(signs: list[dict]) -> list[dict]:
    counter = Counter(sign["value"] for sign in signs)
    voted: list[dict] = []
    for value, count in counter.most_common():
        evidence = [
            evidence_id
            for sign in signs
            if sign["value"] == value
            for evidence_id in sign.get("evidence", [])
            if evidence_id
        ]
        confidences = [
            float(sign.get("confidence") or 0.0)
            for sign in signs
            if sign["value"] == value
        ]
        voted.append(
            {
                "value": value,
                "vote_count": count,
                "confidence": "high" if count >= 3 else "medium" if count >= 2 else "low",
                "mean_ocr_confidence": round(float(np.mean(confidences)), 3) if confidences else 0.0,
                "evidence": evidence,
            }
        )
    return voted


def _choose_signal_vote(
    color_scores: dict[str, float],
    vote_counts: Counter[str],
) -> tuple[str, str, dict]:
    ordered = sorted(color_scores.items(), key=lambda item: item[1], reverse=True)
    selected_color, selected_score = ordered[0]
    runner_up_score = ordered[1][1] if len(ordered) > 1 else 0.0
    margin = selected_score - runner_up_score
    vote_count = vote_counts[selected_color]
    confidence = "low"
    if vote_count >= 3 and margin >= 20:
        confidence = "high"
    elif vote_count >= 2 and margin >= 8:
        confidence = "medium"
    return (
        selected_color,
        confidence,
        {
            "score_by_color": {color: round(score, 3) for color, score in color_scores.items()},
            "vote_count_by_color": dict(vote_counts),
            "winner_margin": round(margin, 3),
        },
    )


def _detect_signal_candidates(image: np.ndarray) -> list[dict]:
    height, width = image.shape[:2]
    upper = image[: int(height * 0.45), :]
    hsv = cv2.cvtColor(upper, cv2.COLOR_BGR2HSV)
    red_mask = cv2.inRange(hsv, (0, 80, 80), (10, 255, 255)) | cv2.inRange(
        hsv, (170, 80, 80), (180, 255, 255)
    )
    yellow_mask = cv2.inRange(hsv, (18, 80, 80), (38, 255, 255))
    green_mask = cv2.inRange(hsv, (40, 60, 60), (90, 255, 255))
    mask = red_mask | yellow_mask | green_mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[dict] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area < 20 or area > width * height * 0.05:
            continue
        pad = max(6, int(max(w, h) * 0.75))
        candidates.append(
            {
                "bbox": [
                    max(0, x - pad),
                    max(0, y - pad),
                    min(width, x + w + pad),
                    min(height, y + h + pad),
                ]
            }
        )
    return candidates
