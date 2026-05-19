from __future__ import annotations

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


def analyze_traffic_control(selected_frames: list[SelectedFrame], ocr_observations: list[dict]) -> dict:
    signal_votes: list[tuple[str, float, str]] = []
    for frame in selected_frames[:20]:
        if not frame.path:
            continue
        image = cv2.imread(frame.path)
        if image is None:
            continue
        height, width = image.shape[:2]
        upper = image[: int(height * 0.45), :]
        # Traffic lights are usually small bright colored blobs in the upper scene.
        color, score = _classify_signal_color(upper)
        if color != "확인불가":
            signal_votes.append((color, score, frame.id))

    if signal_votes:
        color_scores: dict[str, float] = {}
        evidence: dict[str, list[str]] = {}
        for color, score, frame_id in signal_votes:
            color_scores[color] = color_scores.get(color, 0.0) + score
            evidence.setdefault(color, []).append(frame_id)
        selected_color = max(color_scores, key=color_scores.get)
        signal = {
            "value": selected_color,
            "visible": True,
            "method": "hsv_temporal_vote",
            "confidence": "low",
            "evidence": evidence[selected_color],
            "note": "HSV 색상 후보이므로 신호등 crop classifier로 후속 검증 필요",
        }
    else:
        signal = {
            "value": "확인불가",
            "visible": False,
            "method": "hsv_temporal_vote",
            "confidence": "unknown",
            "evidence": [],
        }

    signs = []
    for observation in ocr_observations:
        text = observation.get("text", "")
        if "30" in text or "50" in text or "60" in text:
            signs.append(
                {
                    "value": "제한속도",
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

    return {
        "signal": signal,
        "signs": signs,
        "crosswalk": {"visible": False, "confidence": "unknown", "method": "not_implemented"},
    }
