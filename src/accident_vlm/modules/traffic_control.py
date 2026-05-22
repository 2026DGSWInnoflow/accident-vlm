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


def _classify_signal_state(image: np.ndarray) -> tuple[str, str, str, float, dict]:
    color, score = _classify_signal_color(image)
    shape = _classify_signal_shape(image, color)
    if color == "녹색" and shape == "left_arrow":
        state = "좌회전녹색"
    else:
        state = color
    return state, color, shape, score, {"color": color, "shape": shape, "score": round(score, 3)}


def analyze_traffic_control(
    selected_frames: list[SelectedFrame],
    ocr_observations: list[dict],
    output_dir: Path | None = None,
) -> dict:
    signal_votes: list[tuple[str, float, str, dict | None, str, dict]] = []
    signal_crops: list[dict] = []
    sign_crops: list[dict] = []
    off_signal_crops: list[dict] = []
    frame_signal_statuses: list[dict] = []
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
        frame_has_lit_signal = False
        frame_has_off_signal = False
        height, width = image.shape[:2]
        for index, candidate in enumerate(_detect_signal_candidates(image)):
            x1, y1, x2, y2 = candidate["bbox"]
            crop = image[y1:y2, x1:x2]
            state, color, shape, score, diagnostics = _classify_signal_state(crop)
            if state == "확인불가":
                _save_failure_case(output_dir, failure_cases, frame.id, crop, "traffic_light_candidate_unclassified", index)
                continue
            frame_has_lit_signal = True
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
                    "shape": shape,
                    "state": state,
                    "score": score,
                    "diagnostics": diagnostics,
                    "purpose": "traffic_light_crop",
                }
                signal_crops.append(crop_record)
            signal_votes.append((state, score, frame.id, crop_record, shape, diagnostics))
        if not frame_has_lit_signal:
            for index, candidate in enumerate(_detect_off_signal_candidates(image)):
                frame_has_off_signal = True
                if output_dir:
                    x1, y1, x2, y2 = candidate["bbox"]
                    crop = image[y1:y2, x1:x2]
                    crop_path = output_dir / f"{frame.id}_signal_off_{index:02d}.jpg"
                    cv2.imwrite(str(crop_path), crop)
                    off_signal_crops.append(
                        {
                            "id": f"signal_off_{frame.id}_{index:02d}",
                            "path": str(crop_path),
                            "frame_id": frame.id,
                            "bbox": candidate["bbox"],
                            "color": "꺼짐",
                            "shape": "unknown",
                            "state": "꺼짐",
                            "score": candidate["score"],
                            "purpose": "traffic_light_crop",
                            "diagnostics": candidate,
                        }
                    )
        frame_signal_statuses.append(
            {
                "frame_id": frame.id,
                "status": "lit" if frame_has_lit_signal else "off_candidate" if frame_has_off_signal else "not_detected",
            }
        )
        for index, candidate in enumerate(_detect_sign_candidates(image)):
            if output_dir:
                x1, y1, x2, y2 = candidate["bbox"]
                crop = image[y1:y2, x1:x2]
                crop_path = output_dir / f"{frame.id}_sign_{index:02d}.jpg"
                cv2.imwrite(str(crop_path), crop)
                sign_crops.append(
                    {
                        "id": f"sign_{frame.id}_{index:02d}",
                        "path": str(crop_path),
                        "frame_id": frame.id,
                        "bbox": candidate["bbox"],
                        "purpose": "sign_crop",
                        "detector": candidate["detector"],
                        "confidence": candidate["confidence"],
                    }
                )

    if signal_votes:
        color_scores: dict[str, float] = {}
        evidence: dict[str, list[str]] = {}
        crops_by_color: dict[str, list[dict]] = {}
        vote_counts: Counter[str] = Counter()
        shapes_by_state: dict[str, list[str]] = {}
        diagnostics_by_state: dict[str, list[dict]] = {}
        for state, score, frame_id, crop_record, shape, diagnostics in signal_votes:
            color_scores[state] = color_scores.get(state, 0.0) + score
            vote_counts[state] += 1
            evidence.setdefault(state, []).append(frame_id)
            shapes_by_state.setdefault(state, []).append(shape)
            diagnostics_by_state.setdefault(state, []).append(diagnostics)
            if crop_record:
                crops_by_color.setdefault(state, []).append(crop_record)
        selected_color, confidence, vote_diagnostics = _choose_signal_vote(color_scores, vote_counts)
        selected_evidence = _combined_signal_items(selected_color, evidence)
        selected_crops = _combined_signal_items(selected_color, crops_by_color)
        selected_diagnostics = _combined_signal_items(selected_color, diagnostics_by_state)
        signal = {
            "value": selected_color,
            "visible": True,
            "method": "traffic_light_hsv_crop_temporal_vote",
            "confidence": confidence,
            "evidence": selected_evidence,
            "crops": selected_crops,
            "shape": _selected_shape(selected_color, shapes_by_state),
            "classifier": "hsv_color_plus_arrow_shape",
            "signal_head_crops": selected_crops,
            "classification_diagnostics": selected_diagnostics,
            "vote_diagnostics": vote_diagnostics,
            "note": "HSV crop 후보 기반이므로 작은 신호등/역광에서는 confidence를 낮게 유지",
        }
        flashing = _flashing_signal_state(selected_color, selected_evidence, frame_signal_statuses)
        if flashing:
            signal.update(flashing)
    else:
        if off_signal_crops:
            signal = {
                "value": "꺼짐",
                "visible": True,
                "method": "traffic_light_head_off_candidate",
                "confidence": "low" if len(off_signal_crops) == 1 else "medium",
                "evidence": [item["frame_id"] for item in off_signal_crops],
                "crops": off_signal_crops,
                "signal_head_crops": off_signal_crops,
                "frame_sequence": _frame_sequence_summary(frame_signal_statuses),
                "note": "불빛 색상은 없지만 어두운 신호 헤드 후보가 확인됨",
            }
        else:
            signal = {
                "value": "확인불가",
                "visible": False,
                "method": "traffic_light_hsv_crop_temporal_vote",
                "confidence": "unknown",
                "evidence": [],
                "crops": signal_crops,
                "frame_sequence": _frame_sequence_summary(frame_signal_statuses),
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
        "sign_crops": sign_crops,
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
                "confidence_margin": count - (counter.most_common(2)[1][1] if len(counter) > 1 else 0),
                "evidence": evidence,
            }
        )
    return voted


def _choose_signal_vote(
    color_scores: dict[str, float],
    vote_counts: Counter[str],
) -> tuple[str, str, dict]:
    ordered = sorted(color_scores.items(), key=lambda item: item[1], reverse=True)
    if "적색" in color_scores and "좌회전녹색" in color_scores:
        selected_score = color_scores["적색"] + color_scores["좌회전녹색"]
        vote_count = vote_counts["적색"] + vote_counts["좌회전녹색"]
        confidence = "medium" if vote_count >= 2 else "low"
        return (
            "적색+좌회전",
            confidence,
            {
                "score_by_color": {color: round(score, 3) for color, score in color_scores.items()},
                "vote_count_by_color": dict(vote_counts),
                "winner_margin": round(selected_score - max(color_scores.values()), 3),
            },
        )
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


def _detect_off_signal_candidates(image: np.ndarray) -> list[dict]:
    height, width = image.shape[:2]
    upper = image[: int(height * 0.45), :]
    gray = cv2.cvtColor(upper, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(
        cv2.medianBlur(gray, 5),
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=20,
        param1=40,
        param2=12,
        minRadius=5,
        maxRadius=max(8, min(width, height) // 12),
    )
    candidates: list[dict] = []
    circle_values = [] if circles is None else circles[0, :]
    for x, y, radius in circle_values:
        x_int, y_int, radius_int = int(round(x)), int(round(y)), int(round(radius))
        x1 = max(0, x_int - radius_int - 6)
        y1 = max(0, y_int - radius_int - 6)
        x2 = min(width, x_int + radius_int + 6)
        y2 = min(height, y_int + radius_int + 6)
        crop = upper[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        saturated_ratio = float((hsv[:, :, 1] > 80).mean())
        brightness = float(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).mean())
        if saturated_ratio < 0.08 and 10 <= brightness <= 120:
            candidates.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "detector": "hough_dark_signal_head",
                    "score": round((1.0 - saturated_ratio) * 100, 3),
                    "brightness": round(brightness, 3),
                }
            )
    dark_mask = cv2.inRange(gray, 25, 130)
    contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area < 80 or area > width * height * 0.08:
            continue
        circularity = min(w, h) / max(w, h)
        if circularity < 0.55:
            continue
        x1 = max(0, x - 6)
        y1 = max(0, y - 6)
        x2 = min(width, x + w + 6)
        y2 = min(height, y + h + 6)
        if any(abs(candidate["bbox"][0] - x1) < 8 and abs(candidate["bbox"][1] - y1) < 8 for candidate in candidates):
            continue
        crop = upper[y1:y2, x1:x2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        saturated_ratio = float((hsv[:, :, 1] > 80).mean())
        brightness = float(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).mean())
        if saturated_ratio < 0.12 and 10 <= brightness <= 130:
            candidates.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "detector": "dark_circular_signal_head",
                    "score": round((1.0 - saturated_ratio) * 90, 3),
                    "brightness": round(brightness, 3),
                }
            )
    return candidates


def _classify_signal_shape(image: np.ndarray, color: str) -> str:
    if color != "녹색":
        return "circle"
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, (40, 60, 60), (90, 255, 255))
    moments = cv2.moments(green_mask)
    if moments["m00"] <= 0:
        return "unknown"
    height, width = green_mask.shape[:2]
    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return "unknown"
    contour = max(contours, key=cv2.contourArea)
    x, _y, w, h = cv2.boundingRect(contour)
    centroid_x = moments["m10"] / moments["m00"]
    if w > h * 1.15 and centroid_x < width * 0.58 and x < width * 0.45:
        return "left_arrow"
    return "circle"


def _selected_shape(selected_color: str, shapes_by_state: dict[str, list[str]]) -> str:
    if selected_color == "적색+좌회전":
        return "left_arrow"
    shapes = shapes_by_state.get(selected_color, [])
    return Counter(shapes).most_common(1)[0][0] if shapes else "unknown"


def _combined_signal_items(selected_color: str, items_by_state: dict[str, list]) -> list:
    if selected_color == "적색+좌회전":
        return [*items_by_state.get("적색", []), *items_by_state.get("좌회전녹색", [])]
    return items_by_state.get(selected_color, [])


def _flashing_signal_state(selected_color: str, evidence: list[str], statuses: list[dict]) -> dict | None:
    if len(statuses) < 3:
        return None
    missing_count = sum(1 for item in statuses if item.get("status") != "lit")
    lit_count = sum(1 for item in statuses if item.get("status") == "lit")
    if lit_count >= 2 and missing_count >= 1:
        return {
            "value": "점멸",
            "base_color": selected_color,
            "confidence": "medium" if lit_count >= 2 else "low",
            "evidence": evidence,
            "frame_sequence": _frame_sequence_summary(statuses),
        }
    return None


def _frame_sequence_summary(statuses: list[dict]) -> dict:
    return {
        "frames": statuses,
        "lit_frame_count": sum(1 for item in statuses if item.get("status") == "lit"),
        "off_candidate_count": sum(1 for item in statuses if item.get("status") == "off_candidate"),
        "missing_frame_count": sum(1 for item in statuses if item.get("status") == "not_detected"),
    }


def _detect_sign_candidates(image: np.ndarray) -> list[dict]:
    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    red_mask = cv2.inRange(hsv, (0, 80, 80), (10, 255, 255)) | cv2.inRange(
        hsv, (170, 80, 80), (180, 255, 255)
    )
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[dict] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area < 80 or area > width * height * 0.08:
            continue
        circularity = min(w, h) / max(w, h)
        if circularity < 0.65 or y < height * 0.4:
            continue
        pad = max(6, int(max(w, h) * 0.25))
        candidates.append(
            {
                "bbox": [max(0, x - pad), max(0, y - pad), min(width, x + w + pad), min(height, y + h + pad)],
                "detector": "opencv_red_circular_sign_candidate",
                "confidence": "medium",
            }
        )
    return candidates


def _save_failure_case(
    output_dir: Path | None,
    failure_cases: list[dict],
    frame_id: str,
    image: np.ndarray,
    reason: str,
    index: int,
) -> None:
    if not output_dir:
        return
    failure_path = output_dir / "failure_cases" / f"{frame_id}_{reason}_{index:02d}.jpg"
    cv2.imwrite(str(failure_path), image)
    failure_cases.append(
        {
            "id": f"failure_{frame_id}_{reason}_{index:02d}",
            "path": str(failure_path),
            "frame_id": frame_id,
            "reason": reason,
            "purpose": "failure_case",
        }
    )
