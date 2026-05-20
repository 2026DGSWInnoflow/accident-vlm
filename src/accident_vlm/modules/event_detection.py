from __future__ import annotations

from itertools import combinations


def _bbox_iou(first: list[int], second: list[int]) -> float:
    x_left = max(first[0], second[0])
    y_top = max(first[1], second[1])
    x_right = min(first[2], second[2])
    y_bottom = min(first[3], second[3])
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0
    intersection = (x_right - x_left) * (y_bottom - y_top)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def detect_event_candidates(
    tracks: list[dict],
    speed_and_distance: dict | None = None,
    input_quality: dict | None = None,
) -> list[dict]:
    events: list[dict] = []
    for track in tracks:
        movement = track.get("movement_candidate")
        if movement in {"차로변경_좌", "차로변경_우", "끼어들기"} and track.get("positions"):
            first_position = track["positions"][0]
            events.append(
                {
                    "time": first_position.get("time"),
                    "event_type": "차로변경시작",
                    "actors": [track.get("track_id")],
                    "confidence": track.get("confidence", "medium"),
                    "signals": ["track_lateral_motion"],
                    "evidence": [first_position.get("frame_id")],
                }
            )
        approach_event = _detect_distance_drop(track)
        if approach_event:
            events.append(approach_event)
        for area_event in _detect_rapid_area_changes(track):
            events.append(area_event)

    positions_by_time: dict[str, list[tuple[str, list[int], str]]] = {}
    for track in tracks:
        for position in track.get("positions", []):
            positions_by_time.setdefault(position.get("time", ""), []).append(
                (track.get("track_id", "unknown"), position.get("bbox", []), position.get("frame_id", ""))
            )

    for time, positions in positions_by_time.items():
        for first, second in combinations(positions, 2):
            if not first[1] or not second[1]:
                continue
            iou = _bbox_iou(first[1], second[1])
            if iou >= 0.05:
                events.append(
                    _with_event_score({
                        "time": time,
                        "event_type": "접촉",
                        "actors": [first[0], second[0]],
                        "confidence": "medium" if iou >= 0.12 else "low",
                        "signals": ["bbox_overlap"],
                        "iou": round(iou, 4),
                        "evidence": [first[2], second[2]],
                    })
                )

    if speed_and_distance:
        sudden_stop = _detect_sudden_stop(speed_and_distance)
        if sudden_stop:
            events.append(sudden_stop)
        for item in speed_and_distance.get("relative_motion", []):
            if item.get("relative_speed_trend") == "접근":
                events.append(
                    _with_event_score({
                        "time": item.get("time", "확인불가"),
                        "event_type": "접근",
                        "actors": [item.get("actor_id")],
                        "confidence": item.get("confidence", "low"),
                        "signals": ["relative_motion"],
                    })
                )
    if input_quality:
        shake_score = input_quality.get("camera_shake_score", {})
        if isinstance(shake_score, dict) and float(shake_score.get("value") or 0.0) >= 20:
            events.append(
                _with_event_score({
                    "time": shake_score.get("time", "확인불가"),
                    "event_type": "충격후보",
                    "actors": [],
                    "confidence": "low",
                    "signals": ["camera_shake"],
                    "evidence": shake_score.get("evidence", []),
                })
            )
    return sorted(
        [_with_event_score(event) for event in events],
        key=lambda event: (-float(event.get("event_score", 0.0)), event.get("time") or ""),
    )


def _bbox_area(bbox: list[int]) -> float:
    if len(bbox) != 4:
        return 0.0
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def _detect_distance_drop(track: dict) -> dict | None:
    positions = track.get("positions", [])
    if len(positions) < 2:
        return None
    first_area = _bbox_area(positions[0].get("bbox", []))
    last_area = _bbox_area(positions[-1].get("bbox", []))
    if first_area <= 0 or last_area < first_area * 2.5:
        return None
    return _with_event_score({
        "time": positions[-1].get("time"),
        "event_type": "급접근",
        "actors": [track.get("track_id")],
        "confidence": "medium" if last_area >= first_area * 4 else "low",
        "signals": ["bbox_area_increase"],
        "evidence": [positions[0].get("frame_id"), positions[-1].get("frame_id")],
        "area_ratio": round(last_area / first_area, 3),
    })


def _detect_rapid_area_changes(track: dict) -> list[dict]:
    positions = track.get("positions", [])
    events: list[dict] = []
    for previous, current in zip(positions, positions[1:]):
        previous_area = _bbox_area(previous.get("bbox", []))
        current_area = _bbox_area(current.get("bbox", []))
        if previous_area <= 0 or current_area < previous_area * 1.8:
            continue
        events.append(
            _with_event_score(
                {
                    "time": current.get("time"),
                    "event_type": "급접근후보",
                    "actors": [track.get("track_id")],
                    "confidence": "medium" if current_area >= previous_area * 2.8 else "low",
                    "signals": ["bbox_area_frame_to_frame_increase"],
                    "evidence": [previous.get("frame_id"), current.get("frame_id")],
                    "area_ratio": round(current_area / previous_area, 3),
                }
            )
        )
    return events


def _detect_sudden_stop(speed_and_distance: dict) -> dict | None:
    estimates = [
        estimate
        for estimate in speed_and_distance.get("speed_estimates", [])
        if isinstance(estimate, dict)
        and estimate.get("actor_id") == "ego_vehicle"
        and estimate.get("numeric_kmh") is not None
    ]
    if len(estimates) < 2:
        return None
    first = estimates[0]
    last = estimates[-1]
    delta = float(first["numeric_kmh"]) - float(last["numeric_kmh"])
    if delta < 20:
        return None
    return _with_event_score({
        "time": last.get("time", "확인불가"),
        "event_type": "급감속",
        "actors": ["ego_vehicle"],
        "confidence": "medium" if delta >= 30 else "low",
        "signals": ["speed_drop"],
        "evidence": [*first.get("evidence", []), *last.get("evidence", [])],
        "speed_delta_kmh": round(delta, 3),
    })


def _with_event_score(event: dict) -> dict:
    confidence_scores = {"high": 30, "medium": 20, "low": 10, "unknown": 0}
    signal_scores = {
        "bbox_overlap": 35,
        "bbox_area_increase": 28,
        "bbox_area_frame_to_frame_increase": 24,
        "speed_drop": 30,
        "track_lateral_motion": 18,
        "relative_motion": 14,
        "camera_shake": 12,
    }
    score = confidence_scores.get(str(event.get("confidence", "unknown")), 0)
    score += sum(signal_scores.get(str(signal), 0) for signal in event.get("signals", []))
    score += min(10, len([item for item in event.get("evidence", []) if item]) * 3)
    if event.get("event_type") in {"접촉", "충격후보"}:
        score += 12
    event["event_score"] = max(0, min(100, int(score)))
    return event
