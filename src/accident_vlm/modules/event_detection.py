from __future__ import annotations

from itertools import combinations
import math

from accident_vlm.utils.timecode import parse_timecode


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
    event_scan_candidates: list[dict] | None = None,
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
                    "supporting_signals": {"movement_candidate": movement},
                    "contradicting_signals": [],
                    "evidence": [first_position.get("frame_id")],
                }
            )
        approach_event = _detect_distance_drop(track)
        if approach_event:
            events.append(approach_event)
        for area_event in _detect_rapid_area_changes(track):
            events.append(area_event)
        for motion_event in _detect_track_motion_events(track):
            events.append(motion_event)

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
                shake_signal = _camera_shake_signal(input_quality, time)
                signals = ["bbox_overlap"]
                supporting_signals = {"bbox_iou": round(iou, 4)}
                previous_iou = _previous_pair_iou(tracks, first[0], second[0], time)
                if previous_iou is not None:
                    supporting_signals["bbox_iou_change_rate"] = round(max(0.0, iou - previous_iou), 4)
                evidence = [first[2], second[2]]
                if shake_signal:
                    signals.append("camera_shake")
                    supporting_signals.update(shake_signal["supporting_signals"])
                    evidence.extend(shake_signal.get("evidence", []))
                events.append(
                    _with_event_score({
                        "time": time,
                        "event_type": "접촉",
                        "candidate_class": "direct_contact_candidate",
                        "actors": [first[0], second[0]],
                        "confidence": "medium" if iou >= 0.12 else "low",
                        "signals": signals,
                        "supporting_signals": supporting_signals,
                        "contradicting_signals": [],
                        "iou": round(iou, 4),
                        "evidence": _dedupe(evidence),
                    })
                )

    if speed_and_distance:
        sudden_stop = _detect_sudden_stop(speed_and_distance)
        if sudden_stop:
            events.append(sudden_stop)
        for item in speed_and_distance.get("relative_motion", []):
            if item.get("relative_speed_trend") == "접근":
                time = item.get("time", "확인불가")
                actor_id = item.get("actor_id")
                events.append(
                    _with_event_score({
                        "time": time,
                        "event_type": "접근",
                        "actors": [actor_id],
                        "confidence": item.get("confidence", "low"),
                        "signals": ["relative_motion"],
                        "supporting_signals": {"relative_speed_trend": "접근"},
                        "contradicting_signals": [],
                    })
                )
                events.append(
                    _with_event_score({
                        "time": time,
                        "event_type": "비접촉후보",
                        "candidate_class": "non_contact_candidate",
                        "actors": [actor_id],
                        "confidence": item.get("confidence", "low"),
                        "signals": ["relative_motion"],
                        "supporting_signals": {"relative_speed_trend": "접근"},
                        "contradicting_signals": ["bbox_overlap_not_confirmed"],
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
                    "supporting_signals": _camera_shake_supporting_signals(shake_score),
                    "contradicting_signals": [],
                    "evidence": shake_score.get("evidence", []),
                })
            )
    for event_scan in event_scan_candidates or []:
        optical_flow_event = _event_scan_optical_flow_candidate(event_scan)
        if optical_flow_event:
            events.append(optical_flow_event)
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
        "supporting_signals": {
            "first_bbox_area": round(first_area, 3),
            "last_bbox_area": round(last_area, 3),
            "area_ratio": round(last_area / first_area, 3),
        },
        "contradicting_signals": [],
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
                    "supporting_signals": {
                        "previous_bbox_area": round(previous_area, 3),
                        "current_bbox_area": round(current_area, 3),
                        "area_ratio": round(current_area / previous_area, 3),
                    },
                    "contradicting_signals": [],
                    "evidence": [previous.get("frame_id"), current.get("frame_id")],
                    "area_ratio": round(current_area / previous_area, 3),
                }
            )
        )
    return events


def _detect_track_motion_events(track: dict) -> list[dict]:
    positions = track.get("positions", [])
    if len(positions) < 3:
        return []
    motion = _motion_samples(positions)
    events: list[dict] = []
    for previous, current in zip(motion, motion[1:]):
        speed_delta = current["speed_px_per_sec"] - previous["speed_px_per_sec"]
        if previous["speed_px_per_sec"] >= 20 and current["speed_px_per_sec"] <= previous["speed_px_per_sec"] * 0.35:
            events.append(
                _with_event_score(
                    {
                        "time": current["time"],
                        "event_type": "사고후정지후보",
                        "candidate_class": "post_event_state_candidate",
                        "actors": [track.get("track_id")],
                        "confidence": "medium" if previous["speed_px_per_sec"] >= 35 else "low",
                        "signals": ["object_speed_change"],
                        "supporting_signals": {
                            "previous_speed_px_per_sec": round(previous["speed_px_per_sec"], 3),
                            "current_speed_px_per_sec": round(current["speed_px_per_sec"], 3),
                            "object_speed_change_rate": round(speed_delta, 3),
                        },
                        "contradicting_signals": [],
                        "evidence": [previous["frame_id"], current["frame_id"]],
                    }
                )
            )
        angle_delta = _angle_delta(previous["angle"], current["angle"])
        if angle_delta >= 60 and previous["speed_px_per_sec"] >= 10 and current["speed_px_per_sec"] >= 10:
            events.append(
                _with_event_score(
                    {
                        "time": current["time"],
                        "event_type": "방향변화후보",
                        "candidate_class": "post_event_state_candidate",
                        "actors": [track.get("track_id")],
                        "confidence": "medium" if angle_delta >= 90 else "low",
                        "signals": ["object_direction_change"],
                        "supporting_signals": {
                            "angle_delta_deg": round(angle_delta, 3),
                            "previous_speed_px_per_sec": round(previous["speed_px_per_sec"], 3),
                            "current_speed_px_per_sec": round(current["speed_px_per_sec"], 3),
                        },
                        "contradicting_signals": [],
                        "evidence": [previous["frame_id"], current["frame_id"]],
                    }
                )
            )
    fall = _detect_fall_candidate(track)
    if fall:
        events.append(fall)
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
        "supporting_signals": {
            "start_speed_kmh": round(float(first["numeric_kmh"]), 3),
            "end_speed_kmh": round(float(last["numeric_kmh"]), 3),
            "speed_delta_kmh": round(delta, 3),
        },
        "contradicting_signals": [],
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
        "event_scan_optical_flow": 22,
        "object_speed_change": 18,
        "object_direction_change": 16,
        "bbox_aspect_ratio_change": 18,
    }
    score = confidence_scores.get(str(event.get("confidence", "unknown")), 0)
    score += sum(signal_scores.get(str(signal), 0) for signal in event.get("signals", []))
    score += min(10, len([item for item in event.get("evidence", []) if item]) * 3)
    if event.get("event_type") in {"접촉", "충격후보"}:
        score += 12
    if event.get("candidate_class") == "direct_contact_candidate":
        score += 8
    if isinstance(event.get("supporting_signals"), dict):
        score += min(10, len(event["supporting_signals"]) * 2)
    event["event_score"] = max(0, min(100, int(score)))
    return event


def _event_scan_optical_flow_candidate(event_scan: dict) -> dict | None:
    signals = event_scan.get("supporting_signals", {})
    if not isinstance(signals, dict):
        return None
    optical_flow_peak = float(signals.get("optical_flow_peak") or 0.0)
    if optical_flow_peak < 1.2:
        return None
    return _with_event_score(
        {
            "time": event_scan.get("time", "확인불가"),
            "event_type": "광류급변후보",
            "candidate_class": "motion_peak_candidate",
            "actors": [],
            "confidence": event_scan.get("confidence", "low"),
            "signals": ["event_scan_optical_flow"],
            "supporting_signals": {
                "optical_flow_peak": round(optical_flow_peak, 3),
                "optical_flow_mean": round(float(signals.get("optical_flow_mean") or 0.0), 3),
                "histogram_change": round(float(signals.get("histogram_change") or 0.0), 3),
            },
            "contradicting_signals": [],
            "evidence": event_scan.get("evidence", []),
            "event_score": event_scan.get("event_score", 0),
            "source": "high_fps_event_scan",
        }
    )


def _previous_pair_iou(tracks: list[dict], first_id: str, second_id: str, time: str) -> float | None:
    first_positions = _positions_by_time(_find_track(tracks, first_id))
    second_positions = _positions_by_time(_find_track(tracks, second_id))
    try:
        target_seconds = parse_timecode(time)
    except ValueError:
        return None
    previous_times = sorted(
        [
            item_time
            for item_time in set(first_positions) & set(second_positions)
            if _parse_time_or_none(item_time) is not None and _parse_time_or_none(item_time) < target_seconds
        ],
        key=lambda item_time: _parse_time_or_none(item_time) or 0.0,
    )
    if not previous_times:
        return None
    previous_time = previous_times[-1]
    return _bbox_iou(
        first_positions[previous_time].get("bbox", []),
        second_positions[previous_time].get("bbox", []),
    )


def _find_track(tracks: list[dict], track_id: str) -> dict:
    return next((track for track in tracks if track.get("track_id") == track_id), {})


def _positions_by_time(track: dict) -> dict[str, dict]:
    return {str(position.get("time", "")): position for position in track.get("positions", [])}


def _parse_time_or_none(value: str) -> float | None:
    try:
        return parse_timecode(value)
    except ValueError:
        return None


def _motion_samples(positions: list[dict]) -> list[dict]:
    samples: list[dict] = []
    sorted_positions = sorted(positions, key=lambda position: _parse_time_or_none(str(position.get("time", ""))) or 0.0)
    for previous, current in zip(sorted_positions, sorted_positions[1:]):
        previous_time = _parse_time_or_none(str(previous.get("time", "")))
        current_time = _parse_time_or_none(str(current.get("time", "")))
        if previous_time is None or current_time is None or current_time <= previous_time:
            continue
        previous_center = _center(previous.get("bbox", []))
        current_center = _center(current.get("bbox", []))
        dx = current_center[0] - previous_center[0]
        dy = current_center[1] - previous_center[1]
        dt = current_time - previous_time
        samples.append(
            {
                "time": current.get("time", "확인불가"),
                "frame_id": current.get("frame_id"),
                "speed_px_per_sec": ((dx**2 + dy**2) ** 0.5) / dt,
                "angle": _angle(dx, dy),
            }
        )
    return samples


def _center(bbox: list[int]) -> tuple[float, float]:
    if len(bbox) != 4:
        return (0.0, 0.0)
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def _angle(dx: float, dy: float) -> float:
    return math.degrees(math.atan2(dy, dx))


def _angle_delta(first: float, second: float) -> float:
    delta = abs(first - second) % 360
    return min(delta, 360 - delta)


def _detect_fall_candidate(track: dict) -> dict | None:
    if str(track.get("type", "")).lower() not in {"person", "보행자", "bicycle", "자전거", "motorcycle", "오토바이", "킥보드"}:
        return None
    positions = track.get("positions", [])
    if len(positions) < 2:
        return None
    first_ratio = _bbox_aspect_ratio(positions[0].get("bbox", []))
    last_ratio = _bbox_aspect_ratio(positions[-1].get("bbox", []))
    if first_ratio <= 0 or last_ratio >= first_ratio * 0.65:
        return None
    return _with_event_score(
        {
            "time": positions[-1].get("time"),
            "event_type": "낙상후보",
            "candidate_class": "post_event_state_candidate",
            "actors": [track.get("track_id")],
            "confidence": "medium" if last_ratio <= first_ratio * 0.45 else "low",
            "signals": ["bbox_aspect_ratio_change"],
            "supporting_signals": {
                "first_bbox_height_width_ratio": round(first_ratio, 3),
                "last_bbox_height_width_ratio": round(last_ratio, 3),
                "aspect_ratio_change": round(first_ratio - last_ratio, 3),
            },
            "contradicting_signals": [],
            "evidence": [positions[0].get("frame_id"), positions[-1].get("frame_id")],
        }
    )


def _bbox_aspect_ratio(bbox: list[int]) -> float:
    if len(bbox) != 4:
        return 0.0
    width = max(1, bbox[2] - bbox[0])
    height = max(0, bbox[3] - bbox[1])
    return height / width


def _camera_shake_signal(input_quality: dict | None, event_time: str) -> dict | None:
    if not input_quality:
        return None
    shake_score = input_quality.get("camera_shake_score", {})
    if not isinstance(shake_score, dict) or float(shake_score.get("value") or 0.0) < 20:
        return None
    if not _is_near_time(event_time, str(shake_score.get("time", "")), tolerance_seconds=0.75):
        return None
    return {
        "supporting_signals": _camera_shake_supporting_signals(shake_score),
        "evidence": shake_score.get("evidence", []),
    }


def _camera_shake_supporting_signals(shake_score: dict) -> dict:
    supporting = {"camera_shake_score": round(float(shake_score.get("value") or 0.0), 3)}
    if shake_score.get("ego_motion_compensated_value") is not None:
        supporting["ego_motion_compensated_shake"] = round(
            float(shake_score.get("ego_motion_compensated_value") or 0.0),
            3,
        )
    return supporting


def _is_near_time(first: str, second: str, tolerance_seconds: float) -> bool:
    try:
        return abs(parse_timecode(first) - parse_timecode(second)) <= tolerance_seconds
    except ValueError:
        return first == second or not first or not second


def _dedupe(items: list) -> list:
    seen = set()
    deduped = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped
