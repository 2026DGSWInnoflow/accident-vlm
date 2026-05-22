import re

from accident_vlm.utils.timecode import parse_timecode

SPEED_METHOD_PRIORITY = {
    "metadata": 0,
    "gps": 0,
    "obd": 0,
    "ocr_overlay": 1,
    "bev_tracking_estimate": 2,
    "relative_motion_only": 3,
}


METHOD_TIE_BREAK = {
    "metadata": 0,
    "gps": 1,
    "obd": 2,
    "ocr_overlay": 3,
    "bev_tracking_estimate": 4,
    "relative_motion_only": 5,
}


def _unknown_speed_estimate(actor_id: str) -> dict:
    return {
        "actor_id": actor_id,
        "value": "모름",
        "numeric_kmh": None,
        "range_kmh": None,
        "method": "not_available",
        "confidence": "unknown",
    }


def choose_speed_estimate(estimates: list[dict], actor_id: str) -> dict:
    actor_estimates = [
        estimate
        for estimate in estimates
        if isinstance(estimate, dict)
        and estimate.get("actor_id") == actor_id
        and estimate.get("method") in SPEED_METHOD_PRIORITY
    ]
    if not actor_estimates:
        return _unknown_speed_estimate(actor_id)

    return sorted(
        actor_estimates,
        key=lambda estimate: (
            SPEED_METHOD_PRIORITY[estimate["method"]],
            METHOD_TIE_BREAK[estimate["method"]],
        ),
    )[0]


def _track_center(position: dict) -> tuple[float, float]:
    bbox = position.get("bbox", [0, 0, 0, 0])
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def _track_bottom_center(position: dict) -> tuple[float, float]:
    bbox = position.get("bbox", [0, 0, 0, 0])
    return ((bbox[0] + bbox[2]) / 2, bbox[3])


def _relative_motion_from_track(track: dict) -> dict:
    positions = track.get("positions", [])
    if len(positions) < 2:
        return {
            "actor_id": track.get("track_id", "unknown"),
            "relative_speed_trend": "확인불가",
            "lateral_movement": "확인불가",
            "confidence": "unknown",
        }
    first = _track_center(positions[0])
    last = _track_center(positions[-1])
    dx = last[0] - first[0]
    dy = last[1] - first[1]
    lateral = "좌측" if dx < -20 else "우측" if dx > 20 else "없음"
    trend = "접근" if dy > 20 else "이탈" if dy < -20 else "유지"
    return {
        "actor_id": track.get("track_id", "unknown"),
        "time": positions[-1].get("time"),
        "relative_speed_trend": trend,
        "lateral_movement": lateral,
        "confidence": track.get("confidence", "medium"),
        "method": "relative_motion_only",
        "evidence": [positions[0].get("frame_id"), positions[-1].get("frame_id")],
    }


def estimate_speed_and_distance(
    ocr_observations: list[dict],
    tracks: list[dict],
    road_geometry: dict,
    ocr_summary: dict | None = None,
) -> dict:
    raw_estimates: list[dict] = []
    ego_speed_kmh: float | None = None
    if ocr_summary:
        summary_speed = ocr_summary.get("speed", {})
        if summary_speed.get("numeric_kmh") is not None:
            ego_speed_kmh = float(summary_speed.get("numeric_kmh"))
            raw_estimates.append(
                {
                    "actor_id": "ego_vehicle",
                    "value": summary_speed.get("value"),
                    "numeric_kmh": summary_speed.get("numeric_kmh"),
                    "range_kmh": summary_speed.get("range_kmh"),
                    "method": "ocr_overlay",
                    "confidence": summary_speed.get("confidence", "medium"),
                    "evidence": summary_speed.get("evidence", []),
                    "source": "ocr_summary",
                }
            )

    distance_estimates: list[dict] = []
    for track in tracks:
        bev_estimate = _bev_tracking_estimate(track, road_geometry, ego_speed_kmh=ego_speed_kmh)
        if bev_estimate:
            raw_estimates.append(bev_estimate["speed_estimate"])
            distance_estimates.append(bev_estimate["distance_estimate"])

    for observation in ocr_observations:
        parsed = observation.get("parsed", {})
        speed_kmh = parsed.get("speed_kmh")
        if speed_kmh is None:
            continue
        raw_estimates.append(
            {
                "actor_id": "ego_vehicle",
                "value": f"{speed_kmh:g}km/h",
                "numeric_kmh": speed_kmh,
                "range_kmh": [max(speed_kmh - 2, 0), speed_kmh + 2],
                "method": "ocr_overlay",
                "confidence": "medium" if observation.get("confidence", 0) < 0.9 else "high",
                "evidence": [observation.get("frame_id")],
            }
        )

    relative_motion = [_relative_motion_from_track(track) for track in tracks]
    speed_estimates = [choose_speed_estimate(raw_estimates, "ego_vehicle")]
    for track in tracks:
        speed_estimates.append(choose_speed_estimate(raw_estimates, track.get("track_id", "unknown")))

    return {
        "speed_estimates": speed_estimates,
        "distance_estimates": distance_estimates,
        "relative_motion": relative_motion,
        "geometry_dependency": {
            "homography_available": road_geometry.get("homography", {}).get("available", False),
            "bev_confidence": road_geometry.get("bev", {}).get("confidence", "unknown"),
            "failure_reason": _geometry_failure_reason(road_geometry),
            "note": "BEV homography와 pixels_per_meter가 충분할 때만 bbox bottom-center 기반 절대 속도/거리 후보를 산출",
        },
    }


def _bev_tracking_estimate(track: dict, road_geometry: dict, ego_speed_kmh: float | None = None) -> dict | None:
    homography = road_geometry.get("homography", {})
    matrix = homography.get("matrix")
    pixels_per_meter = homography.get("pixels_per_meter")
    if not homography.get("available") or not matrix or not pixels_per_meter:
        return None
    positions = track.get("positions", [])
    if len(positions) < 2:
        return None
    first = positions[0]
    last = positions[-1]
    first_time = _parse_time_or_none(str(first.get("time", "")))
    last_time = _parse_time_or_none(str(last.get("time", "")))
    if first_time is None or last_time is None or last_time <= first_time:
        return None
    first_point = _apply_homography(_track_bottom_center(first), matrix)
    last_point = _apply_homography(_track_bottom_center(last), matrix)
    delta_px = ((last_point[0] - first_point[0]) ** 2 + (last_point[1] - first_point[1]) ** 2) ** 0.5
    distance_m = delta_px / float(pixels_per_meter)
    delta_time = last_time - first_time
    delta_frames = _frame_delta(first.get("frame_id"), last.get("frame_id"))
    speed_kmh = distance_m / delta_time * 3.6
    confidence = _bev_estimate_confidence(road_geometry)
    actor_id = track.get("track_id", "unknown")
    formula = {
        "point": "bbox_bottom_center",
        "delta_px_bev": round(delta_px, 3),
        "pixels_per_meter": float(pixels_per_meter),
        "distance_m": round(distance_m, 3),
        "delta_time_sec": round(delta_time, 3),
        "delta_frames": delta_frames,
        "implied_fps": round(delta_frames / delta_time, 3) if delta_frames is not None else None,
        "speed_kmh": round(speed_kmh, 3),
        "ego_motion_compensation": _ego_motion_compensation_note(ego_speed_kmh),
    }
    evidence = [first.get("frame_id"), last.get("frame_id")]
    absolute_range = _combine_with_ego_speed(speed_kmh, ego_speed_kmh)
    return {
        "speed_estimate": {
            "actor_id": actor_id,
            "value": f"{speed_kmh:.1f}km/h",
            "numeric_kmh": round(speed_kmh, 3),
            "range_kmh": [round(speed_kmh * 0.8, 3), round(speed_kmh * 1.2, 3)],
            "method": "bev_tracking_estimate",
            "confidence": confidence,
            "source": "bev_homography_bbox_bottom_center",
            "formula": formula,
            "relative_speed_kmh": round(speed_kmh, 3),
            "absolute_speed_range_kmh": absolute_range,
            "evidence": evidence,
        },
        "distance_estimate": {
            "actor_id": actor_id,
            "value_m": round(distance_m, 3),
            "range_m": [round(distance_m * 0.8, 3), round(distance_m * 1.2, 3)],
            "method": "bev_homography_bbox_bottom_center",
            "confidence": confidence,
            "formula": formula,
            "evidence": evidence,
        },
    }


def _apply_homography(point: tuple[float, float], matrix: list[list[float]]) -> tuple[float, float]:
    x, y = point
    denominator = matrix[2][0] * x + matrix[2][1] * y + matrix[2][2]
    if abs(denominator) < 1e-9:
        return (x, y)
    next_x = (matrix[0][0] * x + matrix[0][1] * y + matrix[0][2]) / denominator
    next_y = (matrix[1][0] * x + matrix[1][1] * y + matrix[1][2]) / denominator
    return (float(next_x), float(next_y))


def _parse_time_or_none(value: str) -> float | None:
    try:
        return parse_timecode(value)
    except ValueError:
        return None


def _frame_delta(first_frame_id: object, last_frame_id: object) -> int | None:
    first_groups = re.findall(r"\d+", str(first_frame_id))
    last_groups = re.findall(r"\d+", str(last_frame_id))
    if not first_groups or not last_groups:
        return None
    return int(last_groups[-1]) - int(first_groups[-1])


def _bev_estimate_confidence(road_geometry: dict) -> str:
    score = float(road_geometry.get("bev", {}).get("confidence_score") or 0.0)
    if score >= 0.8:
        return "high"
    if score >= 0.45:
        return "medium"
    return "low"


def _geometry_failure_reason(road_geometry: dict) -> str | None:
    homography = road_geometry.get("homography", {})
    if homography.get("available") and homography.get("matrix") and homography.get("pixels_per_meter"):
        return None
    reasons = homography.get("failure_reasons") or road_geometry.get("bev", {}).get("failure_reasons")
    if isinstance(reasons, list) and reasons:
        return "; ".join(str(reason) for reason in reasons)
    return "homography_or_pixels_per_meter_not_available"


def _combine_with_ego_speed(relative_speed_kmh: float, ego_speed_kmh: float | None) -> list[float] | None:
    if ego_speed_kmh is None:
        return None
    return [round(max(0.0, ego_speed_kmh - relative_speed_kmh), 3), round(ego_speed_kmh + relative_speed_kmh, 3)]


def _ego_motion_compensation_note(ego_speed_kmh: float | None) -> dict:
    if ego_speed_kmh is None:
        return {
            "applied": False,
            "reason": "ego OCR/metadata speed unavailable; absolute speed is treated as low-confidence BEV-relative estimate",
        }
    return {
        "applied": True,
        "ego_speed_reference_kmh": round(ego_speed_kmh, 3),
        "note": "ego speed is retained as a reference range because monocular BEV direction calibration is uncertain",
    }
