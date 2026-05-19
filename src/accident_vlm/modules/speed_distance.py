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
) -> dict:
    raw_estimates: list[dict] = []
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
        speed_estimates.append(_unknown_speed_estimate(track.get("track_id", "unknown")))

    return {
        "speed_estimates": speed_estimates,
        "distance_estimates": [],
        "relative_motion": relative_motion,
        "geometry_dependency": {
            "homography_available": road_geometry.get("homography", {}).get("available", False),
            "note": "절대 거리/속도 산출은 BEV homography와 카메라 캘리브레이션 고도화 후 활성화",
        },
    }
