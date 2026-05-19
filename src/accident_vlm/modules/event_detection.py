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


def detect_event_candidates(tracks: list[dict], speed_and_distance: dict | None = None) -> list[dict]:
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
                    {
                        "time": time,
                        "event_type": "접촉",
                        "actors": [first[0], second[0]],
                        "confidence": "medium" if iou >= 0.12 else "low",
                        "signals": ["bbox_overlap"],
                        "iou": round(iou, 4),
                        "evidence": [first[2], second[2]],
                    }
                )

    if speed_and_distance:
        for item in speed_and_distance.get("relative_motion", []):
            if item.get("relative_speed_trend") == "접근":
                events.append(
                    {
                        "time": item.get("time", "확인불가"),
                        "event_type": "접근",
                        "actors": [item.get("actor_id")],
                        "confidence": item.get("confidence", "low"),
                        "signals": ["relative_motion"],
                    }
                )
    return sorted(events, key=lambda event: event.get("time") or "")
