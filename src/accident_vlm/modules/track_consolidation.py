from __future__ import annotations

from copy import deepcopy


def consolidate_tracks(
    tracks: list[dict],
    max_frame_gap: int = 45,
    max_center_distance_px: float = 240.0,
) -> list[dict]:
    consolidated: list[dict] = []
    for track in sorted(
        [deepcopy(track) for track in tracks],
        key=lambda item: _first_frame_index(item),
    ):
        target = _find_merge_target(track, consolidated, max_frame_gap, max_center_distance_px)
        if target is None:
            track["consolidated_from"] = [track.get("track_id")]
            consolidated.append(track)
            continue
        target["positions"] = sorted(
            [*target.get("positions", []), *track.get("positions", [])],
            key=lambda position: _frame_index_from_id(position.get("frame_id", "")),
        )
        target.setdefault("consolidated_from", [target.get("track_id")])
        if track.get("track_id") not in target["consolidated_from"]:
            target["consolidated_from"].append(track.get("track_id"))
        target["confidence"] = _merged_confidence(target)
        target["movement_candidate"] = _movement_from_positions(target.get("positions", []))
        target["tracking_method"] = "consolidated_track"
    return consolidated


def _find_merge_target(
    track: dict,
    candidates: list[dict],
    max_frame_gap: int,
    max_center_distance_px: float,
) -> dict | None:
    if not track.get("positions"):
        return None
    first_position = track["positions"][0]
    first_frame = _frame_index_from_id(first_position.get("frame_id", ""))
    first_center = _center(first_position.get("bbox", []))
    for candidate in candidates:
        if candidate.get("type") != track.get("type") or not candidate.get("positions"):
            continue
        last_position = candidate["positions"][-1]
        last_frame = _frame_index_from_id(last_position.get("frame_id", ""))
        if first_frame < last_frame or first_frame - last_frame > max_frame_gap:
            continue
        if _distance(first_center, _center(last_position.get("bbox", []))) <= max_center_distance_px:
            return candidate
    return None


def _first_frame_index(track: dict) -> int:
    positions = track.get("positions", [])
    if not positions:
        return 10**9
    return _frame_index_from_id(positions[0].get("frame_id", ""))


def _frame_index_from_id(frame_id: str) -> int:
    digits = "".join(character for character in str(frame_id) if character.isdigit())
    return int(digits) if digits else 0


def _center(bbox: list[int]) -> tuple[float, float]:
    if len(bbox) != 4:
        return (0.0, 0.0)
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def _distance(first: tuple[float, float], second: tuple[float, float]) -> float:
    return ((first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2) ** 0.5


def _merged_confidence(track: dict) -> str:
    positions = track.get("positions", [])
    if len(positions) >= 5:
        return "high"
    if len(positions) >= 2:
        return "medium"
    return track.get("confidence", "low")


def _movement_from_positions(positions: list[dict]) -> str:
    if len(positions) < 2:
        return "확인불가"
    first = _center(positions[0].get("bbox", []))
    last = _center(positions[-1].get("bbox", []))
    dx = last[0] - first[0]
    dy = last[1] - first[1]
    if abs(dx) > abs(dy) and abs(dx) > 20:
        return "차로변경_우" if dx > 0 else "차로변경_좌"
    if dy > 20:
        return "접근"
    if dy < -20:
        return "이탈"
    return "직진"
