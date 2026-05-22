from __future__ import annotations

from copy import deepcopy
import re


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
    for track in consolidated:
        track["positions"] = sorted(
            track.get("positions", []),
            key=lambda position: _frame_index_from_id(position.get("frame_id", "")),
        )
        track["track_quality"] = _track_quality(track)
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
    groups = re.findall(r"\d+", str(frame_id))
    return int(groups[-1]) if groups else 0


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


def _track_quality(track: dict) -> dict:
    positions = track.get("positions", [])
    frame_indices = [_frame_index_from_id(position.get("frame_id", "")) for position in positions]
    gaps = [
        current - previous
        for previous, current in zip(frame_indices, frame_indices[1:])
        if current >= previous
    ]
    max_gap = max(gaps, default=0)
    missing_gap_count = len([gap for gap in gaps if gap > 6])
    merged_fragments = max(0, len(track.get("consolidated_from", [])) - 1)
    fragmentation_score = min(100, int(max_gap * 1.5 + missing_gap_count * 8 + merged_fragments * 12))
    bbox_area_timeline = [
        {
            "frame_id": position.get("frame_id"),
            "time": position.get("time"),
            "area": _bbox_area(position.get("bbox", [])),
        }
        for position in positions
    ]
    occlusion_candidates = _occlusion_candidates(bbox_area_timeline, gaps)
    return {
        "position_count": len(positions),
        "frame_span": [frame_indices[0], frame_indices[-1]] if frame_indices else [0, 0],
        "frame_indices": frame_indices,
        "max_frame_gap": max_gap,
        "gap_count_over_6_frames": missing_gap_count,
        "fragmentation_score": fragmentation_score,
        "visibility": "fragmented" if fragmentation_score >= 30 else "continuous",
        "bbox_area_timeline": bbox_area_timeline,
        "occlusion_candidates": occlusion_candidates,
        "confidence": _quality_confidence(len(positions), fragmentation_score),
    }


def _bbox_area(bbox: list[int]) -> float:
    if len(bbox) != 4:
        return 0.0
    return float(max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1]))


def _occlusion_candidates(bbox_area_timeline: list[dict], gaps: list[int]) -> list[dict]:
    candidates: list[dict] = []
    for index, (previous, current) in enumerate(zip(bbox_area_timeline, bbox_area_timeline[1:]), start=1):
        previous_area = float(previous.get("area") or 0.0)
        current_area = float(current.get("area") or 0.0)
        if previous_area > 0 and current_area < previous_area * 0.45:
            candidates.append(
                {
                    "time": current.get("time"),
                    "frame_id": current.get("frame_id"),
                    "reason": "bbox_area_drop",
                    "area_ratio": round(current_area / previous_area, 3),
                }
            )
        if index - 1 < len(gaps) and gaps[index - 1] > 15:
            candidates.append(
                {
                    "time": current.get("time"),
                    "frame_id": current.get("frame_id"),
                    "reason": "tracking_gap",
                    "frame_gap": gaps[index - 1],
                }
            )
    return candidates


def _quality_confidence(position_count: int, fragmentation_score: int) -> str:
    if position_count >= 6 and fragmentation_score < 25:
        return "high"
    if position_count >= 3 and fragmentation_score < 55:
        return "medium"
    return "low"
