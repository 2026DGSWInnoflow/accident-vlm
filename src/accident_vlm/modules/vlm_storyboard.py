from __future__ import annotations

from copy import deepcopy
from typing import Any

from accident_vlm.utils.timecode import parse_timecode


PHASE_TARGETS = {
    "insurance_context": 3,
    "scene_context": 2,
    "pre_event": 3,
    "approach": 3,
    "impact_candidate": 4,
    "post_event": 3,
    "detail": 2,
}

PHASE_HARD_LIMITS = {
    "insurance_context": 3,
    "detail": 4,
}

PHASE_ORDER = {
    "insurance_context": 0,
    "scene_context": 1,
    "pre_event": 2,
    "approach": 3,
    "impact_candidate": 4,
    "post_event": 5,
    "detail": 6,
}

PURPOSE_PHASES = {
    "regular_context": "scene_context",
    "event_window_context": "scene_context",
    "event_scan_pre_context": "scene_context",
    "pre_context": "pre_event",
    "event_scan_pre_impact": "approach",
    "motion_keyframe": "approach",
    "pre_impact": "approach",
    "event_scan_impact_candidate": "impact_candidate",
    "impact_candidate": "impact_candidate",
    "event_segment": "impact_candidate",
    "event_scan_post_impact": "post_event",
    "post_impact": "post_event",
    "actor_crop": "detail",
    "tracking_overlay": "detail",
    "track_overlay": "detail",
    "traffic_light_crop": "insurance_context",
    "sign_crop": "insurance_context",
    "bev_overlay": "insurance_context",
    "lane_segmentation_overlay": "insurance_context",
    "frame_selection_contact_sheet": "detail",
}

PHASE_ROLES = {
    "insurance_context": "보험/RAG 핵심 항목인 신호, 표지, 차로 수, 도로 형태, 진행방향 확인",
    "scene_context": "도로 구조, 차로, 시야, 날씨/조명 등 사고 전 배경 확인",
    "pre_event": "사고 전 객체 배치와 새 객체 등장 여부 확인",
    "approach": "객체 간 거리 변화, 접근, 차로 변화, 위험 증가 확인",
    "impact_candidate": "접촉/충돌 후보 순간과 직전·직후 변화 확인",
    "post_event": "충돌 후 정지, 이탈, 낙상, 파편, 후속 상태 확인",
    "detail": "원본 프레임에서 작게 보이는 객체, 표지, 신호, tracking 근거 확대 확인",
}


def build_vlm_storyboard(evidence_package: dict[str, Any], max_items: int = 20) -> list[dict[str, Any]]:
    if max_items <= 0:
        return []

    facts = evidence_package.get("precomputed_facts", {})
    fact_index = _build_fact_index(facts)
    records = _collect_candidate_records(evidence_package)
    enriched = [_enrich_record(record, fact_index) for record in records if record.get("path")]
    selected = _select_with_phase_targets(enriched, max_items)
    ordered = sorted(selected, key=_storyboard_order_key)
    return [
        {
            "slot": index,
            **item,
        }
        for index, item in enumerate(ordered, start=1)
    ]


def _collect_candidate_records(evidence_package: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for section in ("evidence_images", "frames", "overlays", "crops"):
        for item in evidence_package.get(section, []):
            if not isinstance(item, dict) or not item.get("path"):
                continue
            path = str(item["path"])
            if path in seen_paths:
                continue
            seen_paths.add(path)
            record = deepcopy(item)
            record.setdefault("source_section", section)
            records.append(record)
    return records


def _enrich_record(record: dict[str, Any], fact_index: dict[str, Any]) -> dict[str, Any]:
    item = {
        key: deepcopy(value)
        for key, value in record.items()
        if key
        in {
            "id",
            "path",
            "frame_id",
            "time",
            "purpose",
            "source",
            "source_section",
            "track_id",
            "tracks",
            "type",
            "bbox",
            "importance_score",
            "rank_reason",
        }
    }
    purpose = str(item.get("purpose") or "")
    phase = PURPOSE_PHASES.get(purpose, "scene_context")
    linked_actor_ids = _linked_actor_ids(item, fact_index)
    hints = _precomputed_hints(item, fact_index, linked_actor_ids)
    item.update(
        {
            "phase": phase,
            "role": PHASE_ROLES[phase],
            "why_selected": _why_selected(item, phase),
            "linked_actor_ids": linked_actor_ids,
            "precomputed_hints": hints,
        }
    )
    return item


def _build_fact_index(facts: dict[str, Any]) -> dict[str, Any]:
    frame_track_ids: dict[str, list[str]] = {}
    frame_actor_types: dict[str, list[str]] = {}
    for track in facts.get("tracks", []):
        if not isinstance(track, dict) or not track.get("track_id"):
            continue
        track_id = str(track["track_id"])
        actor_type = str(track.get("type") or "")
        for position in track.get("positions", []):
            if not isinstance(position, dict) or not position.get("frame_id"):
                continue
            frame_id = str(position["frame_id"])
            frame_track_ids.setdefault(frame_id, []).append(track_id)
            if actor_type:
                frame_actor_types.setdefault(frame_id, []).append(actor_type)

    event_candidates: list[dict[str, Any]] = []
    for event in facts.get("event_candidates", []):
        if not isinstance(event, dict):
            continue
        seconds = _safe_seconds(event.get("time"))
        if seconds is not None:
            event_candidates.append({"seconds": seconds, "event": event})

    return {
        "frame_track_ids": {
            frame_id: sorted(dict.fromkeys(track_ids))
            for frame_id, track_ids in frame_track_ids.items()
        },
        "frame_actor_types": {
            frame_id: list(dict.fromkeys(actor_types))
            for frame_id, actor_types in frame_actor_types.items()
        },
        "event_candidates": sorted(event_candidates, key=lambda item: item["seconds"]),
    }


def _select_with_phase_targets(records: list[dict[str, Any]], max_items: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_paths: set[str] = set()
    phase_counts: dict[str, int] = {}
    by_phase = {
        phase: sorted(
            [record for record in records if record.get("phase") == phase],
            key=_rank_key,
        )
        for phase in PHASE_TARGETS
    }
    for phase, target in PHASE_TARGETS.items():
        phase_limit = min(target, max_items - len(selected))
        for record in by_phase.get(phase, [])[:phase_limit]:
            _append_unique(selected, selected_paths, record, phase_counts)
            if len(selected) >= max_items:
                return selected

    for record in sorted(records, key=_rank_key):
        _append_unique(selected, selected_paths, record, phase_counts)
        if len(selected) >= max_items:
            break
    return selected


def _append_unique(
    selected: list[dict[str, Any]],
    selected_paths: set[str],
    record: dict[str, Any],
    phase_counts: dict[str, int],
) -> None:
    path = str(record.get("path") or "")
    if not path or path in selected_paths:
        return
    phase = str(record.get("phase") or "")
    phase_limit = PHASE_HARD_LIMITS.get(phase)
    phase_count = phase_counts.get(phase, 0)
    if phase_limit is not None and phase_count >= phase_limit:
        return
    selected_paths.add(path)
    selected.append(record)
    phase_counts[phase] = phase_count + 1


def _linked_actor_ids(item: dict[str, Any], fact_index: dict[str, Any]) -> list[str]:
    linked: list[str] = []
    for key in ("track_id",):
        if item.get(key):
            linked.append(str(item[key]))
    tracks = item.get("tracks")
    if isinstance(tracks, list):
        linked.extend(str(track) for track in tracks if track)

    frame_id = item.get("frame_id")
    if frame_id and not linked:
        linked.extend(fact_index.get("frame_track_ids", {}).get(str(frame_id), []))
    return sorted(dict.fromkeys(linked))


def _precomputed_hints(
    item: dict[str, Any],
    fact_index: dict[str, Any],
    linked_actor_ids: list[str],
) -> dict[str, Any]:
    frame_id = item.get("frame_id")
    visible_actor_candidates = (
        fact_index.get("frame_actor_types", {}).get(str(frame_id), [])
        if frame_id
        else []
    )

    hints: dict[str, Any] = {
        "visible_actor_candidates": list(dict.fromkeys(visible_actor_candidates)),
        "linked_actor_ids": linked_actor_ids,
        "nearby_event_candidates": _nearby_events(item, fact_index),
    }
    if item.get("type"):
        hints["actor_type_hint"] = item["type"]
    if item.get("bbox"):
        hints["localized_bbox"] = item["bbox"]
    return {key: value for key, value in hints.items() if value not in (None, [], {})}


def _nearby_events(item: dict[str, Any], fact_index: dict[str, Any]) -> list[dict[str, Any]]:
    item_seconds = _safe_seconds(item.get("time"))
    if item_seconds is None:
        return []
    nearby: list[dict[str, Any]] = []
    for indexed_event in fact_index.get("event_candidates", []):
        event_seconds = indexed_event["seconds"]
        if abs(event_seconds - item_seconds) > 1.5:
            continue
        event = indexed_event["event"]
        nearby.append(
            {
                key: event.get(key)
                for key in ("time", "event_type", "event_score", "confidence", "reason")
                if event.get(key) is not None
            }
        )
    return nearby[:3]


def _why_selected(item: dict[str, Any], phase: str) -> str:
    reasons = [f"phase:{phase}", f"purpose:{item.get('purpose', 'unknown')}"]
    if item.get("importance_score") is not None:
        reasons.append(f"importance:{item['importance_score']}")
    if item.get("frame_id"):
        reasons.append(f"frame:{item['frame_id']}")
    if item.get("track_id") or item.get("tracks"):
        reasons.append("actor-linked")
    return ", ".join(reasons)


def _rank_key(item: dict[str, Any]) -> tuple[int, float, str]:
    return (
        -int(item.get("importance_score", 0) or 0),
        _safe_seconds(item.get("time")) or 0.0,
        str(item.get("id") or ""),
    )


def _storyboard_order_key(item: dict[str, Any]) -> tuple[float, int, int, str]:
    seconds = _safe_seconds(item.get("time"))
    return (
        seconds if seconds is not None else float("inf"),
        PHASE_ORDER.get(str(item.get("phase")), 99),
        1 if item.get("phase") == "detail" else 0,
        str(item.get("id") or ""),
    )


def _safe_seconds(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return parse_timecode(str(value))
    except ValueError:
        return None
