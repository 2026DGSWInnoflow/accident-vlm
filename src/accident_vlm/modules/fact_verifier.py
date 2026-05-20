from __future__ import annotations

from copy import deepcopy
from typing import Any

from accident_vlm.schemas.preprocessing import PipelineContext


def verify_vlm_payload_against_context(payload: dict[str, Any], context: PipelineContext) -> dict[str, Any]:
    verified = deepcopy(payload)
    evidence_ids = _context_evidence_ids(context)
    _verify_timeline(verified, evidence_ids)
    _verify_actors(verified, evidence_ids)
    _verify_collision(verified, context, evidence_ids)
    _verify_traffic_control(verified, context)
    _verify_speed_and_distance(verified)
    _verify_rag_hints(verified)
    _add_precision_policy(verified)
    return verified


def _context_evidence_ids(context: PipelineContext) -> set[str]:
    ids = {frame.id for frame in context.selected_frames}
    for item in [*context.evidence_images, *context.overlays, *context.crops]:
        if isinstance(item, dict):
            for key in ("id", "frame_id"):
                if item.get(key):
                    ids.add(str(item[key]))
    return ids


def _verify_timeline(payload: dict[str, Any], evidence_ids: set[str]) -> None:
    for event in payload.get("timeline", []):
        if not isinstance(event, dict):
            continue
        event["evidence"] = _valid_evidence(event.get("evidence", []), evidence_ids)
        if not event["evidence"] or _confidence_rank(event.get("confidence")) < _confidence_rank("medium"):
            event["confidence"] = "unknown"
            event["event"] = "근거 부족으로 세부 사건 확인불가"
            _append_uncertainty(payload, "timeline 사건이 근거 부족으로 강등됨")


def _verify_actors(payload: dict[str, Any], evidence_ids: set[str]) -> None:
    for actor in payload.get("actors", []):
        if not isinstance(actor, dict):
            continue
        actor["evidence"] = _valid_evidence(actor.get("evidence", []), evidence_ids)
        if not actor["evidence"] or _confidence_rank(actor.get("confidence")) < _confidence_rank("medium"):
            actor["confidence"] = "unknown"
            for key in ("movement", "lane", "lane_or_position"):
                if key in actor:
                    actor[key] = "확인불가"
            _append_uncertainty(payload, "actor 세부 정보가 high-precision 정책으로 강등됨")


def _verify_collision(payload: dict[str, Any], context: PipelineContext, evidence_ids: set[str]) -> None:
    collision = payload.get("collision")
    if not isinstance(collision, dict):
        return
    collision["evidence"] = _valid_evidence(collision.get("evidence", []), evidence_ids)
    has_collision_event = any(
        event.get("event_type") in {"접촉", "충격후보"}
        and int(event.get("event_score", 0)) >= 35
        for event in context.event_candidates
        if isinstance(event, dict)
    )
    if (
        collision.get("detected") is True
        and (not has_collision_event or _confidence_rank(collision.get("confidence")) < _confidence_rank("medium"))
    ):
        collision["detected"] = False
        collision["confidence"] = "unknown"
        collision["note"] = "전처리 충돌 후보 근거가 부족하여 충돌 확인 불가"
        _append_uncertainty(payload, "collision detected 값이 전처리 근거 부족으로 강등됨")


def _verify_traffic_control(payload: dict[str, Any], context: PipelineContext) -> None:
    traffic = payload.get("traffic_control")
    if not isinstance(traffic, dict):
        return
    detected_signal = context.traffic_control.get("signal", {})
    for key in ("signal", "traffic_light"):
        signal = traffic.get(key)
        if not isinstance(signal, dict):
            continue
        if detected_signal.get("value") in {None, "확인불가"}:
            signal["value"] = "확인불가"
            signal["confidence"] = "unknown"
            _append_uncertainty(payload, f"traffic_control.{key} 값이 전처리 근거 부족으로 강등됨")
            continue
        if _confidence_rank(detected_signal.get("confidence")) < _confidence_rank("medium"):
            signal["value"] = "확인불가"
            signal["confidence"] = "unknown"
            signal["evidence"] = []
            _append_uncertainty(payload, f"traffic_control.{key} 값이 low confidence로 강등됨")
            continue
        signal["value"] = detected_signal.get("value")
        signal["confidence"] = detected_signal.get("confidence", signal.get("confidence", "unknown"))
        signal["evidence"] = detected_signal.get("evidence", signal.get("evidence", []))


def _verify_speed_and_distance(payload: dict[str, Any]) -> None:
    speed = payload.get("speed_and_distance")
    if not isinstance(speed, dict):
        return
    for estimate in speed.get("speed_estimates", []):
        if not isinstance(estimate, dict) or estimate.get("numeric_kmh") is None:
            continue
        if _confidence_rank(estimate.get("confidence")) < _confidence_rank("medium"):
            estimate["value"] = "모름"
            estimate["numeric_kmh"] = None
            estimate["range_kmh"] = None
            estimate["confidence"] = "unknown"
            _append_uncertainty(payload, "speed estimate가 high-precision 정책으로 강등됨")


def _verify_rag_hints(payload: dict[str, Any]) -> None:
    rag_hints = payload.get("rag_hints")
    if not isinstance(rag_hints, dict):
        return
    accident_type = rag_hints.get("accident_type")
    if accident_type in {None, "확인불가"}:
        return
    actors = payload.get("actors", [])
    confirmed_actor_count = sum(
        1
        for actor in actors
        if isinstance(actor, dict)
        and actor.get("evidence")
        and _confidence_rank(actor.get("confidence")) >= _confidence_rank("medium")
    )
    collision = payload.get("collision", {})
    collision_confirmed = isinstance(collision, dict) and (
        collision.get("detected") is True
        or collision.get("impact_type") not in {None, "확인불가"}
    )
    if confirmed_actor_count < 1 and not collision_confirmed:
        rag_hints["accident_type"] = "확인불가"
        _append_uncertainty(payload, "rag_hints.accident_type이 high-precision 정책으로 강등됨")


def _valid_evidence(evidence: Any, evidence_ids: set[str]) -> list[str]:
    if not isinstance(evidence, list):
        return []
    return [str(item) for item in evidence if str(item) in evidence_ids]


def _append_uncertainty(payload: dict[str, Any], message: str) -> None:
    payload.setdefault("uncertainties", [])
    if message not in payload["uncertainties"]:
        payload["uncertainties"].append(message)


def _confidence_rank(confidence: Any) -> int:
    return {"unknown": 0, "low": 1, "medium": 2, "high": 3}.get(str(confidence), 0)


def _add_precision_policy(payload: dict[str, Any]) -> None:
    payload.setdefault("evidence_index", {})
    if isinstance(payload["evidence_index"], dict):
        payload["evidence_index"]["precision_policy"] = {
            "target_precision": ">=90% estimated for retained facts",
            "strategy": "unsupported_or_low_confidence_facts_downgraded_to_unknown",
            "recall_tradeoff": "ambiguous facts may remain 확인불가",
        }
