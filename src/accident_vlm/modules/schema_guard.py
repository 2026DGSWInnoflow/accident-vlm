import json
from copy import deepcopy

from accident_vlm.schemas.final_output import AccidentFactOutput


FORBIDDEN_LEGAL_TERMS = [
    "가해",
    "피해",
    "과실",
    "위반",
    "불법",
    "책임",
    "주의의무",
    "신호위반",
    "안전거리 미확보",
]

LEGAL_JUDGMENT_REPLACEMENT = "[법적 판단 표현 제거]"

STATUS_SYNONYMS = {
    "confirmed": "observed",
    "detected": "observed",
    "visible": "observed",
    "observed": "observed",
    "computed": "computed",
    "estimated": "computed",
    "inferred": "inferred",
    "unknown": "unknown",
    "확인불가": "unknown",
}


INSURANCE_CLAIM_FIELD_DEFAULTS = {
    "accident_datetime": {
        "value": "확인불가",
        "status": "unknown",
        "confidence": "unknown",
        "source": [],
        "evidence": [],
        "note": "블랙박스 OCR/메타데이터에서 확인되지 않음",
    },
    "location": {
        "value": "확인불가",
        "status": "unknown",
        "confidence": "unknown",
        "source": [],
        "evidence": [],
        "note": "영상만으로 장소 특정 불가",
    },
    "road_shape": {
        "value": "확인불가",
        "status": "unknown",
        "confidence": "unknown",
        "source": [],
        "evidence": [],
    },
    "lane_count": {
        "value": "확인불가",
        "status": "unknown",
        "confidence": "unknown",
        "source": [],
        "evidence": [],
    },
    "ego_direction": {
        "value": "확인불가",
        "status": "unknown",
        "confidence": "unknown",
        "source": [],
        "evidence": [],
    },
    "other_direction": {
        "value": "확인불가",
        "status": "unknown",
        "confidence": "unknown",
        "source": [],
        "evidence": [],
    },
    "damage_parts": [],
    "police_report_visible": {
        "value": "확인불가",
        "status": "unknown",
        "confidence": "unknown",
        "source": [],
        "evidence": [],
    },
}

ACCIDENT_TYPE_CANDIDATE_DEFAULTS = {
    "vehicle_to_pedestrian": {
        "label": "차대보행자",
        "status": "unknown",
        "confidence": "unknown",
        "source": [],
        "evidence": [],
    },
    "vehicle_to_vehicle": {
        "label": "차대차",
        "status": "unknown",
        "confidence": "unknown",
        "source": [],
        "evidence": [],
    },
    "vehicle_to_motorcycle": {
        "label": "차대이륜차",
        "status": "unknown",
        "confidence": "unknown",
        "source": [],
        "evidence": [],
    },
    "vehicle_to_bicycle": {
        "label": "차대자전거",
        "status": "unknown",
        "confidence": "unknown",
        "source": [],
        "evidence": [],
    },
    "vehicle_to_kickboard": {
        "label": "차대전동킥보드",
        "status": "unknown",
        "confidence": "unknown",
        "source": [],
        "evidence": [],
    },
    "vehicle_to_facility": {
        "label": "차대시설물",
        "status": "unknown",
        "confidence": "unknown",
        "source": [],
        "evidence": [],
    },
    "single_vehicle": {
        "label": "단독사고",
        "status": "unknown",
        "confidence": "unknown",
        "source": [],
        "evidence": [],
    },
    "non_contact": {
        "label": "비접촉사고",
        "status": "unknown",
        "confidence": "unknown",
        "source": [],
        "evidence": [],
    },
    "multi_collision": {
        "label": "다중추돌",
        "status": "unknown",
        "confidence": "unknown",
        "source": [],
        "evidence": [],
    },
}


DEFAULT_PAYLOAD = {
    "schema_version": "accident_video_facts.v1",
    "input_quality": {},
    "scene_type": {
        "value": "확인불가",
        "status": "unknown",
        "confidence": "unknown",
        "source": [],
        "evidence": [],
    },
    "road_conditions": {},
    "traffic_control": {},
    "actors": [],
    "timeline": [],
    "collision": {},
    "speed_and_distance": {},
    "insurance_claim_fields": deepcopy(INSURANCE_CLAIM_FIELD_DEFAULTS),
    "accident_type_candidates": deepcopy(ACCIDENT_TYPE_CANDIDATE_DEFAULTS),
    "uncertainties": [],
    "evidence_index": {},
    "rag_hints": {"accident_type": "확인불가", "scenario_keywords": []},
    "objective_summary": "확인 가능한 객관 사실이 제한적임.",
}


def normalize_vlm_payload(payload: dict) -> dict:
    normalized = deepcopy(payload)

    def visit(value):
        if isinstance(value, dict):
            status = value.get("status")
            if isinstance(status, str):
                mapped = STATUS_SYNONYMS.get(status.strip().lower())
                if mapped is not None:
                    value["status"] = mapped
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(normalized)
    return normalized


def repair_and_constrain_payload(payload: dict) -> dict:
    repaired = deepcopy(DEFAULT_PAYLOAD)
    _deep_update(repaired, normalize_vlm_payload(payload))
    repaired["uncertainties"] = _normalize_uncertainties(repaired.get("uncertainties"))

    _require_evidence_for_field(repaired, "scene_type")
    signal = repaired.get("traffic_control", {}).get("signal")
    if isinstance(signal, dict) and signal.get("value") not in {None, "확인불가"}:
        if not signal.get("evidence") and not signal.get("crops"):
            signal["value"] = "확인불가"
            signal["status"] = "unknown"
            signal["confidence"] = "unknown"
            _append_uncertainty(repaired, "근거 없는 값이 확인불가로 조정됨: traffic_control.signal")
    _require_speed_evidence(repaired)
    _require_collision_evidence(repaired)
    _require_timeline_evidence(repaired)
    _require_actor_evidence(repaired)
    _require_insurance_field_evidence(repaired)
    _require_accident_type_candidate_evidence(repaired)
    return repaired


def _normalize_uncertainties(value) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        value = [value]

    normalized: list[str] = []
    for item in value:
        message = _uncertainty_to_string(item)
        if message and message not in normalized:
            normalized.append(message)
    return normalized


def _uncertainty_to_string(value) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("description", "message", "reason", "note", "value"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                source = value.get("source")
                if isinstance(source, list) and source:
                    return f"{candidate.strip()} (source: {', '.join(map(str, source))})"
                return candidate.strip()
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value).strip()


def _deep_update(target: dict, source: dict) -> None:
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = deepcopy(value)


def _require_evidence_for_field(payload: dict, key: str) -> None:
    field = payload.get(key)
    if not isinstance(field, dict):
        return
    status = field.get("status")
    has_evidence = bool(field.get("source")) and bool(field.get("evidence"))
    if status != "unknown" and not has_evidence:
        field["value"] = "확인불가"
        field["status"] = "unknown"
        field["confidence"] = "unknown"
        field["source"] = []
        field["evidence"] = []
        _append_uncertainty(payload, f"근거 없는 값이 확인불가로 조정됨: {key}")


def _require_evidence_for_nested_field(payload: dict, field: dict, path: str) -> None:
    status = field.get("status")
    if status == "unknown":
        return
    if _has_evidence(field):
        return
    if "value" in field:
        field["value"] = "확인불가"
    field["status"] = "unknown"
    field["confidence"] = "unknown"
    field["source"] = []
    field["evidence"] = []
    _append_uncertainty(payload, f"근거 없는 값이 확인불가로 조정됨: {path}")


def _append_uncertainty(payload: dict, message: str) -> None:
    if message not in payload["uncertainties"]:
        payload["uncertainties"].append(message)


def _has_evidence(value: dict) -> bool:
    return bool(value.get("evidence") or value.get("source") or value.get("crops"))


def _require_speed_evidence(payload: dict) -> None:
    estimates = payload.get("speed_and_distance", {}).get("speed_estimates", [])
    if not isinstance(estimates, list):
        return
    for index, estimate in enumerate(estimates):
        if not isinstance(estimate, dict) or estimate.get("numeric_kmh") is None:
            continue
        if _has_evidence(estimate):
            continue
        estimate["value"] = "모름"
        estimate["numeric_kmh"] = None
        estimate["range_kmh"] = None
        estimate["confidence"] = "unknown"
        _append_uncertainty(
            payload,
            f"근거 없는 값이 확인불가로 조정됨: speed_and_distance.speed_estimates[{index}]",
        )


def _require_collision_evidence(payload: dict) -> None:
    collision = payload.get("collision")
    if not isinstance(collision, dict) or not collision:
        return
    if _has_evidence(collision):
        return
    for key in ("impact_type", "impact_location", "damage_or_injury"):
        if key in collision:
            collision[key] = "확인불가"
    collision["confidence"] = "unknown"
    _append_uncertainty(payload, "근거 없는 값이 확인불가로 조정됨: collision")


def _require_timeline_evidence(payload: dict) -> None:
    timeline = payload.get("timeline", [])
    if not isinstance(timeline, list):
        return
    for index, event in enumerate(timeline):
        if not isinstance(event, dict) or _has_evidence(event):
            continue
        event["event"] = "근거 부족으로 세부 사건 확인불가"
        event["confidence"] = "unknown"
        _append_uncertainty(payload, f"근거 없는 값이 확인불가로 조정됨: timeline[{index}]")


def _require_actor_evidence(payload: dict) -> None:
    actors = payload.get("actors", [])
    if not isinstance(actors, list):
        return
    for index, actor in enumerate(actors):
        if not isinstance(actor, dict):
            continue
        if actor.get("movement") not in {None, "확인불가"} and not _has_evidence(actor):
            actor["movement"] = "확인불가"
            actor["confidence"] = "unknown"
            _append_uncertainty(payload, f"근거 없는 값이 확인불가로 조정됨: actors[{index}].movement")


def _require_insurance_field_evidence(payload: dict) -> None:
    fields = payload.get("insurance_claim_fields")
    if not isinstance(fields, dict):
        payload["insurance_claim_fields"] = deepcopy(INSURANCE_CLAIM_FIELD_DEFAULTS)
        return
    _deep_update(fields, {key: value for key, value in INSURANCE_CLAIM_FIELD_DEFAULTS.items() if key not in fields})
    for key, field in fields.items():
        if not isinstance(field, dict):
            continue
        _require_evidence_for_nested_field(payload, field, f"insurance_claim_fields.{key}")


def _require_accident_type_candidate_evidence(payload: dict) -> None:
    candidates = payload.get("accident_type_candidates")
    if not isinstance(candidates, dict):
        payload["accident_type_candidates"] = deepcopy(ACCIDENT_TYPE_CANDIDATE_DEFAULTS)
        return
    _deep_update(
        candidates,
        {key: value for key, value in ACCIDENT_TYPE_CANDIDATE_DEFAULTS.items() if key not in candidates},
    )
    for key, candidate in candidates.items():
        if not isinstance(candidate, dict):
            continue
        _require_evidence_for_nested_field(payload, candidate, f"accident_type_candidates.{key}")


def _matched_spans(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    for term in FORBIDDEN_LEGAL_TERMS:
        start = text.find(term)
        while start != -1:
            spans.append((start, start + len(term), term))
            start = text.find(term, start + len(term))
    return spans


def _is_subspan_of_longer_match(
    candidate: tuple[int, int, str], spans: list[tuple[int, int, str]]
) -> bool:
    candidate_start, candidate_end, candidate_term = candidate
    for start, end, term in spans:
        if len(term) <= len(candidate_term):
            continue
        if start <= candidate_start and candidate_end <= end:
            return True
    return False


def find_forbidden_terms(text: str) -> list[str]:
    """Return forbidden legal judgment terms in configured order."""
    spans = _matched_spans(text)
    matched_terms = {
        term
        for span_start, span_end, term in spans
        if not _is_subspan_of_longer_match((span_start, span_end, term), spans)
    }
    return [term for term in FORBIDDEN_LEGAL_TERMS if term in matched_terms]


def sanitize_summary(text: str) -> str:
    sanitized = text
    spans = _matched_spans(text)
    replacement_spans = [
        (start, end)
        for start, end, term in spans
        if not _is_subspan_of_longer_match((start, end, term), spans)
    ]
    for start, end in sorted(replacement_spans, reverse=True):
        sanitized = sanitized[:start] + LEGAL_JUDGMENT_REPLACEMENT + sanitized[end:]
    return sanitized


def validate_final_output(payload: dict) -> AccidentFactOutput:
    output = AccidentFactOutput.model_validate(repair_and_constrain_payload(payload))
    forbidden = find_forbidden_terms(output.objective_summary)
    if not forbidden:
        return output

    return output.model_copy(
        update={
            "objective_summary": sanitize_summary(output.objective_summary),
            "uncertainties": [
                *output.uncertainties,
                f"법적 판단 표현이 제거됨: {', '.join(forbidden)}",
            ],
        }
    )
