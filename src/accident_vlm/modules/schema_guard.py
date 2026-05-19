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
    if not isinstance(repaired.get("uncertainties"), list):
        repaired["uncertainties"] = [str(repaired["uncertainties"])]

    _require_evidence_for_field(repaired, "scene_type")
    signal = repaired.get("traffic_control", {}).get("signal")
    if isinstance(signal, dict) and signal.get("value") not in {None, "확인불가"}:
        if not signal.get("evidence") and not signal.get("crops"):
            signal["value"] = "확인불가"
            signal["status"] = "unknown"
            signal["confidence"] = "unknown"
            _append_uncertainty(repaired, "근거 없는 값이 확인불가로 조정됨: traffic_control.signal")
    return repaired


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


def _append_uncertainty(payload: dict, message: str) -> None:
    if message not in payload["uncertainties"]:
        payload["uncertainties"].append(message)


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
