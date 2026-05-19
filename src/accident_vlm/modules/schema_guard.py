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
    output = AccidentFactOutput.model_validate(payload)
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
