from accident_vlm.modules.schema_guard import (
    find_forbidden_terms,
    sanitize_summary,
    validate_final_output,
)
from accident_vlm.schemas.common import Confidence, Status
from accident_vlm.schemas.final_output import AccidentFactOutput, AccidentType, SceneType


def _minimal_final_output_payload(objective_summary: str, uncertainties: list[str] | None = None):
    payload = {
        "scene_type": {
            "value": SceneType.ROAD,
            "status": Status.OBSERVED,
            "confidence": Confidence.MEDIUM,
            "source": ["visual"],
            "evidence": ["frame_000030"],
        },
        "rag_hints": {
            "accident_type": AccidentType.VEHICLE_TO_VEHICLE,
            "scenario_keywords": [],
        },
        "objective_summary": objective_summary,
    }
    if uncertainties is not None:
        payload["uncertainties"] = uncertainties
    return payload


def test_find_forbidden_terms_returns_matches_in_configured_order():
    assert find_forbidden_terms("상대 차량의 신호위반과 과실이 있는 것으로 보임") == [
        "과실",
        "신호위반",
    ]


def test_find_forbidden_terms_prefers_specific_overlapping_terms():
    assert find_forbidden_terms("신호위반") == ["신호위반"]


def test_sanitize_summary_replaces_legal_judgment_terms():
    sanitized = sanitize_summary("상대 차량의 과실이 관찰됨")

    assert sanitized == "상대 차량의 [법적 판단 표현 제거]이 관찰됨"
    assert "과실" not in sanitized


def test_sanitize_summary_replaces_specific_overlapping_term_once():
    assert sanitize_summary("신호위반") == "[법적 판단 표현 제거]"


def test_validate_final_output_sanitizes_summary_and_records_uncertainty():
    output = validate_final_output(
        _minimal_final_output_payload(
            "상대 차량의 과실이 관찰됨",
            uncertainties=["차량 속도는 확인불가"],
        )
    )

    assert isinstance(output, AccidentFactOutput)
    assert output.objective_summary == "상대 차량의 [법적 판단 표현 제거]이 관찰됨"
    assert output.uncertainties == [
        "차량 속도는 확인불가",
        "법적 판단 표현이 제거됨: 과실",
    ]


def test_validate_final_output_does_not_mutate_input_payload():
    payload = _minimal_final_output_payload(
        "상대 차량의 과실이 관찰됨",
        uncertainties=["existing"],
    )

    output = validate_final_output(payload)

    assert "과실" not in output.objective_summary
    assert output.uncertainties == [
        "existing",
        "법적 판단 표현이 제거됨: 과실",
    ]
    assert payload["objective_summary"] == "상대 차량의 과실이 관찰됨"
    assert payload["uncertainties"] == ["existing"]
