from accident_vlm.modules.schema_guard import (
    find_forbidden_terms,
    repair_and_constrain_payload,
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


def test_validate_final_output_normalizes_vlm_status_synonyms_without_mutating_input():
    payload = _minimal_final_output_payload("차량과 보행자가 횡단보도 부근에서 관찰됨")
    payload["scene_type"]["status"] = "confirmed"

    output = validate_final_output(payload)

    assert output.scene_type.status == Status.OBSERVED
    assert payload["scene_type"]["status"] == "confirmed"


def test_repair_and_constrain_payload_fills_required_fields_and_unknowns_unsupported_values():
    payload = {
        "scene_type": {
            "value": "교차로",
            "status": "observed",
            "confidence": "high",
            "source": [],
            "evidence": [],
        },
        "rag_hints": {},
        "objective_summary": "신호 상태는 녹색으로 보임",
        "traffic_control": {"signal": {"value": "녹색", "evidence": []}},
    }

    repaired = repair_and_constrain_payload(payload)

    assert repaired["schema_version"] == "accident_video_facts.v1"
    assert repaired["scene_type"]["value"] == "확인불가"
    assert repaired["scene_type"]["status"] == "unknown"
    assert repaired["traffic_control"]["signal"]["value"] == "확인불가"
    assert "accident_datetime" in repaired["insurance_claim_fields"]
    assert "vehicle_to_vehicle" in repaired["accident_type_candidates"]
    assert "근거 없는 값이 확인불가로 조정됨: scene_type" in repaired["uncertainties"]


def test_repair_and_constrain_payload_guards_insurance_fields_and_accident_candidates():
    payload = _minimal_final_output_payload("객관 사실 요약")
    payload["insurance_claim_fields"] = {
        "road_shape": {
            "value": "교차로",
            "status": "observed",
            "confidence": "high",
            "source": ["lane_overlay"],
            "evidence": [],
        },
        "lane_count": {
            "value": "3차로",
            "status": "observed",
            "confidence": "medium",
            "source": [],
            "evidence": ["storyboard_slot_01"],
        },
        "ego_direction": {
            "value": "직진",
            "status": "observed",
            "confidence": "medium",
            "source": [],
            "evidence": [],
        },
    }
    payload["accident_type_candidates"] = {
        "vehicle_to_pedestrian": {
            "status": "observed",
            "confidence": "high",
            "source": ["actor_crop"],
            "evidence": [],
        },
        "vehicle_to_vehicle": {
            "status": "observed",
            "confidence": "medium",
            "source": [],
            "evidence": ["storyboard_slot_02"],
        },
        "single_vehicle": {
            "status": "observed",
            "confidence": "low",
            "source": [],
            "evidence": [],
        },
    }

    repaired = repair_and_constrain_payload(payload)

    assert repaired["insurance_claim_fields"]["road_shape"]["value"] == "교차로"
    assert repaired["insurance_claim_fields"]["lane_count"]["value"] == "3차로"
    assert repaired["insurance_claim_fields"]["ego_direction"]["value"] == "확인불가"
    assert repaired["accident_type_candidates"]["vehicle_to_pedestrian"]["status"] == "observed"
    assert repaired["accident_type_candidates"]["vehicle_to_vehicle"]["status"] == "observed"
    assert repaired["accident_type_candidates"]["single_vehicle"]["status"] == "unknown"
    assert "근거 없는 값이 확인불가로 조정됨: insurance_claim_fields.ego_direction" in repaired[
        "uncertainties"
    ]


def test_repair_and_constrain_payload_requires_evidence_for_speed_collision_and_timeline():
    payload = _minimal_final_output_payload("객관 사실 요약")
    payload["speed_and_distance"] = {
        "speed_estimates": [
            {
                "actor_id": "ego_vehicle",
                "value": "47km/h",
                "numeric_kmh": 47,
                "method": "ocr_overlay",
                "confidence": "high",
                "evidence": [],
            }
        ]
    }
    payload["collision"] = {
        "impact_type": "측면충돌",
        "confidence": "medium",
        "evidence": [],
    }
    payload["timeline"] = [
        {
            "time": "00:06.000",
            "event": "T1과 T2가 접촉함",
            "confidence": "medium",
            "evidence": [],
        }
    ]
    payload["actors"] = [
        {
            "id": "T1",
            "type": "승용차",
            "movement": "차로변경",
            "evidence": [],
        }
    ]

    repaired = repair_and_constrain_payload(payload)

    assert repaired["speed_and_distance"]["speed_estimates"][0]["value"] == "모름"
    assert repaired["speed_and_distance"]["speed_estimates"][0]["numeric_kmh"] is None
    assert repaired["collision"]["impact_type"] == "확인불가"
    assert repaired["timeline"][0]["event"] == "근거 부족으로 세부 사건 확인불가"
    assert repaired["actors"][0]["movement"] == "확인불가"
    assert "근거 없는 값이 확인불가로 조정됨: speed_and_distance.speed_estimates[0]" in repaired[
        "uncertainties"
    ]


def test_repair_and_constrain_payload_normalizes_structured_uncertainties():
    payload = _minimal_final_output_payload("객관 사실 요약")
    payload["uncertainties"] = [
        {"description": "영상 화질 저하", "source": ["chunk_1", "chunk_2"]},
        {"reason": "신호등 확인불가"},
        123,
    ]

    repaired = repair_and_constrain_payload(payload)

    assert repaired["uncertainties"] == [
        "영상 화질 저하 (source: chunk_1, chunk_2)",
        "신호등 확인불가",
        "123",
    ]


def test_validate_final_output_accepts_structured_uncertainties_from_vlm():
    payload = _minimal_final_output_payload("객관 사실 요약")
    payload["uncertainties"] = [
        {"description": "영상이 흐림", "source": ["chunk_1"]},
    ]

    output = validate_final_output(payload)

    assert output.uncertainties == ["영상이 흐림 (source: chunk_1)"]
