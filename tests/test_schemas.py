import pytest
from pydantic import ValidationError

from accident_vlm.schemas.common import Confidence, EvidenceField, Status
from accident_vlm.schemas.final_output import AccidentFactOutput, AccidentType, SceneType
from accident_vlm.schemas.preprocessing import SelectedFrame, VideoMetadata


def test_evidence_field_accepts_unknown_without_evidence():
    field = EvidenceField(
        value="확인불가",
        status=Status.UNKNOWN,
        confidence=Confidence.UNKNOWN,
        note="영상에서 확인되지 않음",
    )

    assert field.value == "확인불가"
    assert field.status == Status.UNKNOWN
    assert field.confidence == Confidence.UNKNOWN
    assert field.source == []
    assert field.evidence == []
    assert field.note == "영상에서 확인되지 않음"


def test_evidence_field_rejects_observed_without_source():
    with pytest.raises(ValidationError):
        EvidenceField(value="주간", status=Status.OBSERVED, confidence=Confidence.HIGH)


def test_final_output_minimal_contract():
    output = AccidentFactOutput(
        scene_type=EvidenceField(
            value=SceneType.ROAD,
            status=Status.OBSERVED,
            confidence=Confidence.MEDIUM,
            source=["visual"],
            evidence=["frame_000030"],
        ),
        rag_hints={
            "accident_type": AccidentType.VEHICLE_TO_VEHICLE,
            "scenario_keywords": [],
        },
        objective_summary="자차와 상대 차량의 접촉이 관찰됨.",
    )

    assert output.schema_version == "accident_video_facts.v1"
    assert output.scene_type.value == SceneType.ROAD
    assert output.scene_type.status == Status.OBSERVED
    assert output.scene_type.confidence == Confidence.MEDIUM
    assert output.scene_type.source == ["visual"]
    assert output.scene_type.evidence == ["frame_000030"]
    assert output.rag_hints == {
        "accident_type": AccidentType.VEHICLE_TO_VEHICLE,
        "scenario_keywords": [],
    }
    assert output.insurance_claim_fields == {}
    assert output.accident_type_candidates == {}
    assert output.objective_summary == "자차와 상대 차량의 접촉이 관찰됨."


@pytest.mark.parametrize(
    "field_name, field_value",
    [
        ("source", [""]),
        ("source", ["   "]),
        ("evidence", [""]),
        ("evidence", ["   "]),
    ],
)
def test_evidence_field_rejects_blank_source_and_evidence_items(field_name, field_value):
    kwargs = {
        "value": "일반도로",
        "status": Status.OBSERVED,
        "confidence": Confidence.MEDIUM,
        "source": ["visual"],
        "evidence": ["frame_000030"],
    }
    kwargs[field_name] = field_value

    with pytest.raises(ValidationError):
        EvidenceField(**kwargs)


@pytest.mark.parametrize(
    "field_name, field_value",
    [
        ("duration_sec", -0.1),
        ("duration_sec", float("inf")),
        ("fps", 0),
        ("fps", float("inf")),
        ("resolution", ""),
        ("resolution", "   "),
        ("frame_count", -1),
    ],
)
def test_video_metadata_rejects_impossible_values(field_name, field_value):
    kwargs = {
        "duration_sec": 10.0,
        "fps": 30.0,
        "resolution": "1920x1080",
        "frame_count": 300,
        "has_audio": True,
    }
    kwargs[field_name] = field_value

    with pytest.raises(ValidationError):
        VideoMetadata(**kwargs)


@pytest.mark.parametrize(
    "field_name, field_value",
    [
        ("id", ""),
        ("id", "   "),
        ("time", ""),
        ("time", "   "),
        ("time", "1.0"),
        ("time", "00:60.000"),
        ("frame_index", -1),
        ("purpose", ""),
        ("purpose", "   "),
    ],
)
def test_selected_frame_rejects_invalid_boundary_values(field_name, field_value):
    kwargs = {
        "id": "frame_000030",
        "time": "00:01.000",
        "frame_index": 30,
        "purpose": "context",
    }
    kwargs[field_name] = field_value

    with pytest.raises(ValidationError):
        SelectedFrame(**kwargs)


def test_schemas_package_exports_input_quality():
    from accident_vlm.schemas import InputQuality

    quality = InputQuality(
        blur="low",
        brightness="normal",
        night_noise="low",
        camera_shake="low",
        occlusion="low",
        analysis_reliability="high",
    )

    assert quality.analysis_reliability == "high"


def test_importing_preprocessing_schema_does_not_load_final_output_schema() -> None:
    import subprocess
    import sys

    script = """
import sys
from accident_vlm.schemas.preprocessing import SelectedFrame
print("accident_vlm.schemas.final_output" in sys.modules)
print("accident_vlm.schemas.common" in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip().splitlines() == ["False", "False"]
