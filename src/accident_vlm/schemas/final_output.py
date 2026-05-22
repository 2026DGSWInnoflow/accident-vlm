from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from accident_vlm.schemas.common import EvidenceField


class SceneType(StrEnum):
    INTERSECTION = "교차로"
    CROSSWALK = "횡단보도"
    ROAD = "일반도로"
    HIGHWAY = "고속도로"
    MOTORWAY = "자동차전용도로"
    PARKING_LOT = "주차장"
    ALLEY = "골목길"
    SIDE_ROAD = "이면도로"
    ROUNDABOUT = "회전교차로"
    TUNNEL = "터널"
    BRIDGE = "교량"
    RAMP = "램프구간"
    MERGE = "합류구간"
    DIVERGE = "분기구간"
    UNKNOWN = "확인불가"


class AccidentType(StrEnum):
    VEHICLE_TO_VEHICLE = "차대차"
    VEHICLE_TO_PEDESTRIAN = "차대보행자"
    VEHICLE_TO_BICYCLE = "차대자전거"
    VEHICLE_TO_MOTORCYCLE = "차대이륜차"
    VEHICLE_TO_KICKBOARD = "차대전동킥보드"
    VEHICLE_TO_FACILITY = "차대시설물"
    SINGLE_VEHICLE = "단독사고"
    NON_CONTACT = "비접촉사고"
    MULTI_COLLISION = "다중추돌"
    UNKNOWN = "확인불가"


class AccidentFactOutput(BaseModel):
    schema_version: str = "accident_video_facts.v1"
    input_quality: dict[str, Any] = Field(default_factory=dict)
    scene_type: EvidenceField
    road_conditions: dict[str, Any] = Field(default_factory=dict)
    traffic_control: dict[str, Any] = Field(default_factory=dict)
    actors: list[dict[str, Any]] = Field(default_factory=list)
    timeline: list[dict[str, Any]] = Field(default_factory=list)
    collision: dict[str, Any] = Field(default_factory=dict)
    speed_and_distance: dict[str, Any] = Field(default_factory=dict)
    insurance_claim_fields: dict[str, Any] = Field(default_factory=dict)
    accident_type_candidates: dict[str, Any] = Field(default_factory=dict)
    uncertainties: list[str] = Field(default_factory=list)
    evidence_index: dict[str, Any] = Field(default_factory=dict)
    rag_hints: dict[str, Any]
    objective_summary: str
