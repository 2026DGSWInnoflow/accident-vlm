from pydantic import BaseModel, Field, field_validator


class VideoMetadata(BaseModel):
    duration_sec: float = Field(ge=0, allow_inf_nan=False)
    fps: float = Field(gt=0, allow_inf_nan=False)
    resolution: str = Field(min_length=1)
    frame_count: int = Field(ge=0)
    has_audio: bool

    @field_validator("resolution")
    @classmethod
    def reject_blank_resolution(cls, value: str) -> str:
        stripped_value = value.strip()
        if not stripped_value:
            raise ValueError("resolution cannot be blank")
        return stripped_value


class InputQuality(BaseModel):
    blur: str
    brightness: str
    night_noise: str
    camera_shake: str
    occlusion: str
    analysis_reliability: str


class SelectedFrame(BaseModel):
    id: str = Field(min_length=1)
    time: str = Field(min_length=1, pattern=r"^\d{2,}:[0-5]\d\.\d{3}$")
    frame_index: int = Field(ge=0)
    path: str | None = None
    purpose: str = Field(min_length=1)

    @field_validator("id", "time", "purpose")
    @classmethod
    def reject_blank_strings(cls, value: str) -> str:
        stripped_value = value.strip()
        if not stripped_value:
            raise ValueError("field cannot be blank")
        return stripped_value


class PipelineContext(BaseModel):
    video_path: str
    video_metadata: VideoMetadata | None = None
    input_quality: InputQuality | None = None
    selected_frames: list[SelectedFrame] = Field(default_factory=list)
    selected_segments: list[dict] = Field(default_factory=list)
    ocr_observations: list[dict] = Field(default_factory=list)
    ocr_summary: dict = Field(default_factory=dict)
    scene_type_candidates: list[dict] = Field(default_factory=list)
    tracks: list[dict] = Field(default_factory=list)
    road_geometry: dict = Field(default_factory=dict)
    speed_and_distance: dict = Field(default_factory=dict)
    traffic_control: dict = Field(default_factory=dict)
    event_candidates: list[dict] = Field(default_factory=list)
    overlays: list[dict] = Field(default_factory=list)
    crops: list[dict] = Field(default_factory=list)
    evidence_package: dict = Field(default_factory=dict)
