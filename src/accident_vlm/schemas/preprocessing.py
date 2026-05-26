from pydantic import BaseModel, ConfigDict, Field, field_validator


class VideoMetadata(BaseModel):
    model_config = ConfigDict(defer_build=True)

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
    model_config = ConfigDict(defer_build=True)

    blur: str
    brightness: str
    night_noise: str
    camera_shake: str
    camera_shake_score: dict = Field(default_factory=dict)
    occlusion: str
    analysis_reliability: str
    timeline: list[dict] = Field(default_factory=list)
    segment_quality: list[dict] = Field(default_factory=list)
    visibility_conditions: dict = Field(default_factory=dict)


class SelectedFrame(BaseModel):
    model_config = ConfigDict(defer_build=True)

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
    model_config = ConfigDict(defer_build=True)

    video_path: str
    video_metadata: VideoMetadata | None = None
    input_quality: InputQuality | None = None
    selected_frames: list[SelectedFrame] = Field(default_factory=list)
    selected_segments: list[dict] = Field(default_factory=list)
    event_scan_candidates: list[dict] = Field(default_factory=list)
    rejected_frame_candidates: list[dict] = Field(default_factory=list)
    ocr_observations: list[dict] = Field(default_factory=list)
    ocr_summary: dict = Field(default_factory=dict)
    scene_type_candidates: list[dict] = Field(default_factory=list)
    tracks: list[dict] = Field(default_factory=list)
    tracker_comparison: dict = Field(default_factory=dict)
    road_geometry: dict = Field(default_factory=dict)
    speed_and_distance: dict = Field(default_factory=dict)
    traffic_control: dict = Field(default_factory=dict)
    event_candidates: list[dict] = Field(default_factory=list)
    preprocessing_uncertainties: list[str] = Field(default_factory=list)
    overlays: list[dict] = Field(default_factory=list)
    crops: list[dict] = Field(default_factory=list)
    contact_sheets: list[dict] = Field(default_factory=list)
    evidence_images: list[dict] = Field(default_factory=list)
    evidence_package: dict = Field(default_factory=dict)
