from pathlib import Path

from pydantic import BaseModel, Field


QUALITY_REGULAR_FRAME_INTERVAL_SEC = 0.5
QUALITY_MAX_SELECTED_FRAMES = 32
QUALITY_PRE_EVENT_WINDOW_SEC = 6.0
QUALITY_POST_EVENT_WINDOW_SEC = 4.0
QUALITY_OBJECT_DETECTOR_BACKEND = "bytetrack"
QUALITY_OBJECT_DETECTOR_MODEL = "yolov8x.pt"
QUALITY_MOTION_SAMPLE_INTERVAL_SEC = 0.25
QUALITY_MAX_MOTION_KEYFRAMES = 16
QUALITY_MIN_MOTION_CHANGE_SCORE = 6.0
QUALITY_SEGMENT_TRACKING_STRIDE_FRAMES = 2
QUALITY_MAX_SEGMENT_TRACKING_FRAMES = 180


class PipelineConfig(BaseModel):
    output_dir: Path = Path("outputs")
    regular_frame_interval_sec: float = Field(default=QUALITY_REGULAR_FRAME_INTERVAL_SEC, gt=0)
    max_selected_frames: int = Field(default=QUALITY_MAX_SELECTED_FRAMES, gt=0)
    pre_event_window_sec: float = Field(default=QUALITY_PRE_EVENT_WINDOW_SEC, gt=0)
    post_event_window_sec: float = Field(default=QUALITY_POST_EVENT_WINDOW_SEC, gt=0)
    enable_ocr: bool = True
    enable_motion_keyframes: bool = True
    enable_scene_analysis: bool = True
    enable_actor_tracking: bool = True
    enable_segment_tracking: bool = True
    enable_road_geometry: bool = True
    enable_speed_distance: bool = True
    enable_traffic_control: bool = True
    enable_event_detection: bool = True
    enable_vlm: bool = False
    frame_output_dirname: str = "frames"
    overlay_output_dirname: str = "overlays"
    crop_output_dirname: str = "crops"
    ocr_backend: str = "auto"
    object_detector_backend: str = QUALITY_OBJECT_DETECTOR_BACKEND
    object_detector_model: str = QUALITY_OBJECT_DETECTOR_MODEL
    qwen_model_id: str = "/home/minsung0830/accident-vlm/models/Qwen3.6-27B"
    device: str = "auto"
    lane_width_m: float = Field(default=3.2, gt=0)
    motion_sample_interval_sec: float = Field(default=QUALITY_MOTION_SAMPLE_INTERVAL_SEC, gt=0)
    max_motion_keyframes: int = Field(default=QUALITY_MAX_MOTION_KEYFRAMES, gt=0)
    min_motion_change_score: float = Field(default=QUALITY_MIN_MOTION_CHANGE_SCORE, ge=0)
    segment_tracking_stride_frames: int = Field(default=QUALITY_SEGMENT_TRACKING_STRIDE_FRAMES, gt=0)
    max_segment_tracking_frames: int = Field(default=QUALITY_MAX_SEGMENT_TRACKING_FRAMES, gt=0)
