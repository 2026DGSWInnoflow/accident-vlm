import os
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
QUALITY_VLM_FRAME_BUDGET = 20
QUALITY_EVENT_SCAN_SAMPLE_FPS = 5.0
QUALITY_EVENT_SCAN_TOP_K = 5
QUALITY_EVENT_SCAN_MIN_SCORE = 8.0
QUALITY_PRECISION_EVENT_FPS = 15.0
QUALITY_MIN_IMPACT_FRAMES = 5
QUALITY_MAX_EVENT_CANDIDATES = 24
DEFAULT_QWEN_MODEL_ID = os.getenv("ACCIDENT_VLM_QWEN_MODEL_ID", "/home/minsung0830/accident-vlm/models/Qwen3.6-27B-AWQ-INT4")


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
    enable_tracker_comparison: bool = False
    enable_road_geometry: bool = True
    enable_speed_distance: bool = True
    enable_traffic_control: bool = True
    enable_event_detection: bool = True
    enable_event_scan: bool = True
    enable_vlm: bool = False
    frame_output_dirname: str = "frames"
    overlay_output_dirname: str = "overlays"
    crop_output_dirname: str = "crops"
    ocr_backend: str = "auto"
    object_detector_backend: str = QUALITY_OBJECT_DETECTOR_BACKEND
    object_detector_model: str = QUALITY_OBJECT_DETECTOR_MODEL
    qwen_model_id: str = DEFAULT_QWEN_MODEL_ID
    device: str = "auto"
    lane_width_m: float = Field(default=3.2, gt=0)
    lane_segmentation_model_path: Path | None = None
    motion_sample_interval_sec: float = Field(default=QUALITY_MOTION_SAMPLE_INTERVAL_SEC, gt=0)
    max_motion_keyframes: int = Field(default=QUALITY_MAX_MOTION_KEYFRAMES, gt=0)
    min_motion_change_score: float = Field(default=QUALITY_MIN_MOTION_CHANGE_SCORE, ge=0)
    segment_tracking_stride_frames: int = Field(default=QUALITY_SEGMENT_TRACKING_STRIDE_FRAMES, gt=0)
    max_segment_tracking_frames: int = Field(default=QUALITY_MAX_SEGMENT_TRACKING_FRAMES, gt=0)
    vlm_frame_budget: int = Field(default=QUALITY_VLM_FRAME_BUDGET, gt=0)
    event_scan_sample_fps: float = Field(default=QUALITY_EVENT_SCAN_SAMPLE_FPS, gt=0)
    event_scan_top_k: int = Field(default=QUALITY_EVENT_SCAN_TOP_K, gt=0)
    event_scan_min_score: float = Field(default=QUALITY_EVENT_SCAN_MIN_SCORE, ge=0)
    precision_event_fps: float = Field(default=QUALITY_PRECISION_EVENT_FPS, gt=0)
    min_impact_frames: int = Field(default=QUALITY_MIN_IMPACT_FRAMES, gt=0)
    max_event_candidates: int = Field(default=QUALITY_MAX_EVENT_CANDIDATES, gt=0)
