import os
from dataclasses import dataclass
from pathlib import Path


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


@dataclass
class PipelineConfig:
    output_dir: Path = Path("outputs")
    regular_frame_interval_sec: float = QUALITY_REGULAR_FRAME_INTERVAL_SEC
    max_selected_frames: int = QUALITY_MAX_SELECTED_FRAMES
    pre_event_window_sec: float = QUALITY_PRE_EVENT_WINDOW_SEC
    post_event_window_sec: float = QUALITY_POST_EVENT_WINDOW_SEC
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
    lane_width_m: float = 3.2
    lane_segmentation_model_path: Path | None = None
    motion_sample_interval_sec: float = QUALITY_MOTION_SAMPLE_INTERVAL_SEC
    max_motion_keyframes: int = QUALITY_MAX_MOTION_KEYFRAMES
    min_motion_change_score: float = QUALITY_MIN_MOTION_CHANGE_SCORE
    segment_tracking_stride_frames: int = QUALITY_SEGMENT_TRACKING_STRIDE_FRAMES
    max_segment_tracking_frames: int = QUALITY_MAX_SEGMENT_TRACKING_FRAMES
    vlm_frame_budget: int = QUALITY_VLM_FRAME_BUDGET
    event_scan_sample_fps: float = QUALITY_EVENT_SCAN_SAMPLE_FPS
    event_scan_top_k: int = QUALITY_EVENT_SCAN_TOP_K
    event_scan_min_score: float = QUALITY_EVENT_SCAN_MIN_SCORE
    precision_event_fps: float = QUALITY_PRECISION_EVENT_FPS
    min_impact_frames: int = QUALITY_MIN_IMPACT_FRAMES
    max_event_candidates: int = QUALITY_MAX_EVENT_CANDIDATES

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        if self.lane_segmentation_model_path is not None:
            self.lane_segmentation_model_path = Path(self.lane_segmentation_model_path)

        for name in (
            "regular_frame_interval_sec",
            "max_selected_frames",
            "pre_event_window_sec",
            "post_event_window_sec",
            "lane_width_m",
            "motion_sample_interval_sec",
            "max_motion_keyframes",
            "segment_tracking_stride_frames",
            "max_segment_tracking_frames",
            "vlm_frame_budget",
            "event_scan_sample_fps",
            "event_scan_top_k",
            "precision_event_fps",
            "min_impact_frames",
            "max_event_candidates",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("min_motion_change_score", "event_scan_min_score"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
