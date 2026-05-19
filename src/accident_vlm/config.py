from pathlib import Path

from pydantic import BaseModel, Field


class PipelineConfig(BaseModel):
    output_dir: Path = Path("outputs")
    regular_frame_interval_sec: float = Field(default=1.0, gt=0)
    max_selected_frames: int = Field(default=16, gt=0)
    pre_event_window_sec: float = Field(default=5.0, gt=0)
    post_event_window_sec: float = Field(default=3.0, gt=0)
    enable_ocr: bool = True
    enable_scene_analysis: bool = True
    enable_actor_tracking: bool = True
    enable_road_geometry: bool = True
    enable_speed_distance: bool = True
    enable_traffic_control: bool = True
    enable_event_detection: bool = True
    enable_vlm: bool = False
    frame_output_dirname: str = "frames"
    overlay_output_dirname: str = "overlays"
    crop_output_dirname: str = "crops"
    ocr_backend: str = "auto"
    object_detector_backend: str = "none"
    object_detector_model: str = "yolov8x.pt"
    qwen_model_id: str = "/home/minsung0830/accident-vlm/models/Qwen3.6-27B"
    device: str = "auto"
    lane_width_m: float = Field(default=3.2, gt=0)
