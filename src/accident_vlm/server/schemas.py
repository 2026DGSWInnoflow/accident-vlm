from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class AnalysisMode(StrEnum):
    PRE_VLM = "pre_vlm"
    FULL = "full"


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class AnalysisOptions(BaseModel):
    mode: AnalysisMode = AnalysisMode.PRE_VLM
    ocr_backend: str = "auto"
    object_detector_backend: str = "none"
    object_detector_model: str = "yolov8x.pt"
    qwen_model_id: str = "/home/minsung0830/accident-vlm/models/Qwen3.6-27B"
    device: str = "auto"
    regular_frame_interval_sec: float = Field(default=1.0, gt=0)
    max_selected_frames: int = Field(default=16, gt=0)
    enable_motion_keyframes: bool = True
    enable_segment_tracking: bool = True
    max_motion_keyframes: int = Field(default=8, gt=0)
    motion_sample_interval_sec: float = Field(default=0.5, gt=0)
    min_motion_change_score: float = Field(default=12.0, ge=0)
    pre_event_window_sec: float = Field(default=5.0, gt=0)
    post_event_window_sec: float = Field(default=3.0, gt=0)
    segment_tracking_stride_frames: int = Field(default=3, gt=0)
    max_segment_tracking_frames: int = Field(default=90, gt=0)
    lane_width_m: float = Field(default=3.2, gt=0)


class PathAnalysisRequest(BaseModel):
    video_path: Path
    options: AnalysisOptions = Field(default_factory=AnalysisOptions)


class JobRecord(BaseModel):
    job_id: str
    status: JobStatus
    mode: AnalysisMode
    video_path: str
    created_at: str
    updated_at: str
    output_dir: str
    pre_vlm_output_path: str | None = None
    final_output_path: str | None = None
    error: str | None = None


class JobCreatedResponse(BaseModel):
    job_id: str
    status: JobStatus
    status_url: str
    result_url: str


class ResultResponse(BaseModel):
    job: JobRecord
    result: dict[str, Any]
