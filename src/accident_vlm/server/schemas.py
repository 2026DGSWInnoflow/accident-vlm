from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from accident_vlm.config import (
    QUALITY_MAX_MOTION_KEYFRAMES,
    QUALITY_MAX_SELECTED_FRAMES,
    QUALITY_MAX_SEGMENT_TRACKING_FRAMES,
    QUALITY_MIN_MOTION_CHANGE_SCORE,
    QUALITY_MOTION_SAMPLE_INTERVAL_SEC,
    QUALITY_OBJECT_DETECTOR_BACKEND,
    QUALITY_OBJECT_DETECTOR_MODEL,
    QUALITY_POST_EVENT_WINDOW_SEC,
    QUALITY_PRE_EVENT_WINDOW_SEC,
    QUALITY_REGULAR_FRAME_INTERVAL_SEC,
    QUALITY_SEGMENT_TRACKING_STRIDE_FRAMES,
    QUALITY_VLM_FRAME_BUDGET,
)


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
    object_detector_backend: str = QUALITY_OBJECT_DETECTOR_BACKEND
    object_detector_model: str = QUALITY_OBJECT_DETECTOR_MODEL
    qwen_model_id: str = "/home/minsung0830/accident-vlm/models/Qwen3.6-27B"
    device: str = "auto"
    regular_frame_interval_sec: float = Field(default=QUALITY_REGULAR_FRAME_INTERVAL_SEC, gt=0)
    max_selected_frames: int = Field(default=QUALITY_MAX_SELECTED_FRAMES, gt=0)
    enable_motion_keyframes: bool = True
    enable_segment_tracking: bool = True
    max_motion_keyframes: int = Field(default=QUALITY_MAX_MOTION_KEYFRAMES, gt=0)
    motion_sample_interval_sec: float = Field(default=QUALITY_MOTION_SAMPLE_INTERVAL_SEC, gt=0)
    min_motion_change_score: float = Field(default=QUALITY_MIN_MOTION_CHANGE_SCORE, ge=0)
    pre_event_window_sec: float = Field(default=QUALITY_PRE_EVENT_WINDOW_SEC, gt=0)
    post_event_window_sec: float = Field(default=QUALITY_POST_EVENT_WINDOW_SEC, gt=0)
    segment_tracking_stride_frames: int = Field(default=QUALITY_SEGMENT_TRACKING_STRIDE_FRAMES, gt=0)
    max_segment_tracking_frames: int = Field(default=QUALITY_MAX_SEGMENT_TRACKING_FRAMES, gt=0)
    vlm_frame_budget: int = Field(default=QUALITY_VLM_FRAME_BUDGET, gt=0)
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
