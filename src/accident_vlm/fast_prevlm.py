from __future__ import annotations

import math
from pathlib import Path


class FastPreVlmContext:
    def __init__(self, data: dict):
        self._data = data

    def model_dump(self) -> dict:
        return self._data


def is_lightweight_fast_config(config) -> bool:
    return (
        getattr(config, "enable_ocr", True) is False
        and getattr(config, "enable_motion_keyframes", True) is False
        and getattr(config, "enable_actor_tracking", True) is False
        and getattr(config, "enable_event_scan", True) is False
        and getattr(config, "enable_input_quality", True) is False
        and getattr(config, "enable_contact_sheet", True) is False
        and getattr(config, "max_selected_frames", None) == 8
    )


def analyze_video_pre_vlm_fast(video_path: Path, config) -> FastPreVlmContext:
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {video_path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if fps <= 0 or frame_count < 0 or width <= 0 or height <= 0:
            raise ValueError(f"cannot read video metadata: {video_path}")
        duration_sec = frame_count / fps if frame_count else 0.0
        selected_frames = select_fast_regular_frames(
            duration_sec=duration_sec,
            fps=fps,
            frame_count=frame_count,
            interval_sec=config.regular_frame_interval_sec,
            max_frames=config.max_selected_frames,
        )
        frame_dir = config.output_dir / video_path.stem / config.frame_output_dirname
        frame_dir.mkdir(parents=True, exist_ok=True)
        extracted_frames = extract_fast_frames(capture, selected_frames, frame_dir)
    finally:
        capture.release()

    metadata = {
        "duration_sec": duration_sec,
        "fps": fps,
        "resolution": f"{width}x{height}",
        "frame_count": frame_count,
        "has_audio": False,
    }
    data = {
        "video_path": str(video_path),
        "video_metadata": metadata,
        "input_quality": None,
        "selected_frames": extracted_frames,
        "selected_segments": [],
        "event_scan_candidates": [],
        "rejected_frame_candidates": [],
        "ocr_observations": [],
        "ocr_summary": {},
        "scene_type_candidates": [],
        "tracks": [],
        "tracker_comparison": {},
        "road_geometry": {},
        "speed_and_distance": {},
        "traffic_control": {},
        "event_candidates": [],
        "preprocessing_uncertainties": [],
        "overlays": [],
        "crops": [],
        "contact_sheets": [],
        "evidence_images": [],
        "evidence_package": {
            "video_path": str(video_path),
            "metadata": metadata,
            "frames": extracted_frames,
            "overlays": [],
            "crops": [],
            "contact_sheets": [],
            "precomputed_facts": {"metadata": metadata},
        },
    }
    return FastPreVlmContext(data)


def select_fast_regular_frames(
    *,
    duration_sec: float,
    fps: float,
    frame_count: int,
    interval_sec: float,
    max_frames: int,
) -> list[dict]:
    if frame_count <= 0:
        return []
    frames: list[dict] = []
    seen_indices: set[int] = set()
    step_count = math.floor((duration_sec / interval_sec) + 1e-9)
    max_frame_index = frame_count - 1
    for step_index in range(step_count + 1):
        frame_index = int(round(step_index * interval_sec * fps))
        if frame_index in seen_indices or frame_index > max_frame_index:
            continue
        seen_indices.add(frame_index)
        frames.append(
            {
                "id": f"frame_{frame_index:06d}",
                "time": frame_to_timecode(frame_index, fps),
                "frame_index": frame_index,
                "path": None,
                "purpose": "regular_context",
            }
        )
    if len(frames) <= max_frames:
        return frames
    if max_frames == 1:
        return [frames[0]]
    last_index = len(frames) - 1
    selected_indices = {
        round(index * last_index / (max_frames - 1)) for index in range(max_frames)
    }
    return [frame for index, frame in enumerate(frames) if index in selected_indices]


def extract_fast_frames(capture, frames: list[dict], frame_dir: Path) -> list[dict]:
    import cv2

    extracted: list[dict] = []
    current_frame = 0
    for frame in frames:
        target_frame = frame["frame_index"]
        while current_frame < target_frame:
            if not capture.grab():
                return extracted
            current_frame += 1
        ok, image = capture.read()
        if not ok:
            return extracted
        current_frame += 1
        frame_path = frame_dir / f"{frame['id']}.jpg"
        if cv2.imwrite(str(frame_path), image):
            extracted.append({**frame, "path": str(frame_path)})
    return extracted


def frame_to_timecode(frame_index: int, fps: float) -> str:
    total_seconds = frame_index / fps if fps else 0.0
    minutes = int(total_seconds // 60)
    seconds = total_seconds - minutes * 60
    return f"{minutes:02d}:{seconds:06.3f}"
