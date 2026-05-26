import math
from pathlib import Path

import cv2

from accident_vlm.modules.image_io import cache_cv_image
from accident_vlm.schemas.preprocessing import SelectedFrame
from accident_vlm.schemas.preprocessing import VideoMetadata
from accident_vlm.utils.timecode import frame_to_timecode, parse_timecode, seconds_to_timecode
from accident_vlm.modules.video_sampling import (
    iter_capture_frames_at_indices,
    iter_sampled_capture_frames,
)

_MOTION_DIFF_SIZE = (96, 54)


_FLOAT_BOUNDARY_EPSILON = 1e-9


def select_regular_frames(
    duration_sec: float,
    fps: float,
    interval_sec: float,
    max_frames: int | None = None,
) -> list[SelectedFrame]:
    if duration_sec < 0:
        raise ValueError("duration_sec must be non-negative")
    if fps <= 0:
        raise ValueError("fps must be positive")
    if interval_sec <= 0:
        raise ValueError("interval_sec must be positive")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive")

    frames: list[SelectedFrame] = []
    seen_frame_indices: set[int] = set()
    step_count = math.floor((duration_sec / interval_sec) + _FLOAT_BOUNDARY_EPSILON)
    max_frame_index = math.floor((duration_sec * fps) + _FLOAT_BOUNDARY_EPSILON)

    for step_index in range(step_count + 1):
        current = step_index * interval_sec
        frame_index = int(round(current * fps))
        if frame_index in seen_frame_indices or frame_index > max_frame_index:
            continue

        seen_frame_indices.add(frame_index)
        frames.append(
            SelectedFrame(
                id=f"frame_{frame_index:06d}",
                time=frame_to_timecode(frame_index, fps),
                frame_index=frame_index,
                purpose="regular_context",
            )
        )

    if max_frames is not None and len(frames) > max_frames:
        if max_frames == 1:
            return [frames[0]]
        last_index = len(frames) - 1
        selected_indices = {
            round(index * last_index / (max_frames - 1)) for index in range(max_frames)
        }
        return [frame for index, frame in enumerate(frames) if index in selected_indices]

    return frames


def select_motion_keyframes(
    video_path: Path,
    metadata: VideoMetadata,
    sample_interval_sec: float = 0.5,
    max_frames: int = 8,
    min_change_score: float = 12.0,
) -> list[SelectedFrame]:
    if sample_interval_sec <= 0:
        raise ValueError("sample_interval_sec must be positive")
    if max_frames <= 0:
        raise ValueError("max_frames must be positive")
    if min_change_score < 0:
        raise ValueError("min_change_score must be non-negative")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {video_path}")

    sample_step = max(1, int(round(metadata.fps * sample_interval_sec)))
    previous_gray = None
    scored_frames: list[tuple[float, int]] = []
    try:
        for frame_index, image in iter_sampled_capture_frames(
            capture,
            metadata.frame_count,
            sample_step,
        ):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, _MOTION_DIFF_SIZE, interpolation=cv2.INTER_AREA)
            if previous_gray is not None:
                score = float(cv2.absdiff(gray, previous_gray).mean())
                if score >= min_change_score:
                    scored_frames.append((score, frame_index))
            previous_gray = gray
    finally:
        capture.release()

    selected = sorted(
        sorted(scored_frames, key=lambda item: item[0], reverse=True)[:max_frames],
        key=lambda item: item[1],
    )
    return [
        SelectedFrame(
            id=f"frame_{frame_index:06d}",
            time=frame_to_timecode(frame_index, metadata.fps),
            frame_index=frame_index,
            purpose="motion_keyframe",
        )
        for _, frame_index in selected
    ]


def merge_selected_frames(*frame_groups: list[SelectedFrame]) -> list[SelectedFrame]:
    frames_by_index: dict[int, SelectedFrame] = {}
    for frame in [item for group in frame_groups for item in group]:
        existing = frames_by_index.get(frame.frame_index)
        if existing is None:
            frames_by_index[frame.frame_index] = frame
            continue
        purposes = existing.purpose.split("+")
        if frame.purpose not in purposes:
            frames_by_index[frame.frame_index] = existing.model_copy(
                update={"purpose": f"{existing.purpose}+{frame.purpose}"}
            )

    return [frames_by_index[index] for index in sorted(frames_by_index)]


def select_event_window_frames(
    event_candidates: list[dict],
    metadata: VideoMetadata,
    max_frames: int = 20,
    pre_event_window_sec: float = 6.0,
    post_event_window_sec: float = 4.0,
) -> list[SelectedFrame]:
    if max_frames <= 0:
        raise ValueError("max_frames must be positive")
    if pre_event_window_sec < 0:
        raise ValueError("pre_event_window_sec must be non-negative")
    if post_event_window_sec < 0:
        raise ValueError("post_event_window_sec must be non-negative")

    center_seconds = _primary_event_seconds(event_candidates)
    if center_seconds is None:
        return select_regular_frames(
            duration_sec=metadata.duration_sec,
            fps=metadata.fps,
            interval_sec=max(metadata.duration_sec / max(max_frames - 1, 1), 0.001),
            max_frames=max_frames,
        )

    start = max(0.0, center_seconds - pre_event_window_sec)
    end = min(metadata.duration_sec, center_seconds + post_event_window_sec)
    impact_start = max(start, center_seconds - 0.75)
    impact_end = min(end, center_seconds + 0.75)
    pre_focus_start = max(start, center_seconds - 2.5)
    pre_context_end = max(start, pre_focus_start)
    post_start = min(end, impact_end)

    groups = [
        ("pre_context", start, pre_context_end, 5),
        ("pre_impact", pre_focus_start, center_seconds, 6),
        ("impact_candidate", impact_start, impact_end, 5),
        ("post_impact", post_start, end, 4),
    ]

    frames: list[SelectedFrame] = []
    for purpose, group_start, group_end, count in groups:
        for seconds in _sample_seconds(group_start, group_end, count):
            frame_index = _seconds_to_frame_index(seconds, metadata)
            frames.append(
                SelectedFrame(
                    id=f"frame_{frame_index:06d}",
                    time=frame_to_timecode(frame_index, metadata.fps),
                    frame_index=frame_index,
                    purpose=purpose,
                )
            )
    merged = merge_selected_frames(frames)
    if len(merged) < max_frames:
        fill_frames = [
            SelectedFrame(
                id=f"frame_{_seconds_to_frame_index(seconds, metadata):06d}",
                time=frame_to_timecode(_seconds_to_frame_index(seconds, metadata), metadata.fps),
                frame_index=_seconds_to_frame_index(seconds, metadata),
                purpose="event_window_context",
            )
            for seconds in _sample_seconds(start, end, max_frames)
        ]
        merged = merge_selected_frames(merged, fill_frames)
    return _limit_evenly(merged, max_frames)


def _primary_event_seconds(event_candidates: list[dict]) -> float | None:
    for event in sorted(
        [event for event in event_candidates if isinstance(event, dict)],
        key=lambda item: -float(item.get("event_score", 0.0) or 0.0),
    ):
        try:
            return parse_timecode(str(event.get("time", "")))
        except ValueError:
            continue
    return None


def _sample_seconds(start: float, end: float, count: int) -> list[float]:
    if count <= 0:
        return []
    if end <= start or count == 1:
        return [start]
    step = (end - start) / (count - 1)
    return [start + index * step for index in range(count)]


def _seconds_to_frame_index(seconds: float, metadata: VideoMetadata) -> int:
    if metadata.frame_count <= 0:
        return 0
    return max(0, min(metadata.frame_count - 1, int(round(seconds * metadata.fps))))


def _limit_evenly(frames: list[SelectedFrame], max_frames: int) -> list[SelectedFrame]:
    if len(frames) <= max_frames:
        return frames
    if max_frames == 1:
        return [frames[0]]
    last_index = len(frames) - 1
    selected_indices = {
        round(index * last_index / (max_frames - 1)) for index in range(max_frames)
    }
    return [frame for index, frame in enumerate(frames) if index in selected_indices]


def build_event_segments(
    event_candidates: list[dict],
    metadata: VideoMetadata,
    pre_event_window_sec: float,
    post_event_window_sec: float,
) -> list[dict]:
    if pre_event_window_sec < 0:
        raise ValueError("pre_event_window_sec must be non-negative")
    if post_event_window_sec < 0:
        raise ValueError("post_event_window_sec must be non-negative")

    segments: list[dict] = []
    for event in event_candidates:
        try:
            center_seconds = parse_timecode(str(event.get("time", "")))
        except ValueError:
            continue
        start_seconds = max(0.0, center_seconds - pre_event_window_sec)
        end_seconds = min(metadata.duration_sec, center_seconds + post_event_window_sec)
        segments.append(
            {
                "id": f"seg_event_{len(segments) + 1:03d}",
                "start": seconds_to_timecode(start_seconds),
                "end": seconds_to_timecode(end_seconds),
                "center_time": seconds_to_timecode(center_seconds),
                "reason": [event.get("event_type", "event_candidate")],
                "actors": event.get("actors", []),
                "confidence": event.get("confidence", "unknown"),
                "evidence": event.get("evidence", []),
            }
        )
    return segments


def extract_selected_frames(
    video_path: Path,
    selected_frames: list[SelectedFrame],
    output_dir: Path,
) -> list[SelectedFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    extracted_by_id: dict[str, SelectedFrame] = {}
    frames_to_extract: list[SelectedFrame] = []
    video_mtime_ns = _safe_mtime_ns(video_path)
    for frame in selected_frames:
        frame_path = output_dir / f"{frame.id}.jpg"
        if _is_reusable_frame_path(frame_path, video_mtime_ns):
            extracted_by_id[frame.id] = frame.model_copy(update={"path": str(frame_path)})
        else:
            frames_to_extract.append(frame)

    if not frames_to_extract:
        return [extracted_by_id[frame.id] for frame in selected_frames if frame.id in extracted_by_id]

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {video_path}")

    frames_by_index: dict[int, list[SelectedFrame]] = {}
    for frame in frames_to_extract:
        frames_by_index.setdefault(frame.frame_index, []).append(frame)
    if not frames_by_index:
        capture.release()
        return []

    try:
        for target_frame, image in iter_capture_frames_at_indices(
            capture,
            list(frames_by_index),
        ):
            for frame in frames_by_index[target_frame]:
                frame_path = output_dir / f"{frame.id}.jpg"
                if cv2.imwrite(str(frame_path), image):
                    cache_cv_image(frame_path, image)
                extracted_by_id[frame.id] = frame.model_copy(update={"path": str(frame_path)})
    finally:
        capture.release()
    return [extracted_by_id[frame.id] for frame in selected_frames if frame.id in extracted_by_id]


def _safe_mtime_ns(path: Path) -> int | None:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return None


def _is_reusable_frame_path(frame_path: Path, video_mtime_ns: int | None) -> bool:
    try:
        frame_mtime_ns = frame_path.stat().st_mtime_ns
    except OSError:
        return False
    return video_mtime_ns is None or frame_mtime_ns >= video_mtime_ns
