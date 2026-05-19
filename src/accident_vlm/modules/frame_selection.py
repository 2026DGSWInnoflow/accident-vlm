from decimal import Decimal, ROUND_FLOOR
from pathlib import Path

import cv2

from accident_vlm.schemas.preprocessing import SelectedFrame
from accident_vlm.schemas.preprocessing import VideoMetadata
from accident_vlm.utils.timecode import frame_to_timecode, parse_timecode, seconds_to_timecode


def _floor_decimal(value: Decimal) -> int:
    return int(value.to_integral_value(rounding=ROUND_FLOOR))


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
    duration = Decimal(str(duration_sec))
    fps_decimal = Decimal(str(fps))
    interval = Decimal(str(interval_sec))
    step_count = _floor_decimal(duration / interval)
    max_frame_index = _floor_decimal(duration * fps_decimal)

    for step_index in range(step_count + 1):
        current = float(Decimal(step_index) * interval)
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
        for frame_index in range(0, metadata.frame_count, sample_step):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, image = capture.read()
            if not ok:
                continue
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA)
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
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {video_path}")

    extracted: list[SelectedFrame] = []
    try:
        for frame in selected_frames:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame.frame_index)
            ok, image = capture.read()
            if not ok:
                continue
            frame_path = output_dir / f"{frame.id}.jpg"
            cv2.imwrite(str(frame_path), image)
            extracted.append(frame.model_copy(update={"path": str(frame_path)}))
    finally:
        capture.release()
    return extracted
