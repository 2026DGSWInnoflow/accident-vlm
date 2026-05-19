from decimal import Decimal, ROUND_FLOOR
from pathlib import Path

import cv2

from accident_vlm.schemas.preprocessing import SelectedFrame
from accident_vlm.utils.timecode import frame_to_timecode


def _floor_decimal(value: Decimal) -> int:
    return int(value.to_integral_value(rounding=ROUND_FLOOR))


def select_regular_frames(
    duration_sec: float, fps: float, interval_sec: float
) -> list[SelectedFrame]:
    if duration_sec < 0:
        raise ValueError("duration_sec must be non-negative")
    if fps <= 0:
        raise ValueError("fps must be positive")
    if interval_sec <= 0:
        raise ValueError("interval_sec must be positive")

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

    return frames


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
