from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from accident_vlm.modules.image_io import read_cv_image
from accident_vlm.schemas.preprocessing import SelectedFrame, VideoMetadata
from accident_vlm.utils.timecode import frame_to_timecode, parse_timecode, seconds_to_timecode
from accident_vlm.modules.video_sampling import iter_sampled_capture_frames

_EVENT_SCAN_FLOW_SIZE = (96, 54)
_FLOW_SKIP_FRAME_DIFF_THRESHOLD = 1.0
_FLOW_SKIP_HISTOGRAM_CHANGE_THRESHOLD = 1.0


def scan_video_event_candidates(
    video_path: Path,
    metadata: VideoMetadata,
    sample_fps: float = 5.0,
    top_k: int = 5,
    min_score: float = 8.0,
    pre_event_window_sec: float = 6.0,
    post_event_window_sec: float = 4.0,
    bbox_area_change_by_frame: dict[int, float] | None = None,
) -> list[dict[str, Any]]:
    if sample_fps <= 0:
        raise ValueError("sample_fps must be positive")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if min_score < 0:
        raise ValueError("min_score must be non-negative")
    if metadata.fps <= 0:
        raise ValueError("metadata.fps must be positive")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {video_path}")

    sample_step = max(1, int(round(metadata.fps / sample_fps)))
    previous_gray: np.ndarray | None = None
    previous_hist: np.ndarray | None = None
    previous_frame_index: int | None = None
    raw_candidates: list[dict[str, Any]] = []
    try:
        for frame_index, image in iter_sampled_capture_frames(
            capture,
            metadata.frame_count,
            sample_step,
        ):
            small_image = cv2.resize(image, _EVENT_SCAN_FLOW_SIZE, interpolation=cv2.INTER_AREA)
            small_gray = cv2.cvtColor(small_image, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([small_gray], [0], None, [32], [0, 256])
            cv2.normalize(hist, hist)
            if previous_gray is None or previous_hist is None or previous_frame_index is None:
                previous_gray = small_gray
                previous_hist = hist
                previous_frame_index = frame_index
                continue

            correlation = float(cv2.compareHist(previous_hist, hist, cv2.HISTCMP_CORREL))
            histogram_change = max(0.0, 1.0 - correlation) * 100.0
            bbox_area_change = (
                float(bbox_area_change_by_frame.get(frame_index, 0.0))
                if bbox_area_change_by_frame
                else 0.0
            )
            frame_diff = float(cv2.absdiff(small_gray, previous_gray).mean())
            if _can_skip_flow(frame_diff, histogram_change, bbox_area_change):
                optical_flow_peak = 0.0
                optical_flow_mean = 0.0
            else:
                flow = cv2.calcOpticalFlowFarneback(
                    previous_gray,
                    small_gray,
                    None,
                    0.5,
                    3,
                    15,
                    3,
                    5,
                    1.2,
                    0,
                )
                flow_magnitude = np.linalg.norm(flow, axis=2)
                optical_flow_peak = _fast_percentile_value(flow_magnitude, 95.0)
                optical_flow_mean = float(flow_magnitude.mean())
            score = _event_scan_score(frame_diff, optical_flow_peak, optical_flow_mean, histogram_change)
            if bbox_area_change:
                score = max(0, min(100, int(round(score + min(25.0, bbox_area_change * 10.0)))))
            if score >= min_score:
                raw_candidates.append(
                    _build_event_scan_candidate(
                        frame_index=frame_index,
                        previous_frame_index=previous_frame_index,
                        metadata=metadata,
                        event_score=score,
                        frame_diff=frame_diff,
                        optical_flow_peak=optical_flow_peak,
                        optical_flow_mean=optical_flow_mean,
                        histogram_change=histogram_change,
                        bbox_area_change=bbox_area_change,
                        pre_event_window_sec=pre_event_window_sec,
                        post_event_window_sec=post_event_window_sec,
                    )
                )
            previous_gray = small_gray
            previous_hist = hist
            previous_frame_index = frame_index
    finally:
        capture.release()

    return _suppress_nearby_candidates(raw_candidates, metadata.fps, top_k)


def _event_scan_score(
    frame_diff: float,
    optical_flow_peak: float,
    optical_flow_mean: float,
    histogram_change: float,
) -> int:
    score = frame_diff * 1.4 + optical_flow_peak * 7.0 + optical_flow_mean * 3.0 + histogram_change * 0.9
    return max(0, min(100, int(round(score))))


def _can_skip_flow(frame_diff: float, histogram_change: float, bbox_area_change: float) -> bool:
    return (
        bbox_area_change <= 0
        and frame_diff < _FLOW_SKIP_FRAME_DIFF_THRESHOLD
        and histogram_change < _FLOW_SKIP_HISTOGRAM_CHANGE_THRESHOLD
    )


def _fast_percentile_value(values: np.ndarray, percentile: float) -> float:
    if values.size == 0:
        return 0.0
    flat = values.reshape(-1)
    rank = int(np.ceil(flat.size * percentile / 100.0)) - 1
    rank = max(0, min(flat.size - 1, rank))
    return float(np.partition(flat, rank)[rank])


def _build_event_scan_candidate(
    *,
    frame_index: int,
    previous_frame_index: int,
    metadata: VideoMetadata,
    event_score: int,
    frame_diff: float,
    optical_flow_peak: float,
    optical_flow_mean: float,
    histogram_change: float,
    bbox_area_change: float = 0.0,
    pre_event_window_sec: float,
    post_event_window_sec: float,
) -> dict[str, Any]:
    seconds = frame_index / metadata.fps
    start = max(0.0, seconds - pre_event_window_sec)
    end = min(metadata.duration_sec, seconds + post_event_window_sec)
    confidence = "high" if event_score >= 70 else "medium" if event_score >= 40 else "low"
    supporting_signals = {
        "frame_diff_mean": round(frame_diff, 3),
        "optical_flow_peak": round(optical_flow_peak, 3),
        "camera_shake_peak": round(optical_flow_peak, 3),
        "optical_flow_mean": round(optical_flow_mean, 3),
        "histogram_change": round(histogram_change, 3),
    }
    if bbox_area_change:
        supporting_signals["bbox_area_change"] = round(bbox_area_change, 3)
    return {
        "id": f"event_scan_{frame_index:06d}",
        "time": frame_to_timecode(frame_index, metadata.fps),
        "frame_index": frame_index,
        "event_type": "event_scan_peak",
        "event_score": event_score,
        "confidence": confidence,
        "source": "high_fps_event_scan",
        "signals": [
            "frame_diff",
            "optical_flow",
            "camera_shake_peak",
            "histogram_change",
            *(['bbox_area_change'] if bbox_area_change else []),
        ],
        "supporting_signals": supporting_signals,
        "contradicting_signals": [],
        "evidence": [
            f"frame_{previous_frame_index:06d}",
            f"frame_{frame_index:06d}",
        ],
        "window": {
            "start": seconds_to_timecode(start),
            "end": seconds_to_timecode(end),
            "pre_event_sec": round(seconds - start, 3),
            "post_event_sec": round(end - seconds, 3),
        },
    }


def _suppress_nearby_candidates(
    candidates: list[dict[str, Any]],
    fps: float,
    top_k: int,
    min_gap_sec: float = 0.4,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    min_gap_frames = max(1, int(round(fps * min_gap_sec)))
    for candidate in sorted(candidates, key=lambda item: -int(item.get("event_score", 0))):
        frame_index = int(candidate.get("frame_index", 0))
        if any(abs(frame_index - int(item.get("frame_index", 0))) < min_gap_frames for item in selected):
            continue
        selected.append(candidate)
        if len(selected) >= top_k:
            break
    return sorted(selected, key=lambda item: int(item.get("frame_index", 0)))


def select_precision_event_frames(
    event_candidates: list[dict[str, Any]],
    metadata: VideoMetadata,
    max_frames: int = 20,
    pre_event_window_sec: float = 6.0,
    post_event_window_sec: float = 4.0,
    precision_fps: float = 15.0,
    min_impact_frames: int = 5,
) -> tuple[list[SelectedFrame], list[dict[str, Any]]]:
    if max_frames <= 0:
        raise ValueError("max_frames must be positive")
    if precision_fps <= 0:
        raise ValueError("precision_fps must be positive")
    if min_impact_frames <= 0:
        raise ValueError("min_impact_frames must be positive")
    center_seconds = _primary_candidate_seconds(event_candidates)
    if center_seconds is None:
        return [], []

    start = max(0.0, center_seconds - pre_event_window_sec)
    end = min(metadata.duration_sec, center_seconds + post_event_window_sec)
    impact_start = max(start, center_seconds - 0.35)
    impact_end = min(end, center_seconds + 0.35)
    pre_focus_start = max(start, center_seconds - 2.5)
    pre_context_end = max(start, pre_focus_start)
    post_start = min(end, impact_end)

    groups = [
        ("event_scan_pre_context", start, pre_context_end, 5),
        ("event_scan_pre_impact", pre_focus_start, center_seconds, 6),
        ("event_scan_impact_candidate", impact_start, impact_end, max(min_impact_frames, 5)),
        ("event_scan_post_impact", post_start, end, 4),
    ]
    candidates: list[SelectedFrame] = []
    for purpose, group_start, group_end, count in groups:
        for seconds in _sample_seconds_by_fps(group_start, group_end, precision_fps, count):
            frame_index = _seconds_to_frame_index(seconds, metadata)
            candidates.append(
                SelectedFrame(
                    id=f"frame_{frame_index:06d}",
                    time=frame_to_timecode(frame_index, metadata.fps),
                    frame_index=frame_index,
                    purpose=purpose,
                )
            )

    deduped = _dedupe_frames(candidates)
    selected = _limit_frames_with_impact_quota(deduped, max_frames, min_impact_frames)
    selected_keys = {frame.frame_index for frame in selected}
    rejected = [
        {
            "id": frame.id,
            "time": frame.time,
            "frame_index": frame.frame_index,
            "purpose": frame.purpose,
            "reason": "vlm_frame_budget_limit",
        }
        for frame in deduped
        if frame.frame_index not in selected_keys
    ]
    return selected, rejected


def _primary_candidate_seconds(event_candidates: list[dict[str, Any]]) -> float | None:
    for event in sorted(
        [event for event in event_candidates if isinstance(event, dict)],
        key=lambda item: -float(item.get("event_score", 0.0) or 0.0),
    ):
        try:
            return parse_timecode(str(event.get("time", "")))
        except ValueError:
            continue
    return None


def _sample_seconds_by_fps(start: float, end: float, fps: float, min_count: int) -> list[float]:
    if end <= start:
        return [start]
    step = 1.0 / fps
    values: list[float] = []
    current = start
    while current <= end + 1e-9:
        values.append(current)
        current += step
    if len(values) < min_count:
        values = _sample_evenly(start, end, min_count)
    return values


def _sample_evenly(start: float, end: float, count: int) -> list[float]:
    if count <= 1 or end <= start:
        return [start]
    step = (end - start) / (count - 1)
    return [start + index * step for index in range(count)]


def _seconds_to_frame_index(seconds: float, metadata: VideoMetadata) -> int:
    if metadata.frame_count <= 0:
        return 0
    return max(0, min(metadata.frame_count - 1, int(round(seconds * metadata.fps))))


def _dedupe_frames(frames: list[SelectedFrame]) -> list[SelectedFrame]:
    by_index: dict[int, SelectedFrame] = {}
    for frame in frames:
        existing = by_index.get(frame.frame_index)
        if existing is None:
            by_index[frame.frame_index] = frame
            continue
        purposes = existing.purpose.split("+")
        if frame.purpose not in purposes:
            by_index[frame.frame_index] = existing.model_copy(
                update={"purpose": f"{existing.purpose}+{frame.purpose}"}
            )
    return [by_index[index] for index in sorted(by_index)]


def _limit_frames_with_impact_quota(
    frames: list[SelectedFrame],
    max_frames: int,
    min_impact_frames: int,
) -> list[SelectedFrame]:
    if len(frames) <= max_frames:
        return frames
    impact = [frame for frame in frames if "impact_candidate" in frame.purpose]
    non_impact = [frame for frame in frames if "impact_candidate" not in frame.purpose]
    selected_impact = _limit_evenly(impact, min(max(min_impact_frames, 1), len(impact), max_frames))
    remaining = max_frames - len(selected_impact)
    selected = [*selected_impact, *_limit_evenly(non_impact, remaining)]
    return sorted(_dedupe_frames(selected), key=lambda frame: frame.frame_index)[:max_frames]


def _limit_evenly(frames: list[SelectedFrame], max_frames: int) -> list[SelectedFrame]:
    if max_frames <= 0:
        return []
    if len(frames) <= max_frames:
        return frames
    if max_frames == 1:
        return [frames[0]]
    last_index = len(frames) - 1
    selected_indices = {
        round(index * last_index / (max_frames - 1)) for index in range(max_frames)
    }
    return [frame for index, frame in enumerate(frames) if index in selected_indices]


def build_frame_selection_contact_sheet(
    selected_frames: list[SelectedFrame],
    output_path: Path,
    title: str = "frame_selection",
    thumb_width: int = 240,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    readable_frame_paths = [Path(str(frame.path)) for frame in selected_frames if frame.path]
    if _is_reusable_contact_sheet(output_path, readable_frame_paths):
        return {
            "id": "contact_sheet_frame_selection",
            "path": str(output_path),
            "purpose": "frame_selection_contact_sheet",
            "source": "frame_selection",
            "frame_count": len(readable_frame_paths),
            "status": "reused",
        }

    images: list[np.ndarray] = []
    for frame in selected_frames:
        if not frame.path:
            continue
        image = read_cv_image(frame.path)
        if image is None:
            continue
        height, width = image.shape[:2]
        scale = thumb_width / max(width, 1)
        thumb = cv2.resize(
            image,
            (thumb_width, max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
        label = f"{frame.time} {frame.purpose}"
        cv2.rectangle(thumb, (0, 0), (thumb.shape[1], 24), (0, 0, 0), -1)
        cv2.putText(
            thumb,
            label[:48],
            (6, 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        images.append(thumb)
    if not images:
        return {
            "id": "contact_sheet_frame_selection",
            "path": str(output_path),
            "purpose": "frame_selection_contact_sheet",
            "source": "frame_selection",
            "frame_count": 0,
            "status": "not_created",
            "reason": "no readable selected frame images",
        }

    columns = min(5, len(images))
    cell_height = max(image.shape[0] for image in images)
    rows = int(np.ceil(len(images) / columns))
    sheet = np.full((rows * cell_height + 34, columns * thumb_width, 3), 245, dtype=np.uint8)
    cv2.putText(
        sheet,
        title,
        (8, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    for index, image in enumerate(images):
        row = index // columns
        column = index % columns
        y = 34 + row * cell_height
        x = column * thumb_width
        sheet[y : y + image.shape[0], x : x + image.shape[1]] = image
    cv2.imwrite(str(output_path), sheet)
    return {
        "id": "contact_sheet_frame_selection",
        "path": str(output_path),
        "purpose": "frame_selection_contact_sheet",
        "source": "frame_selection",
        "frame_count": len(images),
        "status": "created",
    }


def _is_reusable_contact_sheet(output_path: Path, frame_paths: list[Path]) -> bool:
    if not frame_paths:
        return False
    try:
        output_mtime_ns = output_path.stat().st_mtime_ns
    except OSError:
        return False
    for frame_path in frame_paths:
        try:
            if frame_path.stat().st_mtime_ns > output_mtime_ns:
                return False
        except OSError:
            return False
    return True
