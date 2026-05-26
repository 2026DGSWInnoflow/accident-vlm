import json
import math
import shutil
import subprocess
from collections import OrderedDict
from fractions import Fraction
from pathlib import Path
from typing import Any

from accident_vlm.schemas.preprocessing import VideoMetadata


class IngestionError(ValueError):
    """Raised when ffprobe output cannot be converted into video metadata."""


_FFPROBE_AVAILABLE: bool | None = None
_VIDEO_METADATA_CACHE_MAX_SIZE = 64
_VIDEO_METADATA_CACHE: OrderedDict[tuple[str, int, int], Any] = OrderedDict()


def parse_ffprobe_streams(data: dict[str, Any]) -> VideoMetadata:
    streams = data.get("streams", [])
    video_stream = _first_video_stream(streams)
    has_audio = any(stream.get("codec_type") == "audio" for stream in streams)

    width = _parse_positive_int(video_stream.get("width"), "width")
    height = _parse_positive_int(video_stream.get("height"), "height")
    fps = _parse_frame_rate(video_stream)
    nb_frames = _parse_optional_positive_int(video_stream.get("nb_frames"))
    duration = _parse_duration(data, video_stream, has_frame_count=nb_frames is not None)
    frame_count = nb_frames if nb_frames is not None else int(round(duration * fps))

    return VideoMetadata(
        duration_sec=duration,
        fps=fps,
        resolution=f"{width}x{height}",
        frame_count=frame_count,
        has_audio=has_audio,
    )


def probe_video(video_path: Path) -> VideoMetadata:
    cache_key = _video_metadata_cache_key(video_path)
    if cache_key is not None:
        cached_metadata = _VIDEO_METADATA_CACHE.get(cache_key)
        if cached_metadata is not None:
            _VIDEO_METADATA_CACHE.move_to_end(cache_key)
            return cached_metadata

    metadata = _probe_video_uncached(video_path)
    if cache_key is not None:
        _VIDEO_METADATA_CACHE[cache_key] = metadata
        _VIDEO_METADATA_CACHE.move_to_end(cache_key)
        while len(_VIDEO_METADATA_CACHE) > _VIDEO_METADATA_CACHE_MAX_SIZE:
            _VIDEO_METADATA_CACHE.popitem(last=False)
    return metadata


def _probe_video_uncached(video_path: Path) -> VideoMetadata:
    global _FFPROBE_AVAILABLE
    if _FFPROBE_AVAILABLE is False:
        return probe_video_with_opencv(video_path)
    if _FFPROBE_AVAILABLE is None and shutil.which("ffprobe") is None:
        _FFPROBE_AVAILABLE = False
        return probe_video_with_opencv(video_path)

    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_streams",
                "-show_format",
                "-of",
                "json",
                str(video_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        _FFPROBE_AVAILABLE = True
        return parse_ffprobe_streams(json.loads(result.stdout))
    except FileNotFoundError:
        _FFPROBE_AVAILABLE = False
        return probe_video_with_opencv(video_path)


def _video_metadata_cache_key(video_path: Path) -> tuple[str, int, int] | None:
    try:
        stat = video_path.stat()
    except OSError:
        return None
    return (str(video_path.resolve()), stat.st_mtime_ns, stat.st_size)


def probe_video_with_opencv(video_path: Path) -> VideoMetadata:
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise IngestionError(f"Cannot open video: {video_path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()

    if not math.isfinite(fps) or fps <= 0:
        raise IngestionError("OpenCV could not determine a positive FPS")
    if frame_count < 0:
        raise IngestionError("OpenCV returned an invalid frame count")
    if width <= 0 or height <= 0:
        raise IngestionError("OpenCV could not determine video dimensions")

    return VideoMetadata(
        duration_sec=frame_count / fps if frame_count else 0.0,
        fps=fps,
        resolution=f"{width}x{height}",
        frame_count=frame_count,
        has_audio=False,
    )


def _first_video_stream(streams: Any) -> dict[str, Any]:
    if not isinstance(streams, list):
        raise IngestionError("Malformed ffprobe payload: streams must be a list")

    for stream in streams:
        if isinstance(stream, dict) and stream.get("codec_type") == "video":
            return stream

    raise IngestionError("No video stream found in ffprobe payload")


def _parse_frame_rate(video_stream: dict[str, Any]) -> float:
    for field_name in ("avg_frame_rate", "r_frame_rate"):
        raw_value = video_stream.get(field_name)
        if _is_missing_probe_value(raw_value):
            continue

        try:
            frame_rate = Fraction(str(raw_value))
        except (ValueError, ZeroDivisionError):
            continue

        if frame_rate > 0:
            return float(frame_rate)

    raise IngestionError("Invalid frame rate: expected positive avg_frame_rate or r_frame_rate")


def _parse_duration(
    data: dict[str, Any],
    video_stream: dict[str, Any],
    *,
    has_frame_count: bool,
) -> float:
    stream_duration = _parse_optional_non_negative_float(video_stream.get("duration"))
    if stream_duration is not None:
        return stream_duration

    format_data = data.get("format", {})
    format_duration = None
    if isinstance(format_data, dict):
        format_duration = _parse_optional_non_negative_float(format_data.get("duration"))
    if format_duration is not None:
        return format_duration

    if has_frame_count:
        return 0.0

    raise IngestionError("Missing valid duration and nb_frames; cannot compute frame_count")


def _parse_optional_non_negative_float(value: Any) -> float | None:
    if _is_missing_probe_value(value):
        return None

    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(parsed) or parsed < 0:
        return None

    return parsed


def _parse_positive_int(value: Any, field_name: str) -> int:
    parsed = _parse_optional_positive_int(value)
    if parsed is None:
        raise IngestionError(f"Invalid {field_name}: expected positive integer")
    return parsed


def _parse_optional_positive_int(value: Any) -> int | None:
    if _is_missing_probe_value(value) or isinstance(value, bool):
        return None

    if isinstance(value, int):
        return value if value > 0 else None

    if isinstance(value, str):
        stripped_value = value.strip()
        if stripped_value.isdecimal():
            parsed = int(stripped_value)
            return parsed if parsed > 0 else None

    return None


def _is_missing_probe_value(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip() in {"", "N/A"})
