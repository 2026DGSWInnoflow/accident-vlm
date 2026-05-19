def seconds_to_timecode(seconds: float) -> str:
    if seconds < 0:
        raise ValueError("seconds must be non-negative")

    total_millis = int(round(seconds * 1000))
    total_seconds, millis = divmod(total_millis, 1000)
    minutes, secs = divmod(total_seconds, 60)

    return f"{minutes:02d}:{secs:02d}.{millis:03d}"


def frame_to_timecode(frame_index: int, fps: float) -> str:
    if frame_index < 0:
        raise ValueError("frame_index must be non-negative")
    if fps <= 0:
        raise ValueError("fps must be positive")

    return seconds_to_timecode(frame_index / fps)
