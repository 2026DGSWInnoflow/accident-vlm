from collections.abc import Iterator
from typing import Any


def iter_sampled_capture_frames(
    capture: Any,
    frame_count: int,
    sample_step: int,
) -> Iterator[tuple[int, Any]]:
    if frame_count <= 0:
        return
    if sample_step <= 0:
        raise ValueError("sample_step must be positive")

    current_frame = 0
    for target_frame in range(0, frame_count, sample_step):
        while current_frame < target_frame:
            if not capture.grab():
                return
            current_frame += 1

        ok, image = capture.read()
        if not ok:
            return
        yield target_frame, image
        current_frame += 1
