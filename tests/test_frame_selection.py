from pathlib import Path
import cv2
import numpy as np
import pytest

from accident_vlm.modules.frame_selection import (
    build_event_segments,
    merge_selected_frames,
    select_event_window_frames,
    select_motion_keyframes,
    select_regular_frames,
)
from accident_vlm.schemas.preprocessing import VideoMetadata
from accident_vlm.schemas.preprocessing import SelectedFrame


def test_select_regular_frames_returns_expected_indices_and_times() -> None:
    frames = select_regular_frames(duration_sec=3.2, fps=30, interval_sec=1.0)

    assert [frame.frame_index for frame in frames] == [0, 30, 60, 90]
    assert [frame.time for frame in frames] == [
        "00:00.000",
        "00:01.000",
        "00:02.000",
        "00:03.000",
    ]
    assert all(isinstance(frame, SelectedFrame) for frame in frames)


def test_select_regular_frames_includes_exact_decimal_boundary() -> None:
    frames = select_regular_frames(duration_sec=0.3, fps=10, interval_sec=0.1)

    assert [frame.frame_index for frame in frames] == [0, 1, 2, 3]


def test_select_regular_frames_deduplicates_rounded_frame_indices() -> None:
    frames = select_regular_frames(duration_sec=2.0, fps=1, interval_sec=0.4)

    assert [frame.frame_index for frame in frames] == [0, 1, 2]
    assert len({frame.id for frame in frames}) == len(frames)
    assert len({frame.time for frame in frames}) == len(frames)


def test_select_regular_frames_does_not_emit_frames_after_duration() -> None:
    frames = select_regular_frames(duration_sec=0.51, fps=1, interval_sec=0.51)

    assert [frame.frame_index for frame in frames] == [0]


def test_select_regular_frames_zero_duration_returns_only_frame_zero() -> None:
    frames = select_regular_frames(duration_sec=0, fps=30, interval_sec=1.0)

    assert [frame.frame_index for frame in frames] == [0]


def test_select_regular_frames_evenly_limits_frame_count() -> None:
    frames = select_regular_frames(
        duration_sec=9,
        fps=10,
        interval_sec=1.0,
        max_frames=4,
    )

    assert [frame.frame_index for frame in frames] == [0, 30, 60, 90]


@pytest.mark.parametrize(
    "duration_sec, fps, interval_sec",
    [
        (-0.1, 30, 1.0),
        (3.2, 0, 1.0),
        (3.2, 30, 0),
    ],
)
def test_select_regular_frames_rejects_invalid_arguments(
    duration_sec: float, fps: float, interval_sec: float
) -> None:
    with pytest.raises(ValueError):
        select_regular_frames(
            duration_sec=duration_sec,
            fps=fps,
            interval_sec=interval_sec,
        )


def test_select_motion_keyframes_finds_high_change_frames(tmp_path) -> None:
    video_path = tmp_path / "motion.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10,
        (64, 48),
    )
    for index in range(10):
        image = np.zeros((48, 64, 3), dtype=np.uint8)
        if index >= 5:
            image[:, :] = 255
        writer.write(image)
    writer.release()

    frames = select_motion_keyframes(
        video_path,
        metadata=VideoMetadata(
            duration_sec=1.0,
            fps=10,
            resolution="64x48",
            frame_count=10,
            has_audio=False,
        ),
        sample_interval_sec=0.1,
        max_frames=2,
        min_change_score=20.0,
    )

    assert [frame.purpose for frame in frames] == ["motion_keyframe"]
    assert frames[0].frame_index in {5, 6}
    assert frames[0].time in {"00:00.500", "00:00.600"}


def test_merge_selected_frames_sorts_and_combines_duplicate_purposes() -> None:
    merged = merge_selected_frames(
        [
            SelectedFrame(
                id="frame_000030",
                time="00:01.000",
                frame_index=30,
                purpose="regular_context",
            )
        ],
        [
            SelectedFrame(
                id="frame_000030",
                time="00:01.000",
                frame_index=30,
                purpose="motion_keyframe",
            ),
            SelectedFrame(
                id="frame_000015",
                time="00:00.500",
                frame_index=15,
                purpose="motion_keyframe",
            ),
        ],
    )

    assert [frame.frame_index for frame in merged] == [15, 30]
    assert merged[1].purpose == "regular_context+motion_keyframe"


def test_select_event_window_frames_balances_before_impact_and_after() -> None:
    frames = select_event_window_frames(
        event_candidates=[
            {"time": "00:10.000", "event_type": "접촉", "event_score": 90},
        ],
        metadata=VideoMetadata(
            duration_sec=20.0,
            fps=10,
            resolution="640x480",
            frame_count=200,
            has_audio=False,
        ),
        max_frames=20,
        pre_event_window_sec=6.0,
        post_event_window_sec=4.0,
    )

    assert len(frames) == 20
    assert frames[0].time == "00:04.000"
    assert frames[-1].time == "00:14.000"
    assert any("pre_impact" in frame.purpose for frame in frames)
    assert any("impact_candidate" in frame.purpose for frame in frames)
    assert any("post_impact" in frame.purpose for frame in frames)


def test_select_event_window_frames_falls_back_to_regular_frames_without_event_time() -> None:
    frames = select_event_window_frames(
        event_candidates=[{"time": "확인불가", "event_score": 90}],
        metadata=VideoMetadata(
            duration_sec=9.0,
            fps=10,
            resolution="640x480",
            frame_count=90,
            has_audio=False,
        ),
        max_frames=5,
    )

    assert [frame.frame_index for frame in frames] == [0, 22, 45, 68, 90]
    assert all(frame.purpose == "regular_context" for frame in frames)


def test_build_event_segments_uses_pre_and_post_windows() -> None:
    segments = build_event_segments(
        event_candidates=[
            {
                "time": "00:06.000",
                "event_type": "접촉",
                "actors": ["T1", "T2"],
                "confidence": "medium",
                "evidence": ["frame_000180"],
            }
        ],
        metadata=VideoMetadata(
            duration_sec=10.0,
            fps=30,
            resolution="640x480",
            frame_count=300,
            has_audio=False,
        ),
        pre_event_window_sec=5.0,
        post_event_window_sec=3.0,
    )

    assert segments == [
        {
            "id": "seg_event_001",
            "start": "00:01.000",
            "end": "00:09.000",
            "center_time": "00:06.000",
            "reason": ["접촉"],
            "actors": ["T1", "T2"],
            "confidence": "medium",
            "evidence": ["frame_000180"],
        }
    ]


def test_build_event_segments_clamps_to_video_bounds_and_skips_unknown_time() -> None:
    segments = build_event_segments(
        event_candidates=[
            {"time": "확인불가", "event_type": "접근"},
            {"time": "00:01.000", "event_type": "급감속", "confidence": "low"},
        ],
        metadata=VideoMetadata(
            duration_sec=2.0,
            fps=30,
            resolution="640x480",
            frame_count=60,
            has_audio=False,
        ),
        pre_event_window_sec=5.0,
        post_event_window_sec=3.0,
    )

    assert segments[0]["start"] == "00:00.000"
    assert segments[0]["end"] == "00:02.000"


def test_select_motion_keyframes_scans_forward_without_random_seeks(monkeypatch, tmp_path) -> None:
    frames = []
    for index in range(6):
        image = np.zeros((48, 64, 3), dtype=np.uint8)
        if index >= 4:
            image[:, :] = 255
        frames.append(image)

    captures = []

    class FakeCapture:
        def __init__(self, path):
            self.index = 0
            self.set_calls = []
            captures.append(self)

        def isOpened(self):
            return True

        def set(self, prop, value):
            self.set_calls.append((prop, value))
            self.index = int(value)
            return True

        def grab(self):
            self.index += 1
            return self.index <= len(frames)

        def read(self):
            if self.index >= len(frames):
                return False, None
            image = frames[self.index]
            self.index += 1
            return True, image

        def release(self):
            pass

    monkeypatch.setattr(cv2, "VideoCapture", FakeCapture)

    selected = select_motion_keyframes(
        tmp_path / "fake.mp4",
        metadata=VideoMetadata(
            duration_sec=0.6,
            fps=10,
            resolution="64x48",
            frame_count=len(frames),
            has_audio=False,
        ),
        sample_interval_sec=0.2,
        max_frames=2,
        min_change_score=20.0,
    )

    assert captures[0].set_calls == []
    assert selected


def test_select_motion_keyframes_uses_compact_diff_frames(monkeypatch, tmp_path) -> None:
    frames = []
    for index in range(4):
        image = np.zeros((240, 320, 3), dtype=np.uint8)
        if index >= 2:
            image[:, :] = 255
        frames.append(image)
    resize_shapes = []
    original_resize = cv2.resize

    class FakeCapture:
        def __init__(self, path):
            self.index = 0

        def isOpened(self):
            return True

        def grab(self):
            self.index += 1
            return self.index <= len(frames)

        def read(self):
            if self.index >= len(frames):
                return False, None
            image = frames[self.index]
            self.index += 1
            return True, image

        def release(self):
            pass

    def record_resize(image, size, *args, **kwargs):
        resize_shapes.append(size)
        return original_resize(image, size, *args, **kwargs)

    monkeypatch.setattr(cv2, "VideoCapture", FakeCapture)
    monkeypatch.setattr(cv2, "resize", record_resize)

    selected = select_motion_keyframes(
        tmp_path / "fake.mp4",
        metadata=VideoMetadata(
            duration_sec=0.4,
            fps=10,
            resolution="320x240",
            frame_count=len(frames),
            has_audio=False,
        ),
        sample_interval_sec=0.1,
        max_frames=2,
        min_change_score=20.0,
    )

    assert selected
    assert resize_shapes
    assert all(width <= 96 and height <= 54 for width, height in resize_shapes)


def test_extract_selected_frames_scans_forward_without_random_seeks(monkeypatch, tmp_path) -> None:
    from accident_vlm.modules.frame_selection import extract_selected_frames

    frames = [np.full((8, 8, 3), index, dtype=np.uint8) for index in range(8)]
    captures = []
    written = []

    class FakeCapture:
        def __init__(self, path):
            self.index = 0
            self.set_calls = []
            captures.append(self)

        def isOpened(self):
            return True

        def set(self, prop, value):
            self.set_calls.append((prop, value))
            self.index = int(value)
            return True

        def grab(self):
            self.index += 1
            return self.index <= len(frames)

        def read(self):
            if self.index >= len(frames):
                return False, None
            image = frames[self.index]
            self.index += 1
            return True, image

        def release(self):
            pass

    monkeypatch.setattr(cv2, "VideoCapture", FakeCapture)
    monkeypatch.setattr(cv2, "imwrite", lambda path, image: written.append((path, int(image[0, 0, 0]))) or True)

    selected = [
        SelectedFrame(id="frame_000002", time="00:00.200", frame_index=2, purpose="a"),
        SelectedFrame(id="frame_000005", time="00:00.500", frame_index=5, purpose="b"),
    ]

    extracted = extract_selected_frames(tmp_path / "fake.mp4", selected, tmp_path / "frames")

    assert captures[0].set_calls == []
    assert [Path(path).name for path, _ in written] == ["frame_000002.jpg", "frame_000005.jpg"]
    assert [value for _, value in written] == [2, 5]
    assert [frame.path for frame in extracted] == [
        str(tmp_path / "frames" / "frame_000002.jpg"),
        str(tmp_path / "frames" / "frame_000005.jpg"),
    ]


def test_extract_selected_frames_skips_existing_outputs(monkeypatch, tmp_path) -> None:
    from accident_vlm.modules.frame_selection import extract_selected_frames

    output_dir = tmp_path / "frames"
    output_dir.mkdir()
    existing_path = output_dir / "frame_000002.jpg"
    existing_path.write_bytes(b"already extracted")
    frames = [np.full((8, 8, 3), index, dtype=np.uint8) for index in range(8)]
    written = []

    class FakeCapture:
        def __init__(self, path):
            self.index = 0

        def isOpened(self):
            return True

        def grab(self):
            self.index += 1
            return self.index <= len(frames)

        def read(self):
            if self.index >= len(frames):
                return False, None
            image = frames[self.index]
            self.index += 1
            return True, image

        def release(self):
            pass

    monkeypatch.setattr(cv2, "VideoCapture", FakeCapture)
    monkeypatch.setattr(cv2, "imwrite", lambda path, image: written.append(Path(path).name) or True)

    selected = [
        SelectedFrame(
            id="frame_000002",
            time="00:00.200",
            frame_index=2,
            purpose="already",
            path=str(existing_path),
        ),
        SelectedFrame(id="frame_000005", time="00:00.500", frame_index=5, purpose="new"),
    ]

    extracted = extract_selected_frames(tmp_path / "fake.mp4", selected, output_dir)

    assert written == ["frame_000005.jpg"]
    assert [frame.path for frame in extracted] == [
        str(existing_path),
        str(output_dir / "frame_000005.jpg"),
    ]


def test_extract_selected_frames_reuses_fresh_output_even_without_frame_path(monkeypatch, tmp_path) -> None:
    from accident_vlm.modules.frame_selection import extract_selected_frames

    video_path = tmp_path / "fake.mp4"
    video_path.write_bytes(b"video")
    output_dir = tmp_path / "frames"
    output_dir.mkdir()
    existing_path = output_dir / "frame_000002.jpg"
    existing_path.write_bytes(b"already extracted")
    captures = []

    class FakeCapture:
        def __init__(self, path):
            captures.append(path)

        def isOpened(self):
            return True

        def release(self):
            pass

    monkeypatch.setattr(cv2, "VideoCapture", FakeCapture)

    selected = [
        SelectedFrame(id="frame_000002", time="00:00.200", frame_index=2, purpose="a"),
    ]

    extracted = extract_selected_frames(video_path, selected, output_dir)

    assert captures == []
    assert extracted[0].path == str(existing_path)


def test_extract_selected_frames_primes_image_cache_for_new_outputs(monkeypatch, tmp_path) -> None:
    from accident_vlm.modules.frame_selection import extract_selected_frames

    frames = [np.full((8, 8, 3), index, dtype=np.uint8) for index in range(4)]
    cached = []

    class FakeCapture:
        def __init__(self, path):
            self.index = 0

        def isOpened(self):
            return True

        def grab(self):
            self.index += 1
            return self.index <= len(frames)

        def read(self):
            if self.index >= len(frames):
                return False, None
            image = frames[self.index]
            self.index += 1
            return True, image

        def release(self):
            pass

    monkeypatch.setattr(cv2, "VideoCapture", FakeCapture)
    monkeypatch.setattr(cv2, "imwrite", lambda path, image: True)
    monkeypatch.setattr(
        "accident_vlm.modules.frame_selection.cache_cv_image",
        lambda path, image: cached.append((Path(path).name, int(image[0, 0, 0]))),
    )

    selected = [SelectedFrame(id="frame_000002", time="00:00.200", frame_index=2, purpose="new")]

    extract_selected_frames(tmp_path / "fake.mp4", selected, tmp_path / "frames")

    assert cached == [("frame_000002.jpg", 2)]
