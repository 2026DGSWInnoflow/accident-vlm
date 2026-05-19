import pytest

from accident_vlm.modules.frame_selection import select_regular_frames
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
