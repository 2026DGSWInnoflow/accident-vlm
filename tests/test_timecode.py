import pytest

from accident_vlm.utils.timecode import frame_to_timecode, parse_timecode, seconds_to_timecode


def test_seconds_to_timecode_formats_minutes_seconds_and_millis() -> None:
    assert seconds_to_timecode(65.432) == "01:05.432"


def test_frame_to_timecode_converts_frame_index_using_fps() -> None:
    assert frame_to_timecode(frame_index=90, fps=30) == "00:03.000"


def test_parse_timecode_converts_minutes_seconds_and_millis() -> None:
    assert parse_timecode("01:05.432") == 65.432


def test_seconds_to_timecode_rejects_negative_seconds() -> None:
    with pytest.raises(ValueError):
        seconds_to_timecode(-1)


def test_frame_to_timecode_rejects_negative_frame_index() -> None:
    with pytest.raises(ValueError):
        frame_to_timecode(frame_index=-1, fps=30)


def test_frame_to_timecode_rejects_non_positive_fps() -> None:
    with pytest.raises(ValueError):
        frame_to_timecode(frame_index=1, fps=0)
