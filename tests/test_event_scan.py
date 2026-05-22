import cv2
import numpy as np

from accident_vlm.modules.event_scan import (
    build_frame_selection_contact_sheet,
    scan_video_event_candidates,
    select_precision_event_frames,
)
from accident_vlm.schemas.preprocessing import SelectedFrame, VideoMetadata


def _write_flash_video(path, fps: int = 30, frame_count: int = 60) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (96, 64),
    )
    for index in range(frame_count):
        image = np.zeros((64, 96, 3), dtype=np.uint8)
        if 18 <= index <= 23:
            image[18:46, 32:64] = 255
        writer.write(image)
    writer.release()


def test_scan_video_event_candidates_finds_short_flash_collision_candidate(tmp_path):
    video_path = tmp_path / "flash.mp4"
    _write_flash_video(video_path)
    metadata = VideoMetadata(
        duration_sec=2.0,
        fps=30,
        resolution="96x64",
        frame_count=60,
        has_audio=False,
    )

    candidates = scan_video_event_candidates(
        video_path,
        metadata,
        sample_fps=5.0,
        top_k=3,
        min_score=1.0,
    )

    assert candidates
    assert candidates[0]["source"] == "high_fps_event_scan"
    assert candidates[0]["event_type"] == "event_scan_peak"
    assert candidates[0]["event_score"] > 0
    assert candidates[0]["window"]["start"] == "00:00.000"
    assert "optical_flow_peak" in candidates[0]["supporting_signals"]
    assert "camera_shake_peak" in candidates[0]["supporting_signals"]
    assert any(abs(candidate["frame_index"] - 18) <= 6 for candidate in candidates)


def test_select_precision_event_frames_keeps_impact_frames_and_rejections():
    metadata = VideoMetadata(
        duration_sec=12.0,
        fps=30,
        resolution="1920x1080",
        frame_count=360,
        has_audio=False,
    )
    candidates = [
        {
            "time": "00:06.000",
            "frame_index": 180,
            "event_score": 90,
            "event_type": "event_scan_peak",
            "source": "high_fps_event_scan",
        },
        {
            "time": "00:09.000",
            "frame_index": 270,
            "event_score": 55,
            "event_type": "event_scan_peak",
            "source": "high_fps_event_scan",
        },
    ]

    frames, rejected = select_precision_event_frames(
        candidates,
        metadata,
        max_frames=20,
        pre_event_window_sec=6.0,
        post_event_window_sec=4.0,
        precision_fps=15.0,
        min_impact_frames=5,
    )

    assert len(frames) == 20
    assert sum(1 for frame in frames if "impact_candidate" in frame.purpose) >= 5
    assert any("event_scan" in frame.purpose for frame in frames)
    assert rejected
    assert all("reason" in item for item in rejected)


def test_build_frame_selection_contact_sheet_writes_review_image(tmp_path):
    image_paths = []
    for index in range(3):
        image = np.full((40, 60, 3), index * 80, dtype=np.uint8)
        path = tmp_path / f"frame_{index}.jpg"
        cv2.imwrite(str(path), image)
        image_paths.append(path)
    frames = [
        SelectedFrame(
            id=f"frame_{index:06d}",
            time=f"00:0{index}.000",
            frame_index=index,
            path=str(path),
            purpose="impact_candidate",
        )
        for index, path in enumerate(image_paths)
    ]

    record = build_frame_selection_contact_sheet(
        frames,
        tmp_path / "contact_sheet.jpg",
        title="phase1",
    )

    assert record["purpose"] == "frame_selection_contact_sheet"
    assert record["path"].endswith("contact_sheet.jpg")
    assert cv2.imread(record["path"]) is not None
