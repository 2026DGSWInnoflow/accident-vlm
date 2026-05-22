import cv2
import numpy as np

from accident_vlm.modules.video_quality import analyze_input_quality
from accident_vlm.schemas.preprocessing import SelectedFrame


def test_analyze_input_quality_reports_camera_shake_peak_score_and_evidence(tmp_path) -> None:
    video_path = tmp_path / "shake.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10,
        (80, 60),
    )
    for index in range(4):
        image = np.zeros((60, 80, 3), dtype=np.uint8)
        offset = 0 if index < 2 else 25
        cv2.rectangle(image, (10 + offset, 20), (30 + offset, 40), (255, 255, 255), -1)
        writer.write(image)
    writer.release()

    quality = analyze_input_quality(
        video_path,
        [
            SelectedFrame(
                id=f"frame_{index:06d}",
                time=f"00:0{index}.000",
                frame_index=index,
                purpose="regular_context",
            )
            for index in range(4)
        ],
    )

    assert quality.camera_shake_score["value"] > 0
    assert quality.camera_shake_score["time"] == "00:02.000"
    assert quality.camera_shake_score["evidence"] == ["frame_000001", "frame_000002"]


def test_analyze_input_quality_reports_timeline_segment_and_visibility_flags(tmp_path) -> None:
    video_path = tmp_path / "quality.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10,
        (80, 60),
    )
    for index in range(6):
        image = np.full((60, 80, 3), 35, dtype=np.uint8)
        if index >= 3:
            image[:, :] = 255
        writer.write(image)
    writer.release()

    quality = analyze_input_quality(
        video_path,
        [
            SelectedFrame(
                id=f"frame_{index:06d}",
                time=f"00:0{index}.000",
                frame_index=index,
                purpose="regular_context",
            )
            for index in range(6)
        ],
        event_windows=[
            {
                "id": "event_scan_000003",
                "time": "00:03.000",
                "window": {"start": "00:02.000", "end": "00:05.000"},
            }
        ],
    )

    assert len(quality.timeline) == 6
    assert {"blur_score", "brightness_score", "noise_score", "glare_ratio"} <= set(
        quality.timeline[0]
    )
    assert quality.camera_shake_score["ego_motion_compensated_value"] >= 0
    assert quality.segment_quality[0]["event_id"] == "event_scan_000003"
    assert quality.visibility_conditions["overexposure_candidate"] is True
