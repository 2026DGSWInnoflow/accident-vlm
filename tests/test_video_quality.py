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


def test_analyze_input_quality_scans_forward_without_random_seeks(monkeypatch, tmp_path) -> None:
    frames = []
    for index in range(8):
        image = np.full((48, 64, 3), index * 20, dtype=np.uint8)
        if index >= 5:
            cv2.rectangle(image, (20, 12), (44, 36), (255, 255, 255), -1)
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

    quality = analyze_input_quality(
        tmp_path / "fake.mp4",
        [
            SelectedFrame(id="frame_000002", time="00:00.200", frame_index=2, purpose="a"),
            SelectedFrame(id="frame_000005", time="00:00.500", frame_index=5, purpose="b"),
        ],
    )

    assert captures[0].set_calls == []
    assert [item["frame_id"] for item in quality.timeline] == [
        "frame_000002",
        "frame_000005",
    ]


def test_analyze_input_quality_downscales_optical_flow_inputs(monkeypatch, tmp_path) -> None:
    frames = []
    for index in range(3):
        image = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.rectangle(image, (40 + index * 20, 80), (120 + index * 20, 160), (255, 255, 255), -1)
        frames.append(image)
    flow_shapes = []

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

    def fake_flow(previous, current, *args):
        flow_shapes.append(previous.shape)
        return np.ones((*previous.shape, 2), dtype=np.float32)

    monkeypatch.setattr(cv2, "VideoCapture", FakeCapture)
    monkeypatch.setattr(cv2, "calcOpticalFlowFarneback", fake_flow)

    analyze_input_quality(
        tmp_path / "fake.mp4",
        [
            SelectedFrame(id=f"frame_{index:06d}", time=f"00:00.{index}00", frame_index=index, purpose="a")
            for index in range(3)
        ],
    )

    assert flow_shapes
    assert all(height <= 96 and width <= 96 for height, width in flow_shapes)
