from pathlib import Path

from accident_vlm.modules.actor_tracking import Detection, UltralyticsTracker, detect_and_track_actors
from accident_vlm.schemas.preprocessing import SelectedFrame


class FakeTrackedBox:
    def __init__(self, track_id, cls_id, confidence, xyxy):
        self.id = [track_id]
        self.cls = [cls_id]
        self.conf = [confidence]
        self.xyxy = [xyxy]


class FakeTrackResult:
    names = {2: "car"}

    def __init__(self, boxes):
        self.boxes = boxes


class FakeYoloModel:
    def track(self, source, tracker, persist, verbose):
        return [FakeTrackResult([FakeTrackedBox(7, 2, 0.88, [10, 20, 60, 80])])]


def test_ultralytics_tracker_uses_tracker_config_and_preserves_track_id() -> None:
    tracker = UltralyticsTracker(
        "unused.pt",
        tracker_config="bytetrack.yaml",
        model=FakeYoloModel(),
    )

    detections = tracker.detect(Path("frame.jpg"))

    assert detections == [
        Detection(label="승용차", confidence=0.88, bbox=[10, 20, 60, 80], track_id="T7")
    ]


class FakeDetector:
    name = "fake"

    def __init__(self) -> None:
        self.index = 0

    def detect(self, image_path):
        self.index += 1
        return [
            Detection(
                label="승용차",
                confidence=0.9,
                bbox=[10 + self.index, 20, 60 + self.index, 80],
                track_id="T42",
            )
        ]


def test_detect_and_track_actors_prefers_detector_track_ids(monkeypatch) -> None:
    monkeypatch.setattr(
        "accident_vlm.modules.actor_tracking._read_frame_shape",
        lambda path: (100, 200),
    )

    tracks = detect_and_track_actors(
        [
            SelectedFrame(
                id="frame_000001",
                time="00:00.033",
                frame_index=1,
                purpose="regular_context",
                path="/tmp/frame1.jpg",
            ),
            SelectedFrame(
                id="frame_000002",
                time="00:00.066",
                frame_index=2,
                purpose="regular_context",
                path="/tmp/frame2.jpg",
            ),
        ],
        FakeDetector(),
    )

    assert len(tracks) == 1
    assert tracks[0]["track_id"] == "T42"
    assert tracks[0]["tracking_method"] == "detector_track_id"
    assert len(tracks[0]["positions"]) == 2
