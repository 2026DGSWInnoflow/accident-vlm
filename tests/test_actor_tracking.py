from pathlib import Path

from accident_vlm.modules.actor_tracking import (
    ACCIDENT_ACTOR_TAXONOMY,
    Detection,
    NoObjectDetector,
    UltralyticsTracker,
    compare_tracker_outputs,
    detect_and_track_actors,
    detect_and_track_segments,
    detector_profile,
)
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


def test_detect_and_track_actors_uses_batch_detector_when_available(monkeypatch) -> None:
    class BatchDetector:
        name = "batch"

        def __init__(self):
            self.batch_calls = []

        def detect(self, image_path):
            raise AssertionError("batch detector should not fall back to per-frame detect")

        def detect_many(self, image_paths):
            self.batch_calls.append([str(path) for path in image_paths])
            return [
                [Detection(label="승용차", confidence=0.9, bbox=[10, 20, 60, 80], track_id="T1")],
                [Detection(label="승용차", confidence=0.8, bbox=[20, 20, 70, 80], track_id="T1")],
            ]

    detector = BatchDetector()
    monkeypatch.setattr("accident_vlm.modules.actor_tracking._read_frame_shape", lambda path: (100, 200))

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
        detector,
    )

    assert detector.batch_calls == [["/tmp/frame1.jpg", "/tmp/frame2.jpg"]]
    assert len(tracks) == 1
    assert len(tracks[0]["positions"]) == 2


def test_detect_and_track_actors_skips_frame_reads_for_disabled_detector(monkeypatch) -> None:
    monkeypatch.setattr(
        "accident_vlm.modules.actor_tracking._read_frame_shape",
        lambda path: (_ for _ in ()).throw(
            AssertionError("disabled detector should not read frame images")
        ),
    )

    tracks = detect_and_track_actors(
        [
            SelectedFrame(
                id="frame_000001",
                time="00:00.033",
                frame_index=1,
                purpose="regular_context",
                path="/tmp/frame1.jpg",
            )
        ],
        NoObjectDetector(),
    )

    assert tracks == []


def test_detect_and_track_segments_extracts_dense_segment_frames(monkeypatch, tmp_path) -> None:
    calls = {}

    def fake_extract(video_path, selected_frames, output_dir):
        calls["indices"] = [frame.frame_index for frame in selected_frames]
        return [
            frame.model_copy(update={"path": str(tmp_path / f"{frame.id}.jpg")})
            for frame in selected_frames
        ]

    monkeypatch.setattr("accident_vlm.modules.actor_tracking.extract_selected_frames", fake_extract)
    monkeypatch.setattr(
        "accident_vlm.modules.actor_tracking._read_frame_shape",
        lambda path: (100, 200),
    )

    tracks = detect_and_track_segments(
        video_path=Path("/tmp/video.mp4"),
        selected_segments=[
            {
                "id": "seg_event_001",
                "start": "00:01.000",
                "end": "00:01.300",
            }
        ],
        fps=10,
        detector=FakeDetector(),
        output_dir=tmp_path / "segments",
        stride_frames=1,
        max_frames_per_segment=4,
    )

    assert calls["indices"] == [10, 11, 12, 13]
    assert tracks[0]["source_stage"] == "segment_tracking"
    assert len(tracks[0]["positions"]) == 4


def test_detect_and_track_segments_skips_extraction_for_disabled_detector(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "accident_vlm.modules.actor_tracking.extract_selected_frames",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("disabled detector should not extract segment frames")
        ),
    )

    tracks = detect_and_track_segments(
        video_path=Path("/tmp/video.mp4"),
        selected_segments=[{"id": "seg_event_001", "start": "00:01.000", "end": "00:01.300"}],
        fps=10,
        detector=NoObjectDetector(),
        output_dir=tmp_path / "segments",
    )

    assert tracks == []


def test_detect_and_track_segments_caps_total_extracted_frames(monkeypatch, tmp_path) -> None:
    calls = {}

    def fake_extract(video_path, selected_frames, output_dir):
        calls["indices"] = [frame.frame_index for frame in selected_frames]
        return [
            frame.model_copy(update={"path": str(tmp_path / f"{frame.id}.jpg")})
            for frame in selected_frames
        ]

    monkeypatch.setattr("accident_vlm.modules.actor_tracking.extract_selected_frames", fake_extract)
    monkeypatch.setattr(
        "accident_vlm.modules.actor_tracking._read_frame_shape",
        lambda path: (100, 200),
    )

    detect_and_track_segments(
        video_path=Path("/tmp/video.mp4"),
        selected_segments=[
            {"id": "seg_event_001", "start": "00:00.000", "end": "00:10.000"},
            {"id": "seg_event_002", "start": "00:10.000", "end": "00:20.000"},
        ],
        fps=10,
        detector=FakeDetector(),
        output_dir=tmp_path / "segments",
        stride_frames=1,
        max_frames_per_segment=4,
        max_total_frames=6,
    )

    assert calls["indices"] == [0, 1, 2, 3, 100, 101]


def test_accident_actor_taxonomy_and_detector_profile_are_explicit() -> None:
    assert "kickboard" in ACCIDENT_ACTOR_TAXONOMY
    assert ACCIDENT_ACTOR_TAXONOMY["traffic_light"]["korean_label"] == "신호등"
    profile = detector_profile("yolov8x.pt", "bytetrack")
    assert profile["base_model_family"] == "coco_pretrained"
    assert profile["custom_traffic_model"] is False
    custom_profile = detector_profile("accident-traffic-custom.pt", "botsort")
    assert custom_profile["custom_traffic_model"] is True


def test_detect_and_track_actors_keeps_low_confidence_actor_with_uncertainty(monkeypatch) -> None:
    class LowConfidenceDetector:
        name = "low"

        def detect(self, image_path):
            return [Detection(label="보행자", confidence=0.22, bbox=[10, 10, 20, 30], track_id="P1")]

    monkeypatch.setattr("accident_vlm.modules.actor_tracking._read_frame_shape", lambda path: (100, 200))

    tracks = detect_and_track_actors(
        [
            SelectedFrame(
                id="frame_000003",
                time="00:00.100",
                frame_index=3,
                purpose="impact_candidate",
                path="/tmp/frame3.jpg",
            )
        ],
        LowConfidenceDetector(),
    )

    assert tracks[0]["confidence"] == "low"
    assert tracks[0]["uncertainty_reasons"] == ["low_detector_confidence"]


def test_compare_tracker_outputs_reports_overlap_and_disagreements() -> None:
    comparison = compare_tracker_outputs(
        bytetrack_tracks=[{"track_id": "T1", "positions": [{"frame_id": "f1"}]}],
        botsort_tracks=[
            {"track_id": "T1", "positions": [{"frame_id": "f1"}, {"frame_id": "f2"}]},
            {"track_id": "T2", "positions": [{"frame_id": "f3"}]},
        ],
    )

    assert comparison["bytetrack_count"] == 1
    assert comparison["botsort_count"] == 2
    assert comparison["shared_track_ids"] == ["T1"]
    assert comparison["disagreement_count"] == 1
