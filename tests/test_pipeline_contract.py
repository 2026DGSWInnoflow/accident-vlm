from accident_vlm.modules.evidence_builder import build_evidence_package
from accident_vlm.pipeline import build_pre_vlm_context
from accident_vlm.pipeline import analyze_video_pre_vlm
from accident_vlm.config import PipelineConfig
from accident_vlm.schemas.preprocessing import PipelineContext, SelectedFrame, VideoMetadata
from accident_vlm.modules.track_consolidation import consolidate_tracks


def test_build_pre_vlm_context_contains_regular_frames() -> None:
    context = build_pre_vlm_context(
        video_path="sample.mp4",
        metadata=VideoMetadata(
            duration_sec=2.0,
            fps=30,
            resolution="1920x1080",
            frame_count=60,
            has_audio=False,
        ),
    )

    assert context.video_path == "sample.mp4"
    assert context.video_metadata is not None
    assert [frame.frame_index for frame in context.selected_frames] == [0, 15, 30, 45]
    assert all(frame.frame_index < context.video_metadata.frame_count for frame in context.selected_frames)
    assert len(context.selected_frames) == 4
    assert context.evidence_package["precomputed_facts"]["metadata"]["fps"] == 30


def test_build_pre_vlm_context_omits_frames_for_zero_frame_metadata() -> None:
    context = build_pre_vlm_context(
        video_path="empty.mp4",
        metadata=VideoMetadata(
            duration_sec=0.0,
            fps=30,
            resolution="1920x1080",
            frame_count=0,
            has_audio=False,
        ),
    )

    assert context.selected_frames == []
    assert context.evidence_package["frames"] == []


def test_build_evidence_package_is_snapshot_of_mutable_context_fields() -> None:
    context = PipelineContext(
        video_path="sample.mp4",
        video_metadata=VideoMetadata(
            duration_sec=2.0,
            fps=30,
            resolution="1920x1080",
            frame_count=60,
            has_audio=False,
        ),
        tracks=[{"id": "vehicle_1", "positions": [{"frame_index": 0, "x": 10}]}],
        tracker_comparison={"disagreement_count": 1},
        preprocessing_uncertainties=["low confidence actor retained: T1"],
        ocr_observations=[
            {
                "frame_id": "frame_000001",
                "roi_name": "bottom_band",
                "image_path": "/tmp/ocr_roi.jpg",
            }
        ],
        road_geometry={
            "lanes": [{"id": "lane_1", "direction": "forward"}],
            "lane_segmentation": {
                "overlays": [
                    {
                        "id": "lane_overlay_1",
                        "path": "/tmp/lane_overlay.jpg",
                        "mask_path": "/tmp/lane_mask.jpg",
                    }
                ]
            },
        },
        traffic_control={
            "signal": {
                "crops": [{"id": "signal_crop_1", "path": "/tmp/signal_crop.jpg"}]
            }
        },
        overlays=[{"id": "tracking_overlay_1", "path": "/tmp/tracking.jpg"}],
        crops=[{"id": "actor_crop_1", "path": "/tmp/actor.jpg"}],
        selected_segments=[{"id": "seg_event_001", "start": "00:00.000", "end": "00:02.000"}],
    )

    evidence_package = build_evidence_package(context)

    context.tracks[0]["positions"][0]["x"] = 99
    context.road_geometry["lanes"][0]["direction"] = "changed"
    context.selected_segments[0]["end"] = "00:09.000"

    precomputed_facts = evidence_package["precomputed_facts"]
    assert precomputed_facts["tracks"][0]["positions"][0]["x"] == 10
    assert precomputed_facts["tracker_comparison"]["disagreement_count"] == 1
    assert precomputed_facts["preprocessing_uncertainties"] == ["low confidence actor retained: T1"]
    assert precomputed_facts["road_geometry"]["lanes"][0]["direction"] == "forward"
    assert evidence_package["selected_segments"][0]["end"] == "00:02.000"
    assert {image["path"] for image in evidence_package["evidence_images"]} == {
        "/tmp/tracking.jpg",
        "/tmp/actor.jpg",
        "/tmp/ocr_roi.jpg",
        "/tmp/lane_overlay.jpg",
        "/tmp/lane_mask.jpg",
        "/tmp/signal_crop.jpg",
    }
    assert evidence_package["evidence_images"][0]["path"] == "/tmp/signal_crop.jpg"
    assert "importance_score" in evidence_package["evidence_images"][0]
    assert evidence_package["vlm_storyboard"]
    assert evidence_package["vlm_storyboard"][0]["slot"] == 1
    assert all("phase" in item for item in evidence_package["vlm_storyboard"])


def test_analyze_video_pre_vlm_merges_motion_keyframes_before_extraction(
    tmp_path, monkeypatch
) -> None:
    captured = {}

    monkeypatch.setattr(
        "accident_vlm.pipeline.probe_video",
        lambda video_path: VideoMetadata(
            duration_sec=1.0,
            fps=30,
            resolution="640x480",
            frame_count=31,
            has_audio=False,
        ),
    )
    monkeypatch.setattr(
        "accident_vlm.pipeline.select_motion_keyframes",
        lambda *args, **kwargs: [
            SelectedFrame(
                id="frame_000015",
                time="00:00.500",
                frame_index=15,
                purpose="motion_keyframe",
            )
        ],
    )

    def fake_extract_selected_frames(video_path, selected_frames, output_dir):
        captured["frame_indices"] = [frame.frame_index for frame in selected_frames]
        captured["purposes"] = [frame.purpose for frame in selected_frames]
        return [frame.model_copy(update={"path": str(tmp_path / f"{frame.id}.jpg")}) for frame in selected_frames]

    monkeypatch.setattr("accident_vlm.pipeline.extract_selected_frames", fake_extract_selected_frames)
    monkeypatch.setattr("accident_vlm.pipeline.analyze_input_quality", lambda *args, **kwargs: None)

    analyze_video_pre_vlm(
        tmp_path / "sample.mp4",
        PipelineConfig(
            output_dir=tmp_path / "outputs",
            regular_frame_interval_sec=1.0,
            enable_motion_keyframes=True,
            enable_ocr=False,
            enable_actor_tracking=False,
            enable_road_geometry=False,
            enable_speed_distance=False,
            enable_traffic_control=False,
            enable_scene_analysis=False,
            enable_event_detection=False,
            enable_event_scan=False,
        ),
    )

    assert captured["frame_indices"] == [0, 15, 30]
    assert captured["purposes"] == ["regular_context", "motion_keyframe", "regular_context"]


def test_analyze_video_pre_vlm_connects_event_scan_candidates(tmp_path, monkeypatch) -> None:
    captured = {"extract_calls": []}

    monkeypatch.setattr(
        "accident_vlm.pipeline.probe_video",
        lambda video_path: VideoMetadata(
            duration_sec=2.0,
            fps=30,
            resolution="640x480",
            frame_count=60,
            has_audio=False,
        ),
    )
    monkeypatch.setattr(
        "accident_vlm.pipeline.scan_video_event_candidates",
        lambda *args, **kwargs: [
            {
                "time": "00:01.000",
                "frame_index": 30,
                "event_type": "event_scan_peak",
                "event_score": 88,
                "source": "high_fps_event_scan",
                "evidence": ["frame_000024", "frame_000030"],
            }
        ],
    )

    def fake_extract_selected_frames(video_path, selected_frames, output_dir):
        captured["extract_calls"].append([frame.purpose for frame in selected_frames])
        return [frame.model_copy(update={"path": str(tmp_path / f"{frame.id}.jpg")}) for frame in selected_frames]

    monkeypatch.setattr("accident_vlm.pipeline.extract_selected_frames", fake_extract_selected_frames)
    monkeypatch.setattr("accident_vlm.pipeline.select_motion_keyframes", lambda *args, **kwargs: [])
    monkeypatch.setattr("accident_vlm.pipeline.analyze_input_quality", lambda *args, **kwargs: None)
    monkeypatch.setattr("accident_vlm.pipeline.create_ocr_backend", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("accident_vlm.pipeline.extract_ocr_observations", lambda *args, **kwargs: [])
    monkeypatch.setattr("accident_vlm.pipeline.summarize_ocr_observations", lambda *_args: {})
    monkeypatch.setattr("accident_vlm.pipeline.create_object_detector", lambda *args, **kwargs: None)
    monkeypatch.setattr("accident_vlm.pipeline.detect_and_track_actors", lambda *args, **kwargs: [])
    monkeypatch.setattr("accident_vlm.pipeline.build_visual_evidence", lambda *args, **kwargs: ([], []))
    monkeypatch.setattr("accident_vlm.pipeline.analyze_road_geometry", lambda *args, **kwargs: {})
    monkeypatch.setattr("accident_vlm.pipeline.estimate_speed_and_distance", lambda *args, **kwargs: {})
    monkeypatch.setattr("accident_vlm.pipeline.analyze_traffic_control", lambda *args, **kwargs: {})
    monkeypatch.setattr("accident_vlm.pipeline.classify_scene_candidates", lambda *args, **kwargs: [])
    monkeypatch.setattr("accident_vlm.pipeline.detect_event_candidates", lambda *args, **kwargs: [])
    monkeypatch.setattr("accident_vlm.pipeline.detect_and_track_segments", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        "accident_vlm.pipeline.build_frame_selection_contact_sheet",
            lambda frames, output_path, title: {
                "id": "contact_sheet_frame_selection",
                "path": str(output_path),
                "purpose": "frame_selection_contact_sheet",
                "status": "created",
            },
        )

    context = analyze_video_pre_vlm(
        tmp_path / "sample.mp4",
        PipelineConfig(
            output_dir=tmp_path / "outputs",
            enable_event_scan=True,
            event_scan_sample_fps=5.0,
            event_scan_top_k=3,
            event_scan_min_score=1.0,
            precision_event_fps=15.0,
            enable_motion_keyframes=True,
            enable_ocr=False,
            enable_actor_tracking=False,
            enable_segment_tracking=False,
            enable_road_geometry=False,
            enable_speed_distance=False,
            enable_traffic_control=False,
            enable_scene_analysis=False,
            enable_event_detection=True,
            vlm_frame_budget=8,
        ),
    )

    assert context.event_scan_candidates
    assert context.rejected_frame_candidates
    assert context.contact_sheets
    assert any(
        any("event_scan" in purpose for purpose in call)
        for call in captured["extract_calls"]
    )
    assert context.evidence_package["precomputed_facts"]["event_scan_candidates"]
    assert context.evidence_package["precomputed_facts"]["rejected_frame_candidates"]


def test_consolidate_tracks_adds_track_quality_without_misreading_segment_ids() -> None:
    tracks = consolidate_tracks(
        [
            {
                "track_id": "T1",
                "type": "승용차",
                "positions": [
                    {
                        "frame_id": "seg_event_001_frame_000030",
                        "time": "00:01.000",
                        "bbox": [0, 0, 20, 20],
                    },
                    {
                        "frame_id": "seg_event_001_frame_000036",
                        "time": "00:01.200",
                        "bbox": [5, 0, 25, 20],
                    },
                    {
                        "frame_id": "seg_event_001_frame_000060",
                        "time": "00:02.000",
                        "bbox": [20, 0, 40, 20],
                    },
                ],
            }
        ]
    )

    quality = tracks[0]["track_quality"]
    assert quality["position_count"] == 3
    assert quality["frame_span"] == [30, 60]
    assert quality["max_frame_gap"] == 24
    assert quality["fragmentation_score"] > 0
