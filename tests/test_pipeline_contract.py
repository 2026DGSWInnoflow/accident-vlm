from accident_vlm.modules.evidence_builder import build_evidence_package
from accident_vlm.pipeline import build_pre_vlm_context
from accident_vlm.pipeline import analyze_video_pre_vlm
from accident_vlm.config import PipelineConfig
from accident_vlm.schemas.preprocessing import PipelineContext, SelectedFrame, VideoMetadata


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
        ),
    )

    assert captured["frame_indices"] == [0, 15, 30]
    assert captured["purposes"] == ["regular_context", "motion_keyframe", "regular_context"]
