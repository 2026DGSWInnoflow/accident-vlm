from accident_vlm.modules.evidence_builder import build_evidence_package
from accident_vlm.pipeline import build_pre_vlm_context
from accident_vlm.schemas.preprocessing import PipelineContext, VideoMetadata


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
    assert [frame.frame_index for frame in context.selected_frames] == [0, 30]
    assert all(frame.frame_index < context.video_metadata.frame_count for frame in context.selected_frames)
    assert len(context.selected_frames) == 2
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
        road_geometry={"lanes": [{"id": "lane_1", "direction": "forward"}]},
    )

    evidence_package = build_evidence_package(context)

    context.tracks[0]["positions"][0]["x"] = 99
    context.road_geometry["lanes"][0]["direction"] = "changed"

    precomputed_facts = evidence_package["precomputed_facts"]
    assert precomputed_facts["tracks"][0]["positions"][0]["x"] == 10
    assert precomputed_facts["road_geometry"]["lanes"][0]["direction"] == "forward"
