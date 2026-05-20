from pathlib import Path

from accident_vlm.server.runner import config_from_options
from accident_vlm.server.schemas import AnalysisOptions


def test_analysis_options_defaults_to_quality_pipeline(tmp_path: Path) -> None:
    config = config_from_options(AnalysisOptions(), output_dir=tmp_path)

    assert config.object_detector_backend == "bytetrack"
    assert config.object_detector_model == "yolov8x.pt"
    assert config.regular_frame_interval_sec == 0.5
    assert config.max_selected_frames == 32
    assert config.enable_motion_keyframes is True
    assert config.max_motion_keyframes == 16
    assert config.motion_sample_interval_sec == 0.25
    assert config.min_motion_change_score == 6.0
    assert config.enable_segment_tracking is True
    assert config.segment_tracking_stride_frames == 2
    assert config.max_segment_tracking_frames == 180
    assert config.pre_event_window_sec == 6.0
    assert config.post_event_window_sec == 4.0


def test_config_from_options_maps_extended_pipeline_options(tmp_path: Path) -> None:
    config = config_from_options(
        AnalysisOptions(
            object_detector_backend="bytetrack",
            regular_frame_interval_sec=0.5,
            max_selected_frames=24,
            enable_motion_keyframes=False,
            enable_segment_tracking=False,
            max_motion_keyframes=12,
            segment_tracking_stride_frames=2,
            max_segment_tracking_frames=60,
            motion_sample_interval_sec=0.25,
            min_motion_change_score=8.0,
            pre_event_window_sec=4.0,
            post_event_window_sec=2.0,
            lane_width_m=3.5,
        ),
        output_dir=tmp_path,
    )

    assert config.object_detector_backend == "bytetrack"
    assert config.enable_motion_keyframes is False
    assert config.enable_segment_tracking is False
    assert config.max_motion_keyframes == 12
    assert config.segment_tracking_stride_frames == 2
    assert config.max_segment_tracking_frames == 60
    assert config.motion_sample_interval_sec == 0.25
    assert config.min_motion_change_score == 8.0
    assert config.pre_event_window_sec == 4.0
    assert config.post_event_window_sec == 2.0
    assert config.lane_width_m == 3.5
