import json
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
    assert config.enable_ocr is False
    assert config.max_motion_keyframes == 16
    assert config.motion_sample_interval_sec == 0.25
    assert config.min_motion_change_score == 6.0
    assert config.enable_segment_tracking is True
    assert config.segment_tracking_stride_frames == 2
    assert config.max_segment_tracking_frames == 180
    assert config.vlm_frame_budget == 20
    assert config.pre_event_window_sec == 6.0
    assert config.post_event_window_sec == 4.0


def test_config_from_options_maps_extended_pipeline_options(tmp_path: Path) -> None:
    config = config_from_options(
        AnalysisOptions(
            object_detector_backend="bytetrack",
            regular_frame_interval_sec=0.5,
            max_selected_frames=24,
            enable_ocr=True,
            enable_motion_keyframes=False,
            enable_segment_tracking=False,
            max_motion_keyframes=12,
            segment_tracking_stride_frames=2,
            max_segment_tracking_frames=60,
            vlm_frame_budget=24,
            motion_sample_interval_sec=0.25,
            min_motion_change_score=8.0,
            pre_event_window_sec=4.0,
            post_event_window_sec=2.0,
            lane_width_m=3.5,
        ),
        output_dir=tmp_path,
    )

    assert config.object_detector_backend == "bytetrack"
    assert config.enable_ocr is True
    assert config.enable_motion_keyframes is False
    assert config.enable_segment_tracking is False
    assert config.max_motion_keyframes == 12
    assert config.segment_tracking_stride_frames == 2
    assert config.max_segment_tracking_frames == 60
    assert config.vlm_frame_budget == 24
    assert config.motion_sample_interval_sec == 0.25
    assert config.min_motion_change_score == 8.0
    assert config.pre_event_window_sec == 4.0
    assert config.post_event_window_sec == 2.0
    assert config.lane_width_m == 3.5


def test_config_from_options_fast_mode_disables_expensive_preprocessing(tmp_path: Path) -> None:
    config = config_from_options(
        AnalysisOptions(speed_mode="fast"),
        output_dir=tmp_path,
    )

    assert config.output_dir == tmp_path
    assert config.ocr_backend == "none"
    assert config.object_detector_backend == "none"
    assert config.enable_ocr is False
    assert config.enable_actor_tracking is False
    assert config.enable_segment_tracking is False
    assert config.enable_motion_keyframes is False
    assert config.enable_event_scan is False
    assert config.enable_event_detection is False
    assert config.enable_road_geometry is False
    assert config.enable_traffic_control is False
    assert config.enable_scene_analysis is False
    assert config.enable_speed_distance is False
    assert config.enable_input_quality is False
    assert config.enable_contact_sheet is False
    assert config.max_selected_frames == 8
    assert config.vlm_frame_budget == 8


def test_runner_import_defers_pipeline_and_vlm_modules() -> None:
    import subprocess
    import sys

    script = """
import sys
import accident_vlm.server.runner
for name in (
    "accident_vlm.pipeline",
    "accident_vlm.modules.vlm_composer",
):
    print(name, name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "True" not in result.stdout


def test_runner_fast_analysis_defers_cli_typer_pipeline_and_vlm_modules(tmp_path: Path) -> None:
    import subprocess
    import sys

    video_path = tmp_path / "sample.mp4"
    output_root = tmp_path / "jobs"
    script = f"""
import cv2
import json
import numpy as np
import sys
from pathlib import Path

from accident_vlm.server.job_store import JobStore
from accident_vlm.server.runner import run_analysis_job
from accident_vlm.server.schemas import AnalysisOptions

video_path = Path({str(video_path)!r})
writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 5, (32, 24))
for _ in range(3):
    writer.write(np.zeros((24, 32, 3), dtype=np.uint8))
writer.release()

store = JobStore(Path({str(output_root)!r}))
record = store.create(AnalysisOptions().mode, video_path)
run_analysis_job(store, record.job_id, video_path, AnalysisOptions(speed_mode="fast"))
record = store.get(record.job_id)
result = json.loads(Path(record.pre_vlm_output_path).read_text(encoding="utf-8"))
print(json.dumps({{
    "typer_loaded": "typer" in sys.modules,
    "cli_loaded": "accident_vlm.cli" in sys.modules,
    "pipeline_loaded": "accident_vlm.pipeline" in sys.modules,
    "vlm_loaded": "accident_vlm.modules.vlm_composer" in sys.modules,
    "selected_frames": len(result["selected_frames"]),
}}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(result.stdout.strip().splitlines()[-1]) == {
        "typer_loaded": False,
        "cli_loaded": False,
        "pipeline_loaded": False,
        "vlm_loaded": False,
        "selected_frames": 1,
    }
