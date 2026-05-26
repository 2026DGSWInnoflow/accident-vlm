from __future__ import annotations

import json
import traceback
from pathlib import Path

from accident_vlm.config import PipelineConfig
from accident_vlm.server.job_store import JobStore
from accident_vlm.server.schemas import AnalysisMode, AnalysisOptions, AnalysisSpeedMode


def analyze_video_pre_vlm(*args, **kwargs):
    config = kwargs.get("config") if "config" in kwargs else args[1] if len(args) > 1 else None
    if config is not None:
        from accident_vlm.fast_prevlm import analyze_video_pre_vlm_fast, is_lightweight_fast_config

        if is_lightweight_fast_config(config):
            video_path = kwargs.get("video_path") if "video_path" in kwargs else args[0]
            return analyze_video_pre_vlm_fast(video_path=video_path, config=config)
    from accident_vlm.pipeline import analyze_video_pre_vlm as implementation

    return implementation(*args, **kwargs)


def compose_final_facts(*args, **kwargs):
    from accident_vlm.modules.vlm_composer import compose_final_facts as implementation

    return implementation(*args, **kwargs)


def get_qwen_backend(*args, **kwargs):
    from accident_vlm.modules.vlm_composer import get_qwen_backend as implementation

    return implementation(*args, **kwargs)


def write_final_facts(*args, **kwargs):
    from accident_vlm.modules.vlm_composer import write_final_facts as implementation

    return implementation(*args, **kwargs)


def config_from_options(options: AnalysisOptions, output_dir: Path) -> PipelineConfig:
    if options.speed_mode == AnalysisSpeedMode.FAST:
        return PipelineConfig(
            output_dir=output_dir,
            regular_frame_interval_sec=1.0,
            max_selected_frames=8,
            enable_ocr=False,
            enable_motion_keyframes=False,
            enable_scene_analysis=False,
            enable_actor_tracking=False,
            enable_segment_tracking=False,
            enable_road_geometry=False,
            enable_speed_distance=False,
            enable_traffic_control=False,
            enable_event_detection=False,
            enable_event_scan=False,
            enable_input_quality=False,
            enable_contact_sheet=False,
            ocr_backend="none",
            object_detector_backend="none",
            object_detector_model=options.object_detector_model,
            qwen_model_id=options.qwen_model_id,
            device=options.device,
            enable_vlm=options.mode == AnalysisMode.FULL,
            vlm_frame_budget=8,
        )
    return PipelineConfig(
        output_dir=output_dir,
        regular_frame_interval_sec=options.regular_frame_interval_sec,
        max_selected_frames=options.max_selected_frames,
        enable_motion_keyframes=options.enable_motion_keyframes,
        enable_segment_tracking=options.enable_segment_tracking,
        max_motion_keyframes=options.max_motion_keyframes,
        motion_sample_interval_sec=options.motion_sample_interval_sec,
        min_motion_change_score=options.min_motion_change_score,
        pre_event_window_sec=options.pre_event_window_sec,
        post_event_window_sec=options.post_event_window_sec,
        segment_tracking_stride_frames=options.segment_tracking_stride_frames,
        max_segment_tracking_frames=options.max_segment_tracking_frames,
        vlm_frame_budget=options.vlm_frame_budget,
        max_event_candidates=options.max_event_candidates,
        enable_ocr=options.enable_ocr,
        lane_width_m=options.lane_width_m,
        ocr_backend=options.ocr_backend,
        object_detector_backend=options.object_detector_backend,
        object_detector_model=options.object_detector_model,
        qwen_model_id=options.qwen_model_id,
        device=options.device,
        enable_vlm=options.mode == AnalysisMode.FULL,
    )


def run_analysis_job(
    job_store: JobStore,
    job_id: str,
    video_path: Path,
    options: AnalysisOptions,
) -> None:
    record = job_store.set_running(job_id)
    output_dir = Path(record.output_dir)
    pre_vlm_output_path = output_dir / "pre_vlm_context.json"
    final_output_path = output_dir / "accident_facts.json"

    try:
        job_store.set_progress(job_id, stage="preprocessing", progress_message="pre-vlm analysis running")
        config = config_from_options(options, output_dir)
        context = analyze_video_pre_vlm(video_path=video_path, config=config)
        pre_vlm_output_path.write_text(
            json.dumps(context.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        job_store.set_progress(
            job_id,
            stage="vlm_composition" if options.mode == AnalysisMode.FULL else "pre_vlm_complete",
            progress_message="pre-vlm context written",
            pre_vlm_output_path=pre_vlm_output_path,
        )

        final_path: Path | None = None
        if options.mode == AnalysisMode.FULL:
            backend = get_qwen_backend(config.qwen_model_id, config.device)
            final_facts = compose_final_facts(context, backend)
            write_final_facts(final_facts, final_output_path)
            final_path = final_output_path
            job_store.set_progress(
                job_id,
                stage="writing_result",
                progress_message="final VLM facts written",
                final_output_path=final_path,
            )

        job_store.set_succeeded(job_id, pre_vlm_output_path, final_path)
    except Exception as exc:  # noqa: BLE001
        job_store.set_failed(job_id, f"{exc}\n{traceback.format_exc()}")
