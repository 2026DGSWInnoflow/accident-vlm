from pathlib import Path

from accident_vlm.config import PipelineConfig
from accident_vlm.modules.actor_tracking import create_object_detector, detect_and_track_actors
from accident_vlm.modules.evidence_builder import build_evidence_package
from accident_vlm.modules.evidence_visuals import build_visual_evidence
from accident_vlm.modules.event_detection import detect_event_candidates
from accident_vlm.modules.frame_selection import (
    build_event_segments,
    extract_selected_frames,
    merge_selected_frames,
    select_motion_keyframes,
    select_regular_frames,
)
from accident_vlm.modules.ingestion import probe_video
from accident_vlm.modules.ocr import (
    create_ocr_backend,
    extract_ocr_observations,
    summarize_ocr_observations,
)
from accident_vlm.modules.road_geometry import analyze_road_geometry
from accident_vlm.modules.scene import classify_scene_candidates
from accident_vlm.modules.speed_distance import estimate_speed_and_distance
from accident_vlm.modules.traffic_control import analyze_traffic_control
from accident_vlm.modules.video_quality import analyze_input_quality
from accident_vlm.schemas.preprocessing import PipelineContext, VideoMetadata


def build_pre_vlm_context(
    video_path: str,
    metadata: VideoMetadata,
    config: PipelineConfig | None = None,
) -> PipelineContext:
    active_config = config or PipelineConfig()
    selected_frames = select_regular_frames(
        duration_sec=metadata.duration_sec,
        fps=metadata.fps,
        interval_sec=active_config.regular_frame_interval_sec,
        max_frames=active_config.max_selected_frames,
    )
    selected_frames = [
        frame for frame in selected_frames if frame.frame_index < metadata.frame_count
    ]
    context = PipelineContext(
        video_path=video_path,
        video_metadata=metadata,
        selected_frames=selected_frames,
    )
    context.evidence_package = build_evidence_package(context)
    return context


def analyze_video_pre_vlm(
    video_path: Path,
    config: PipelineConfig | None = None,
) -> PipelineContext:
    active_config = config or PipelineConfig()
    metadata = probe_video(video_path)
    context = build_pre_vlm_context(str(video_path), metadata, active_config)
    if active_config.enable_motion_keyframes:
        motion_frames = select_motion_keyframes(
            video_path=video_path,
            metadata=metadata,
            sample_interval_sec=active_config.motion_sample_interval_sec,
            max_frames=active_config.max_motion_keyframes,
            min_change_score=active_config.min_motion_change_score,
        )
        context.selected_frames = merge_selected_frames(context.selected_frames, motion_frames)

    run_output_dir = active_config.output_dir / Path(video_path).stem
    frame_output_dir = run_output_dir / active_config.frame_output_dirname
    context.selected_frames = extract_selected_frames(
        video_path=video_path,
        selected_frames=context.selected_frames,
        output_dir=frame_output_dir,
    )

    context.input_quality = analyze_input_quality(video_path, context.selected_frames)

    if active_config.enable_ocr:
        ocr_backend = create_ocr_backend(active_config.ocr_backend)
        context.ocr_observations = extract_ocr_observations(
            context.selected_frames,
            ocr_backend,
            roi_output_dir=run_output_dir / "ocr_rois",
        )
        context.ocr_summary = summarize_ocr_observations(context.ocr_observations)

    if active_config.enable_actor_tracking:
        detector = create_object_detector(
            active_config.object_detector_backend,
            active_config.object_detector_model,
        )
        context.tracks = detect_and_track_actors(context.selected_frames, detector)
        context.overlays, context.crops = build_visual_evidence(
            context.selected_frames,
            context.tracks,
            run_output_dir,
        )

    if active_config.enable_road_geometry:
        context.road_geometry = analyze_road_geometry(
            context.selected_frames,
            lane_width_m=active_config.lane_width_m,
            output_dir=run_output_dir / "road_geometry",
        )

    if active_config.enable_speed_distance:
        context.speed_and_distance = estimate_speed_and_distance(
            context.ocr_observations,
            context.tracks,
            context.road_geometry,
            context.ocr_summary,
        )

    if active_config.enable_traffic_control:
        context.traffic_control = analyze_traffic_control(
            context.selected_frames,
            context.ocr_observations,
            output_dir=run_output_dir / "traffic_control",
        )

    if active_config.enable_scene_analysis:
        context.scene_type_candidates = classify_scene_candidates(
            context.selected_frames,
            context.road_geometry,
            context.traffic_control,
        )

    if active_config.enable_event_detection:
        context.event_candidates = detect_event_candidates(
            context.tracks,
            context.speed_and_distance,
            context.input_quality.model_dump() if context.input_quality else None,
        )
        context.selected_segments = build_event_segments(
            context.event_candidates,
            metadata,
            active_config.pre_event_window_sec,
            active_config.post_event_window_sec,
        )

    context.evidence_package = build_evidence_package(context)
    return context
