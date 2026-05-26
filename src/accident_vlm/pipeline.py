from pathlib import Path

from accident_vlm.config import PipelineConfig
from accident_vlm.modules.event_detection import detect_event_candidates
from accident_vlm.modules.frame_selection import (
    build_event_segments,
    extract_selected_frames,
    merge_selected_frames,
    select_event_window_frames,
    select_motion_keyframes,
    select_regular_frames,
)
from accident_vlm.modules.ingestion import probe_video
from accident_vlm.modules.ocr import (
    create_ocr_backend,
    extract_ocr_observations,
    summarize_ocr_observations,
)
from accident_vlm.modules.scene import classify_scene_candidates
from accident_vlm.modules.speed_distance import estimate_speed_and_distance
from accident_vlm.modules.track_consolidation import consolidate_tracks
from accident_vlm.modules.video_quality import analyze_input_quality
from accident_vlm.schemas.preprocessing import PipelineContext, VideoMetadata


def build_evidence_package(*args, **kwargs):
    from accident_vlm.modules.evidence_builder import build_evidence_package as implementation

    return implementation(*args, **kwargs)


def build_event_evidence_overlays(*args, **kwargs):
    from accident_vlm.modules.evidence_visuals import build_event_evidence_overlays as implementation

    return implementation(*args, **kwargs)


def build_frame_selection_contact_sheet(*args, **kwargs):
    from accident_vlm.modules.event_scan import build_frame_selection_contact_sheet as implementation

    return implementation(*args, **kwargs)


def build_visual_evidence(*args, **kwargs):
    from accident_vlm.modules.evidence_visuals import build_visual_evidence as implementation

    return implementation(*args, **kwargs)


def compare_tracker_outputs(*args, **kwargs):
    from accident_vlm.modules.actor_tracking import compare_tracker_outputs as implementation

    return implementation(*args, **kwargs)


def create_object_detector(*args, **kwargs):
    from accident_vlm.modules.actor_tracking import create_object_detector as implementation

    return implementation(*args, **kwargs)


def detect_and_track_actors(*args, **kwargs):
    from accident_vlm.modules.actor_tracking import detect_and_track_actors as implementation

    return implementation(*args, **kwargs)


def detect_and_track_segments(*args, **kwargs):
    from accident_vlm.modules.actor_tracking import detect_and_track_segments as implementation

    return implementation(*args, **kwargs)


def scan_video_event_candidates(*args, **kwargs):
    from accident_vlm.modules.event_scan import scan_video_event_candidates as implementation

    return implementation(*args, **kwargs)


def select_precision_event_frames(*args, **kwargs):
    from accident_vlm.modules.event_scan import select_precision_event_frames as implementation

    return implementation(*args, **kwargs)


def analyze_road_geometry(*args, **kwargs):
    from accident_vlm.modules.road_geometry import analyze_road_geometry as implementation

    return implementation(*args, **kwargs)


def analyze_traffic_control(*args, **kwargs):
    from accident_vlm.modules.traffic_control import analyze_traffic_control as implementation

    return implementation(*args, **kwargs)


def build_pre_vlm_context(
    video_path: str,
    metadata: VideoMetadata,
    config: PipelineConfig | None = None,
    build_initial_evidence_package: bool = True,
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
    if build_initial_evidence_package:
        context.evidence_package = build_evidence_package(context)
    return context


def analyze_video_pre_vlm(
    video_path: Path,
    config: PipelineConfig | None = None,
) -> PipelineContext:
    active_config = config or PipelineConfig()
    metadata = probe_video(video_path)
    context = build_pre_vlm_context(
        str(video_path),
        metadata,
        active_config,
        build_initial_evidence_package=False,
    )
    precision_frames: list = []
    if active_config.enable_event_scan:
        context.event_scan_candidates = scan_video_event_candidates(
            video_path=video_path,
            metadata=metadata,
            sample_fps=active_config.event_scan_sample_fps,
            top_k=active_config.event_scan_top_k,
            min_score=active_config.event_scan_min_score,
            pre_event_window_sec=active_config.pre_event_window_sec,
            post_event_window_sec=active_config.post_event_window_sec,
        )
        precision_frames, rejected_frames = select_precision_event_frames(
            context.event_scan_candidates,
            metadata,
            max_frames=active_config.vlm_frame_budget,
            pre_event_window_sec=active_config.pre_event_window_sec,
            post_event_window_sec=active_config.post_event_window_sec,
            precision_fps=active_config.precision_event_fps,
            min_impact_frames=active_config.min_impact_frames,
        )
        context.selected_frames = merge_selected_frames(context.selected_frames, precision_frames)
        context.rejected_frame_candidates = rejected_frames

    if active_config.enable_motion_keyframes and not active_config.enable_event_scan:
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

    context.input_quality = analyze_input_quality(
        video_path,
        context.selected_frames,
        event_windows=context.event_scan_candidates,
    )

    if active_config.enable_ocr and _ocr_backend_enabled(active_config.ocr_backend):
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
        context.tracks = consolidate_tracks(context.tracks)
        if active_config.enable_tracker_comparison:
            alternate_backend = "botsort" if active_config.object_detector_backend != "botsort" else "bytetrack"
            alternate_detector = create_object_detector(
                alternate_backend,
                active_config.object_detector_model,
            )
            alternate_tracks = consolidate_tracks(
                detect_and_track_actors(context.selected_frames, alternate_detector)
            )
            if active_config.object_detector_backend == "botsort":
                context.tracker_comparison = compare_tracker_outputs(alternate_tracks, context.tracks)
            else:
                context.tracker_comparison = compare_tracker_outputs(context.tracks, alternate_tracks)
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
            lane_segmentation_model_path=active_config.lane_segmentation_model_path,
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
        detected_event_candidates = detect_event_candidates(
            context.tracks,
            context.speed_and_distance,
            context.input_quality.model_dump() if context.input_quality else None,
            context.event_scan_candidates,
        )
        context.event_candidates = _limit_event_candidates(
            [*detected_event_candidates, *context.event_scan_candidates],
            active_config.max_event_candidates,
        )
        context.selected_segments = build_event_segments(
            context.event_candidates,
            metadata,
            active_config.pre_event_window_sec,
            active_config.post_event_window_sec,
        )
        if (
            active_config.enable_actor_tracking
            and active_config.enable_segment_tracking
            and context.selected_segments
        ):
            segment_tracks = detect_and_track_segments(
                video_path=video_path,
                selected_segments=context.selected_segments,
                fps=metadata.fps,
                detector=detector,
                output_dir=run_output_dir / "segment_tracking_frames",
                stride_frames=active_config.segment_tracking_stride_frames,
                max_frames_per_segment=active_config.max_segment_tracking_frames,
                max_total_frames=active_config.max_segment_tracking_frames,
            )
            context.tracks = consolidate_tracks([*context.tracks, *segment_tracks])
            context.overlays, context.crops = build_visual_evidence(
                context.selected_frames,
                context.tracks,
                run_output_dir,
            )
        context.overlays.extend(
            build_event_evidence_overlays(
                context.selected_frames,
                context.event_candidates,
                context.tracks,
                run_output_dir,
            )
        )
        context.preprocessing_uncertainties = _collect_preprocessing_uncertainties(context)
        event_window_frames = select_event_window_frames(
            context.event_candidates,
            metadata,
            max_frames=active_config.vlm_frame_budget,
            pre_event_window_sec=active_config.pre_event_window_sec,
            post_event_window_sec=active_config.post_event_window_sec,
        )
        context.selected_frames = extract_selected_frames(
            video_path=video_path,
            selected_frames=merge_selected_frames(context.selected_frames, event_window_frames),
            output_dir=frame_output_dir,
        )

    contact_sheet = build_frame_selection_contact_sheet(
        context.selected_frames,
        run_output_dir / "reports" / "frame_selection_contact_sheet.jpg",
        title=f"{Path(video_path).name} frame selection",
    )
    if contact_sheet.get("status") in {"created", "reused"}:
        context.contact_sheets = [contact_sheet]

    if not context.preprocessing_uncertainties:
        context.preprocessing_uncertainties = _collect_preprocessing_uncertainties(context)
    context.evidence_package = build_evidence_package(context)
    return context


def _collect_preprocessing_uncertainties(context: PipelineContext) -> list[str]:
    uncertainties: list[str] = []
    if context.input_quality and context.input_quality.analysis_reliability in {"low", "medium"}:
        uncertainties.append(f"input_quality reliability={context.input_quality.analysis_reliability}")
    for track in context.tracks:
        if not isinstance(track, dict):
            continue
        if track.get("confidence") in {"low", "unknown"}:
            uncertainties.append(f"low confidence actor retained: {track.get('track_id', 'unknown')}")
        for reason in track.get("uncertainty_reasons", []):
            uncertainties.append(f"actor {track.get('track_id', 'unknown')}: {reason}")
    for event in context.event_candidates:
        if not isinstance(event, dict):
            continue
        if event.get("confidence") in {"low", "unknown"}:
            uncertainties.append(f"low confidence event candidate retained: {event.get('event_type', 'unknown')}")
    if context.tracker_comparison.get("disagreement_count"):
        uncertainties.append(
            f"tracker disagreement count={context.tracker_comparison.get('disagreement_count')}"
        )
    deduped = []
    for item in uncertainties:
        if item not in deduped:
            deduped.append(item)
    return deduped


def _ocr_backend_enabled(name: str) -> bool:
    return name.strip().lower() not in {"none", "disabled", "off", "false", "0"}


def _limit_event_candidates(event_candidates: list[dict], max_candidates: int) -> list[dict]:
    ranked = sorted(
        event_candidates,
        key=lambda event: (-float(event.get("event_score", 0.0) or 0.0), event.get("time") or ""),
    )
    return ranked[:max_candidates]
