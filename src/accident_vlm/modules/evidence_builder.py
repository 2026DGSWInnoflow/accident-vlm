from copy import deepcopy
from pathlib import Path
from typing import Any

from accident_vlm.modules.evidence_scoring import rank_evidence_images, summarize_evidence_images
from accident_vlm.modules.vlm_storyboard import build_vlm_storyboard
from accident_vlm.schemas.preprocessing import PipelineContext


MAX_EVIDENCE_IMAGES = 160
PURPOSE_EVIDENCE_CAPS = {
    "traffic_light_crop": 12,
    "sign_crop": 8,
    "ocr_roi": 24,
    "actor_crop": 40,
    "tracking_overlay": 24,
    "track_overlay": 24,
}


def build_evidence_package(context: PipelineContext) -> dict:
    metadata = context.video_metadata.model_dump() if context.video_metadata else {}
    evidence_images = _cap_evidence_images(rank_evidence_images(collect_evidence_images(context)))
    context.evidence_images = deepcopy(evidence_images)
    package = {
        "frames": [frame.model_dump() for frame in context.selected_frames],
        "selected_segments": deepcopy(context.selected_segments),
        "overlays": deepcopy(context.overlays),
        "crops": deepcopy(context.crops),
        "contact_sheets": deepcopy(context.contact_sheets),
        "evidence_images": evidence_images,
        "precomputed_facts": {
            "metadata": metadata,
            "input_quality": context.input_quality.model_dump() if context.input_quality else {},
            "preprocessing_uncertainties": deepcopy(context.preprocessing_uncertainties),
            "ocr": deepcopy(context.ocr_observations),
            "ocr_summary": deepcopy(context.ocr_summary),
            "scene_type_candidates": deepcopy(context.scene_type_candidates),
            "tracks": deepcopy(context.tracks),
            "tracker_comparison": deepcopy(context.tracker_comparison),
            "road_geometry": deepcopy(context.road_geometry),
            "speed_estimates": deepcopy(context.speed_and_distance),
            "traffic_control": deepcopy(context.traffic_control),
            "event_scan_candidates": deepcopy(context.event_scan_candidates),
            "rejected_frame_candidates": deepcopy(context.rejected_frame_candidates),
            "event_candidates": deepcopy(context.event_candidates),
            "evidence_summary": summarize_evidence_images(evidence_images),
        },
    }
    package["vlm_storyboard"] = build_vlm_storyboard(package)
    return package


def collect_evidence_images(context: PipelineContext) -> list[dict]:
    records: list[dict] = []
    seen_paths: set[str] = set()
    selected_frame_quality = _selected_frame_quality_by_id(context)

    for frame in context.selected_frames:
        if frame.path:
            _append_image_record(
                records,
                seen_paths,
                path=frame.path,
                image_id=frame.id,
                purpose=frame.purpose,
                source="selected_frame",
                frame_id=frame.id,
                evidence_quality=selected_frame_quality.get(frame.id),
            )

    for item in [*context.overlays, *context.crops, *context.contact_sheets]:
        if isinstance(item, dict):
            _append_from_dict(records, seen_paths, item, source=item.get("purpose", "visual_evidence"))

    for observation in context.ocr_observations:
        if isinstance(observation, dict):
            _append_from_dict(records, seen_paths, observation, source="ocr_roi", path_key="image_path")

    _walk_nested_images(records, seen_paths, context.road_geometry, source="road_geometry")
    _walk_nested_images(records, seen_paths, context.traffic_control, source="traffic_control")
    return records


def _cap_evidence_images(records: list[dict]) -> list[dict]:
    capped: list[dict] = []
    purpose_counts: dict[str, int] = {}
    for record in records:
        purpose = str(record.get("purpose") or "")
        purpose_count = purpose_counts.get(purpose, 0)
        purpose_limit = PURPOSE_EVIDENCE_CAPS.get(purpose)
        if purpose_limit is not None and purpose_count >= purpose_limit:
            continue
        capped.append(record)
        purpose_counts[purpose] = purpose_count + 1
        if len(capped) >= MAX_EVIDENCE_IMAGES:
            break
    return capped


def _walk_nested_images(records: list[dict], seen_paths: set[str], value: Any, source: str) -> None:
    if isinstance(value, dict):
        _append_from_dict(records, seen_paths, value, source=source)
        for key in ("mask_path", "image_path"):
            if value.get(key):
                _append_from_dict(records, seen_paths, value, source=source, path_key=key)
        for child in value.values():
            _walk_nested_images(records, seen_paths, child, source)
    elif isinstance(value, list):
        for child in value:
            _walk_nested_images(records, seen_paths, child, source)


def _append_from_dict(
    records: list[dict],
    seen_paths: set[str],
    item: dict,
    source: str,
    path_key: str = "path",
) -> None:
    path = item.get(path_key)
    if not path:
        return
    _append_image_record(
        records,
        seen_paths,
        path=path,
        image_id=item.get("id") or Path(str(path)).stem,
        purpose=item.get("purpose") or source,
        source=source,
        frame_id=item.get("frame_id"),
    )


def _append_image_record(
    records: list[dict],
    seen_paths: set[str],
    *,
    path: str,
    image_id: str,
    purpose: str,
    source: str,
    frame_id: str | None = None,
    evidence_quality: dict | None = None,
) -> None:
    if path in seen_paths:
        return
    seen_paths.add(path)
    record = {
        "id": image_id,
        "path": path,
        "purpose": purpose,
        "source": source,
    }
    if frame_id:
        record["frame_id"] = frame_id
    if evidence_quality:
        record["evidence_quality"] = deepcopy(evidence_quality)
        record["quality_confidence"] = evidence_quality.get("analysis_reliability")
    records.append(record)


def _selected_frame_quality_by_id(context: PipelineContext) -> dict[str, dict]:
    if not context.input_quality:
        return {}
    quality_by_id: dict[str, dict] = {}
    for item in context.input_quality.timeline:
        if not isinstance(item, dict) or not item.get("frame_id"):
            continue
        quality_by_id[str(item["frame_id"])] = _quality_from_timeline_item(item)
    return quality_by_id


def _quality_from_timeline_item(item: dict) -> dict:
    flags = set(item.get("quality_flags") or [])
    weak_count = sum(
        flag in flags
        for flag in (
            "blur",
            "low_light",
            "overexposure",
            "night_noise",
            "glare",
            "low_contrast_fog_or_dirty_lens_candidate",
        )
    )
    reliability = "low" if weak_count >= 2 else "medium" if weak_count else "high"
    brightness = "dark" if "low_light" in flags else "overexposed" if "overexposure" in flags else "normal"
    return {
        "blur": "high" if "blur" in flags else "low",
        "brightness": brightness,
        "night_noise": "high" if "night_noise" in flags else "low",
        "analysis_reliability": reliability,
        "blur_score": round(float(item.get("blur_score", 0.0) or 0.0), 3),
        "brightness_score": round(float(item.get("brightness_score", 0.0) or 0.0), 3),
        "noise_score": round(float(item.get("noise_score", 0.0) or 0.0), 3),
        "glare_ratio": round(float(item.get("glare_ratio", 0.0) or 0.0), 5),
        "contrast_score": round(float(item.get("contrast_score", 0.0) or 0.0), 3),
    }
