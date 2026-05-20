from copy import deepcopy
from pathlib import Path
from typing import Any

from accident_vlm.schemas.preprocessing import PipelineContext


def build_evidence_package(context: PipelineContext) -> dict:
    metadata = context.video_metadata.model_dump() if context.video_metadata else {}
    evidence_images = collect_evidence_images(context)
    context.evidence_images = deepcopy(evidence_images)
    return {
        "frames": [frame.model_dump() for frame in context.selected_frames],
        "selected_segments": deepcopy(context.selected_segments),
        "overlays": deepcopy(context.overlays),
        "crops": deepcopy(context.crops),
        "evidence_images": evidence_images,
        "precomputed_facts": {
            "metadata": metadata,
            "input_quality": context.input_quality.model_dump() if context.input_quality else {},
            "ocr": deepcopy(context.ocr_observations),
            "ocr_summary": deepcopy(context.ocr_summary),
            "scene_type_candidates": deepcopy(context.scene_type_candidates),
            "tracks": deepcopy(context.tracks),
            "road_geometry": deepcopy(context.road_geometry),
            "speed_estimates": deepcopy(context.speed_and_distance),
            "traffic_control": deepcopy(context.traffic_control),
            "event_candidates": deepcopy(context.event_candidates),
        },
    }


def collect_evidence_images(context: PipelineContext) -> list[dict]:
    records: list[dict] = []
    seen_paths: set[str] = set()

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
            )

    for item in [*context.overlays, *context.crops]:
        if isinstance(item, dict):
            _append_from_dict(records, seen_paths, item, source=item.get("purpose", "visual_evidence"))

    for observation in context.ocr_observations:
        if isinstance(observation, dict):
            _append_from_dict(records, seen_paths, observation, source="ocr_roi", path_key="image_path")

    _walk_nested_images(records, seen_paths, context.road_geometry, source="road_geometry")
    _walk_nested_images(records, seen_paths, context.traffic_control, source="traffic_control")
    return records


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
    records.append(record)
