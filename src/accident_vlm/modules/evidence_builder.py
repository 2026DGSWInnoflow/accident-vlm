from copy import deepcopy

from accident_vlm.schemas.preprocessing import PipelineContext


def build_evidence_package(context: PipelineContext) -> dict:
    metadata = context.video_metadata.model_dump() if context.video_metadata else {}
    return {
        "frames": [frame.model_dump() for frame in context.selected_frames],
        "selected_segments": deepcopy(context.selected_segments),
        "overlays": deepcopy(context.overlays),
        "crops": deepcopy(context.crops),
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
