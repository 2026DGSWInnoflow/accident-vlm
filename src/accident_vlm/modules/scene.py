from __future__ import annotations

from accident_vlm.schemas.preprocessing import SelectedFrame


def classify_scene_candidates(
    selected_frames: list[SelectedFrame],
    road_geometry: dict,
    traffic_control: dict,
) -> list[dict]:
    candidates: list[dict] = []
    crosswalk = traffic_control.get("crosswalk", {})
    signal = traffic_control.get("signal", {})
    lane_count_value = road_geometry.get("visible_lane_count", {}).get("value")

    if crosswalk.get("visible"):
        candidates.append(
            {
                "value": "횡단보도",
                "confidence": crosswalk.get("confidence", "medium"),
                "source": "traffic_control",
                "evidence": crosswalk.get("evidence", []),
            }
        )
    if signal.get("visible"):
        candidates.append(
            {
                "value": "교차로",
                "confidence": "low",
                "source": "traffic_signal_presence",
                "evidence": signal.get("evidence", []),
                "note": "신호등 존재는 교차로 후보이나 단독 근거로 확정하지 않음",
            }
        )
    if isinstance(lane_count_value, int) and lane_count_value >= 2:
        candidates.append(
            {
                "value": "일반도로",
                "confidence": road_geometry.get("visible_lane_count", {}).get("confidence", "medium"),
                "source": "road_geometry",
                "evidence": road_geometry.get("visible_lane_count", {}).get("evidence", []),
            }
        )
    if not candidates:
        candidates.append(
            {
                "value": "확인불가",
                "confidence": "unknown",
                "source": "not_available",
                "evidence": [frame.id for frame in selected_frames[:3]],
            }
        )
    return candidates
