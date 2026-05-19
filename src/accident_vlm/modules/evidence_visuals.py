from __future__ import annotations

from pathlib import Path

import cv2

from accident_vlm.schemas.preprocessing import SelectedFrame


def build_visual_evidence(
    selected_frames: list[SelectedFrame],
    tracks: list[dict],
    output_dir: Path,
) -> tuple[list[dict], list[dict]]:
    overlay_dir = output_dir / "overlays"
    crop_dir = output_dir / "crops"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)

    positions_by_frame: dict[str, list[tuple[dict, dict]]] = {}
    for track in tracks:
        for position in track.get("positions", []):
            positions_by_frame.setdefault(position.get("frame_id", ""), []).append((track, position))

    overlays: list[dict] = []
    crops: list[dict] = []
    frame_by_id = {frame.id: frame for frame in selected_frames}

    for frame_id, track_positions in positions_by_frame.items():
        frame = frame_by_id.get(frame_id)
        if not frame or not frame.path:
            continue
        image = cv2.imread(frame.path)
        if image is None:
            continue
        overlay = image.copy()
        for track, position in track_positions:
            bbox = _clip_bbox(position.get("bbox", []), image.shape[1], image.shape[0])
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            label = f"{track.get('track_id')} {track.get('type')} {track.get('movement_candidate', '')}"
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 180, 255), 2)
            cv2.putText(
                overlay,
                label,
                (x1, max(y1 - 6, 14)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                overlay,
                label,
                (x1, max(y1 - 6, 14)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

            crop = image[y1:y2, x1:x2]
            if crop.size:
                crop_path = crop_dir / f"{frame_id}_{track.get('track_id')}.jpg"
                cv2.imwrite(str(crop_path), crop)
                crops.append(
                    {
                        "id": f"crop_{frame_id}_{track.get('track_id')}",
                        "path": str(crop_path),
                        "frame_id": frame_id,
                        "time": frame.time,
                        "track_id": track.get("track_id"),
                        "type": track.get("type"),
                        "bbox": bbox,
                        "purpose": "actor_crop",
                    }
                )

        overlay_path = overlay_dir / f"{frame_id}_tracking.jpg"
        cv2.imwrite(str(overlay_path), overlay)
        overlays.append(
            {
                "id": f"overlay_{frame_id}_tracking",
                "path": str(overlay_path),
                "frame_id": frame_id,
                "time": frame.time,
                "purpose": "tracking_overlay",
                "tracks": [track.get("track_id") for track, _ in track_positions],
            }
        )

    return overlays, crops


def _clip_bbox(bbox: list[int], width: int, height: int) -> list[int] | None:
    if len(bbox) != 4:
        return None
    x1 = max(0, min(int(bbox[0]), width - 1))
    y1 = max(0, min(int(bbox[1]), height - 1))
    x2 = max(0, min(int(bbox[2]), width))
    y2 = max(0, min(int(bbox[3]), height))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]
