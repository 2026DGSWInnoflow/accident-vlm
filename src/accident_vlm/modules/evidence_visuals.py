from __future__ import annotations

from pathlib import Path
from html import escape

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
            padded_bbox = _pad_bbox(bbox, image.shape[1], image.shape[0])
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

            px1, py1, px2, py2 = padded_bbox
            crop = image[py1:py2, px1:px2]
            if crop.size:
                quality = _crop_quality(bbox, image.shape[1], image.shape[0], track)
                crop_to_write = _upscale_small_crop(crop)
                crop_path = crop_dir / f"{frame_id}_{track.get('track_id')}.jpg"
                cv2.imwrite(str(crop_path), crop_to_write)
                crops.append(
                    {
                        "id": f"crop_{frame_id}_{track.get('track_id')}",
                        "path": str(crop_path),
                        "frame_id": frame_id,
                        "time": frame.time,
                        "track_id": track.get("track_id"),
                        "type": track.get("type"),
                        "bbox": bbox,
                        "padded_bbox": padded_bbox,
                        "context_padding_px": max(px1 - x1, y1 - py1, x2 - px2, y2 - py2, 0),
                        "purpose": "actor_crop",
                        "quality": quality,
                        "risk": _crop_risk(quality),
                        "why_selected": "tracked actor appears in selected frame",
                        "expected_use": "actor identity, position, movement, and collision-context verification",
                        "source_frame_path": frame.path,
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
                "paired_original_frame_id": frame_id,
                "paired_original_path": frame.path,
            }
        )

    report_path = _write_evidence_report(output_dir, overlays, crops)
    for overlay in overlays:
        overlay["report_path"] = str(report_path)
    return overlays, crops


def build_event_evidence_overlays(
    selected_frames: list[SelectedFrame],
    event_candidates: list[dict],
    tracks: list[dict],
    output_dir: Path,
    max_events: int = 8,
) -> list[dict]:
    event_dir = output_dir / "event_overlays"
    event_dir.mkdir(parents=True, exist_ok=True)
    frames_by_id = {frame.id: frame for frame in selected_frames if frame.path}
    positions_by_frame = _positions_by_frame(tracks)
    overlays: list[dict] = []
    for event_index, event in enumerate(
        sorted(event_candidates, key=lambda item: -float(item.get("event_score") or 0.0))[:max_events]
    ):
        evidence_ids = [str(item) for item in event.get("evidence", []) if str(item) in frames_by_id]
        if not evidence_ids:
            continue
        rendered_images = []
        for frame_id in evidence_ids[:2]:
            frame = frames_by_id[frame_id]
            image = cv2.imread(frame.path)
            if image is None:
                continue
            overlay = image.copy()
            for track, position in positions_by_frame.get(frame_id, []):
                bbox = _clip_bbox(position.get("bbox", []), image.shape[1], image.shape[0])
                if bbox is None:
                    continue
                x1, y1, x2, y2 = bbox
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 80, 255), 2)
                cv2.putText(
                    overlay,
                    str(track.get("track_id")),
                    (x1, max(16, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
            cv2.putText(
                overlay,
                f"{event.get('event_type', 'event')} score={event.get('event_score', '')}",
                (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            rendered_images.append(overlay)
        if not rendered_images:
            continue
        if len(rendered_images) == 1:
            pair_image = rendered_images[0]
        else:
            pair_image = cv2.hconcat(_match_heights(rendered_images))
        overlay_path = event_dir / f"event_{event_index:02d}_{_safe_event_id(event)}.jpg"
        cv2.imwrite(str(overlay_path), pair_image)
        record = {
            "id": f"event_overlay_{event_index:02d}",
            "path": str(overlay_path),
            "frame_id": evidence_ids[0],
            "event_type": event.get("event_type"),
            "event_score": event.get("event_score"),
            "candidate_class": event.get("candidate_class"),
            "evidence": evidence_ids[:2],
            "purpose": "event_candidate_overlay",
            "why_selected": "event candidate supporting frame pair",
            "expected_use": "collision timing, actor proximity, and uncertainty verification",
            "risk": event.get("contradicting_signals", []),
        }
        overlays.append(record)
        event.setdefault("overlay_evidence", []).append(record["id"])
    return overlays


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


def _positions_by_frame(tracks: list[dict]) -> dict[str, list[tuple[dict, dict]]]:
    positions_by_frame: dict[str, list[tuple[dict, dict]]] = {}
    for track in tracks:
        for position in track.get("positions", []):
            positions_by_frame.setdefault(position.get("frame_id", ""), []).append((track, position))
    return positions_by_frame


def _match_heights(images: list) -> list:
    max_height = max(image.shape[0] for image in images)
    matched = []
    for image in images:
        if image.shape[0] == max_height:
            matched.append(image)
            continue
        scale = max_height / image.shape[0]
        matched.append(cv2.resize(image, (int(image.shape[1] * scale), max_height), interpolation=cv2.INTER_AREA))
    return matched


def _safe_event_id(event: dict) -> str:
    value = str(event.get("id") or event.get("event_type") or "candidate")
    return "".join(character if character.isalnum() else "_" for character in value)[:48]


def _pad_bbox(bbox: list[int], width: int, height: int, ratio: float = 0.35) -> list[int]:
    x1, y1, x2, y2 = bbox
    pad = max(8, int(max(x2 - x1, y2 - y1) * ratio))
    return [
        max(0, x1 - pad),
        max(0, y1 - pad),
        min(width, x2 + pad),
        min(height, y2 + pad),
    ]


def _upscale_small_crop(crop):
    height, width = crop.shape[:2]
    min_side = min(height, width)
    if min_side >= 48:
        return crop
    scale = max(1, int(round(48 / max(1, min_side))))
    return cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def _crop_quality(bbox: list[int], width: int, height: int, track: dict) -> dict:
    x1, y1, x2, y2 = bbox
    area = max(0, x2 - x1) * max(0, y2 - y1)
    frame_area = max(1, width * height)
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    center_distance = (((center_x - width / 2) / width) ** 2 + ((center_y - height / 2) / height) ** 2) ** 0.5
    clipped = x1 <= 0 or y1 <= 0 or x2 >= width or y2 >= height
    confidence = str(track.get("confidence", "unknown"))
    return {
        "area_px": area,
        "area_ratio": round(area / frame_area, 5),
        "too_small": area < 32 * 32,
        "center_distance_norm": round(center_distance, 4),
        "clipped": clipped,
        "track_confidence": confidence,
    }


def _crop_risk(quality: dict) -> list[str]:
    risks = []
    if quality.get("too_small"):
        risks.append("small_crop")
    if quality.get("clipped"):
        risks.append("bbox_clipped")
    if quality.get("track_confidence") in {"low", "unknown"}:
        risks.append("low_track_confidence")
    if quality.get("center_distance_norm", 0) > 0.45:
        risks.append("edge_object")
    return risks


def _write_evidence_report(output_dir: Path, overlays: list[dict], crops: list[dict]) -> Path:
    report_path = output_dir / "evidence_report.html"
    rows = []
    for crop in crops:
        rows.append(
            "<tr>"
            f"<td>{escape(str(crop.get('frame_id')))}</td>"
            f"<td>{escape(str(crop.get('track_id')))}</td>"
            f"<td>{escape(str(crop.get('type')))}</td>"
            f"<td>{escape(','.join(crop.get('risk', [])))}</td>"
            f"<td>{escape(str(crop.get('quality', {}).get('area_ratio')))}</td>"
            f"<td><img src='{escape(Path(crop['path']).as_posix())}' width='120'></td>"
            "</tr>"
        )
    report_path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>Evidence Report</title></head>"
        "<body><h1>Evidence Report</h1>"
        f"<p>overlays={len(overlays)} crops={len(crops)}</p>"
        "<table border='1' cellspacing='0' cellpadding='4'>"
        "<tr><th>frame</th><th>track</th><th>type</th><th>risk</th><th>area_ratio</th><th>crop</th></tr>"
        + "".join(rows)
        + "</table></body></html>",
        encoding="utf-8",
    )
    return report_path
