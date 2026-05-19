from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2

from accident_vlm.schemas.preprocessing import SelectedFrame


SUPPORTED_ACTOR_LABELS = {
    "car": "승용차",
    "truck": "화물차",
    "bus": "버스",
    "motorcycle": "이륜차",
    "bicycle": "자전거",
    "person": "보행자",
}


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    bbox: list[int]


class ObjectDetector(Protocol):
    name: str

    def detect(self, image_path: Path) -> list[Detection]:
        ...


class NoObjectDetector:
    name = "none"

    def detect(self, image_path: Path) -> list[Detection]:
        return []


class UltralyticsDetector:
    name = "ultralytics"

    def __init__(self, model_name: str) -> None:
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as exc:
            raise RuntimeError("ultralytics is not installed") from exc
        self._model = YOLO(model_name)

    def detect(self, image_path: Path) -> list[Detection]:
        results = self._model(str(image_path), verbose=False)
        detections: list[Detection] = []
        for result in results:
            names = result.names
            for box in result.boxes:
                cls_id = int(box.cls[0])
                raw_label = names.get(cls_id, str(cls_id))
                label = SUPPORTED_ACTOR_LABELS.get(raw_label, raw_label)
                xyxy = [int(value) for value in box.xyxy[0].tolist()]
                detections.append(
                    Detection(label=label, confidence=float(box.conf[0]), bbox=xyxy)
                )
        return detections


def create_object_detector(backend: str, model_name: str) -> ObjectDetector:
    if backend in {"none", "disabled"}:
        return NoObjectDetector()
    if backend in {"auto", "ultralytics", "yolo"}:
        try:
            return UltralyticsDetector(model_name)
        except RuntimeError:
            return NoObjectDetector()
    return NoObjectDetector()


def _center(bbox: list[int]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def _distance(first: tuple[float, float], second: tuple[float, float]) -> float:
    return ((first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2) ** 0.5


def _relative_position(center_x: float, center_y: float, width: int, height: int) -> str:
    horizontal = "좌측" if center_x < width * 0.4 else "우측" if center_x > width * 0.6 else "중앙"
    vertical = "전방" if center_y < height * 0.65 else "근접"
    return f"{vertical}{horizontal}" if horizontal != "중앙" else vertical


def _movement_from_points(first: tuple[float, float], last: tuple[float, float]) -> str:
    dx = last[0] - first[0]
    dy = last[1] - first[1]
    if abs(dx) > abs(dy) and abs(dx) > 20:
        return "차로변경_우" if dx > 0 else "차로변경_좌"
    if dy > 20:
        return "접근"
    if dy < -20:
        return "이탈"
    return "직진"


def detect_and_track_actors(
    selected_frames: list[SelectedFrame],
    detector: ObjectDetector,
    max_match_distance_px: float = 120.0,
) -> list[dict]:
    tracks: list[dict] = []
    next_track_index = 1

    for frame in selected_frames:
        if not frame.path:
            continue
        image = cv2.imread(frame.path)
        if image is None:
            continue
        height, width = image.shape[:2]
        detections = detector.detect(Path(frame.path))
        for detection in detections:
            center = _center(detection.bbox)
            candidate_index: int | None = None
            candidate_distance = max_match_distance_px
            for index, track in enumerate(tracks):
                if track["type"] != detection.label or not track["positions"]:
                    continue
                distance = _distance(center, track["positions"][-1]["center"])
                if distance < candidate_distance:
                    candidate_index = index
                    candidate_distance = distance
            if candidate_index is None:
                track = {
                    "track_id": f"T{next_track_index}",
                    "type": detection.label,
                    "role_candidate": "상대 차량" if detection.label != "보행자" else "보행자",
                    "positions": [],
                    "confidence": "medium",
                    "source": detector.name,
                }
                next_track_index += 1
                tracks.append(track)
                candidate_index = len(tracks) - 1
            tracks[candidate_index]["positions"].append(
                {
                    "time": frame.time,
                    "frame_id": frame.id,
                    "bbox": detection.bbox,
                    "center": center,
                    "confidence": detection.confidence,
                    "relative_position": _relative_position(center[0], center[1], width, height),
                }
            )

    for track in tracks:
        positions = track["positions"]
        if positions:
            first = positions[0]
            last = positions[-1]
            track["relative_position_start"] = first["relative_position"]
            track["relative_position_end"] = last["relative_position"]
            track["movement_candidate"] = _movement_from_points(first["center"], last["center"])
            for position in positions:
                position.pop("center", None)
        else:
            track["relative_position_start"] = "확인불가"
            track["relative_position_end"] = "확인불가"
            track["movement_candidate"] = "확인불가"
    return tracks
