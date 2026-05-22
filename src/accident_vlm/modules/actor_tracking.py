from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Protocol

import cv2

from accident_vlm.modules.frame_selection import extract_selected_frames
from accident_vlm.schemas.preprocessing import SelectedFrame
from accident_vlm.utils.timecode import frame_to_timecode, parse_timecode


SUPPORTED_ACTOR_LABELS = {
    "car": "승용차",
    "truck": "화물차",
    "bus": "버스",
    "motorcycle": "이륜차",
    "bicycle": "자전거",
    "person": "보행자",
}

ACCIDENT_ACTOR_TAXONOMY = {
    "passenger_car": {"korean_label": "승용차", "source": "custom_accident_taxonomy"},
    "truck": {"korean_label": "화물차", "source": "custom_accident_taxonomy"},
    "bus": {"korean_label": "버스", "source": "custom_accident_taxonomy"},
    "motorcycle": {"korean_label": "이륜차", "source": "custom_accident_taxonomy"},
    "bicycle": {"korean_label": "자전거", "source": "custom_accident_taxonomy"},
    "pedestrian": {"korean_label": "보행자", "source": "custom_accident_taxonomy"},
    "kickboard": {"korean_label": "전동킥보드", "source": "custom_accident_taxonomy"},
    "traffic_light": {"korean_label": "신호등", "source": "custom_accident_taxonomy"},
    "left_turn_signal": {"korean_label": "좌회전신호", "source": "custom_accident_taxonomy"},
    "speed_limit_sign": {"korean_label": "제한속도표지", "source": "custom_accident_taxonomy"},
    "stop_sign": {"korean_label": "정지표지", "source": "custom_accident_taxonomy"},
    "crosswalk": {"korean_label": "횡단보도", "source": "custom_accident_taxonomy"},
    "lane": {"korean_label": "차선", "source": "custom_accident_taxonomy"},
    "stop_line": {"korean_label": "정지선", "source": "custom_accident_taxonomy"},
    "centerline": {"korean_label": "중앙선", "source": "custom_accident_taxonomy"},
}

COCO_PRETRAINED_CLASSES = {"승용차", "화물차", "버스", "이륜차", "자전거", "보행자"}


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    bbox: list[int]
    track_id: str | None = None


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

    def __init__(self, model_name: str, model=None) -> None:
        if model is not None:
            self._model = model
            return
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


class UltralyticsTracker(UltralyticsDetector):
    name = "ultralytics_track"

    def __init__(
        self,
        model_name: str,
        tracker_config: str = "bytetrack.yaml",
        model=None,
    ) -> None:
        super().__init__(model_name, model=model)
        self.tracker_config = tracker_config

    def detect(self, image_path: Path) -> list[Detection]:
        results = self._model.track(
            source=str(image_path),
            tracker=self.tracker_config,
            persist=True,
            verbose=False,
        )
        detections: list[Detection] = []
        for result in results:
            names = result.names
            for box in result.boxes:
                cls_id = int(box.cls[0])
                raw_label = names.get(cls_id, str(cls_id))
                label = SUPPORTED_ACTOR_LABELS.get(raw_label, raw_label)
                xyxy = [int(value) for value in box.xyxy[0]]
                raw_track_id = getattr(box, "id", None)
                track_id = None
                if raw_track_id is not None:
                    track_id = f"T{int(raw_track_id[0])}"
                detections.append(
                    Detection(
                        label=label,
                        confidence=float(box.conf[0]),
                        bbox=xyxy,
                        track_id=track_id,
                    )
                )
        return detections


@lru_cache(maxsize=8)
def create_object_detector(backend: str, model_name: str) -> ObjectDetector:
    if backend in {"none", "disabled"}:
        return NoObjectDetector()
    if backend in {"bytetrack", "ultralytics_track", "track"}:
        try:
            return UltralyticsTracker(model_name, tracker_config="bytetrack.yaml")
        except RuntimeError:
            return NoObjectDetector()
    if backend in {"botsort", "bot-sort", "bo-sort"}:
        try:
            return UltralyticsTracker(model_name, tracker_config="botsort.yaml")
        except RuntimeError:
            return NoObjectDetector()
    if backend in {"auto", "ultralytics", "yolo"}:
        try:
            return UltralyticsDetector(model_name)
        except RuntimeError:
            return NoObjectDetector()
    return NoObjectDetector()


def detector_profile(model_name: str, backend: str) -> dict:
    custom_traffic_model = any(token in model_name.lower() for token in ("custom", "traffic", "accident"))
    return {
        "backend": backend,
        "model_name": model_name,
        "base_model_family": "custom_accident_traffic" if custom_traffic_model else "coco_pretrained",
        "custom_traffic_model": custom_traffic_model,
        "coco_supported_classes": sorted(COCO_PRETRAINED_CLASSES),
        "custom_taxonomy_classes": [
            value["korean_label"] for value in ACCIDENT_ACTOR_TAXONOMY.values()
        ],
    }


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


def _read_frame_shape(image_path: Path) -> tuple[int, int] | None:
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    height, width = image.shape[:2]
    return height, width


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
        shape = _read_frame_shape(Path(frame.path))
        if shape is None:
            continue
        height, width = shape
        detections = detector.detect(Path(frame.path))
        for detection in detections:
            center = _center(detection.bbox)
            candidate_index: int | None = None
            candidate_distance = max_match_distance_px
            if detection.track_id:
                for index, track in enumerate(tracks):
                    if track["track_id"] == detection.track_id:
                        candidate_index = index
                        break
            if candidate_index is None:
                for index, track in enumerate(tracks):
                    if track["type"] != detection.label or not track["positions"]:
                        continue
                    distance = _distance(center, track["positions"][-1]["center"])
                    if distance < candidate_distance:
                        candidate_index = index
                        candidate_distance = distance
            if candidate_index is None:
                track = {
                    "track_id": detection.track_id or f"T{next_track_index}",
                    "type": detection.label,
                    "role_candidate": "상대 차량" if detection.label != "보행자" else "보행자",
                    "positions": [],
                    "confidence": "medium",
                    "uncertainty_reasons": [],
                    "source": detector.name,
                    "tracking_method": "detector_track_id" if detection.track_id else "center_distance",
                }
                if detection.track_id is None:
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
            _update_track_confidence(tracks[candidate_index])

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


def _update_track_confidence(track: dict) -> None:
    confidences = [float(position.get("confidence") or 0.0) for position in track.get("positions", [])]
    if not confidences:
        track["confidence"] = "unknown"
        return
    mean_confidence = sum(confidences) / len(confidences)
    if mean_confidence >= 0.7:
        track["confidence"] = "high"
    elif mean_confidence >= 0.35:
        track["confidence"] = "medium"
    else:
        track["confidence"] = "low"
        reasons = track.setdefault("uncertainty_reasons", [])
        if "low_detector_confidence" not in reasons:
            reasons.append("low_detector_confidence")


def compare_tracker_outputs(bytetrack_tracks: list[dict], botsort_tracks: list[dict]) -> dict:
    bytetrack_ids = {str(track.get("track_id")) for track in bytetrack_tracks if isinstance(track, dict)}
    botsort_ids = {str(track.get("track_id")) for track in botsort_tracks if isinstance(track, dict)}
    shared = sorted(bytetrack_ids & botsort_ids)
    only_bytetrack = sorted(bytetrack_ids - botsort_ids)
    only_botsort = sorted(botsort_ids - bytetrack_ids)
    return {
        "bytetrack_count": len(bytetrack_tracks),
        "botsort_count": len(botsort_tracks),
        "shared_track_ids": shared,
        "only_bytetrack_track_ids": only_bytetrack,
        "only_botsort_track_ids": only_botsort,
        "disagreement_count": len(only_bytetrack) + len(only_botsort),
        "method": "track_id_overlap_comparison",
    }


def detect_and_track_segments(
    video_path: Path,
    selected_segments: list[dict],
    fps: float,
    detector: ObjectDetector,
    output_dir: Path,
    stride_frames: int = 3,
    max_frames_per_segment: int = 90,
) -> list[dict]:
    if fps <= 0:
        raise ValueError("fps must be positive")
    if stride_frames <= 0:
        raise ValueError("stride_frames must be positive")
    if max_frames_per_segment <= 0:
        raise ValueError("max_frames_per_segment must be positive")

    segment_frames: list[SelectedFrame] = []
    for segment in selected_segments:
        try:
            start_frame = int(round(parse_timecode(str(segment.get("start"))) * fps))
            end_frame = int(round(parse_timecode(str(segment.get("end"))) * fps))
        except ValueError:
            continue
        frame_indices = list(range(start_frame, max(start_frame, end_frame) + 1, stride_frames))
        for frame_index in frame_indices[:max_frames_per_segment]:
            segment_frames.append(
                SelectedFrame(
                    id=f"{segment.get('id', 'seg')}_frame_{frame_index:06d}",
                    time=frame_to_timecode(frame_index, fps),
                    frame_index=frame_index,
                    purpose="segment_tracking",
                )
            )

    extracted = extract_selected_frames(video_path, segment_frames, output_dir)
    tracks = detect_and_track_actors(extracted, detector)
    for track in tracks:
        track["source_stage"] = "segment_tracking"
    return tracks
