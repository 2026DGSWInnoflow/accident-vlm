from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from statistics import median
from typing import Protocol

import cv2
import numpy as np

from accident_vlm.schemas.preprocessing import SelectedFrame


class OcrBackend(Protocol):
    name: str

    def read_text(self, image_path: Path) -> list[dict]:
        ...


class UnavailableOcrBackend:
    name = "unavailable"

    def __init__(self, reason: str) -> None:
        self.reason = reason

    def read_text(self, image_path: Path) -> list[dict]:
        return [
            {
                "text": "",
                "confidence": 0.0,
                "bbox": None,
                "source": self.name,
                "note": self.reason,
                "image_path": str(image_path),
            }
        ]


class TesseractOcrBackend:
    name = "pytesseract"

    def __init__(self) -> None:
        try:
            import pytesseract  # type: ignore
        except ImportError as exc:
            raise RuntimeError("pytesseract is not installed") from exc
        self._pytesseract = pytesseract

    def read_text(self, image_path: Path) -> list[dict]:
        image = cv2.imread(str(image_path))
        if image is None:
            return []
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        data = self._pytesseract.image_to_data(
            rgb,
            output_type=self._pytesseract.Output.DICT,
            config="--psm 6",
        )
        observations: list[dict] = []
        for index, text in enumerate(data.get("text", [])):
            stripped = text.strip()
            if not stripped:
                continue
            confidence_raw = data.get("conf", ["-1"])[index]
            try:
                confidence = max(float(confidence_raw), 0.0) / 100.0
            except ValueError:
                confidence = 0.0
            observations.append(
                {
                    "text": stripped,
                    "confidence": confidence,
                    "bbox": [
                        int(data["left"][index]),
                        int(data["top"][index]),
                        int(data["left"][index] + data["width"][index]),
                        int(data["top"][index] + data["height"][index]),
                    ],
                    "source": self.name,
                    "image_path": str(image_path),
                }
            )
        return observations


class EasyOcrBackend:
    name = "easyocr"

    def __init__(self, languages: list[str] | None = None) -> None:
        try:
            import easyocr  # type: ignore
        except ImportError as exc:
            raise RuntimeError("easyocr is not installed") from exc
        self._reader = easyocr.Reader(languages or ["ko", "en"], gpu=True)

    def read_text(self, image_path: Path) -> list[dict]:
        results = self._reader.readtext(str(image_path))
        observations: list[dict] = []
        for bbox, text, confidence in results:
            xs = [int(point[0]) for point in bbox]
            ys = [int(point[1]) for point in bbox]
            observations.append(
                {
                    "text": text.strip(),
                    "confidence": float(confidence),
                    "bbox": [min(xs), min(ys), max(xs), max(ys)],
                    "source": self.name,
                    "image_path": str(image_path),
                }
            )
        return [observation for observation in observations if observation["text"]]


@lru_cache(maxsize=8)
def create_ocr_backend(name: str = "auto") -> OcrBackend:
    if name in {"auto", "easyocr"}:
        try:
            return EasyOcrBackend()
        except RuntimeError as exc:
            if name == "easyocr":
                return UnavailableOcrBackend(str(exc))
    if name in {"auto", "pytesseract", "tesseract"}:
        try:
            return TesseractOcrBackend()
        except RuntimeError as exc:
            return UnavailableOcrBackend(str(exc))
    if name in {"none", "disabled"}:
        return UnavailableOcrBackend("OCR disabled")
    return UnavailableOcrBackend(f"unknown OCR backend: {name}")


OVERLAY_ROIS = {
    "full_frame": (0.0, 0.0, 1.0, 1.0),
    "top_band": (0.0, 0.0, 1.0, 0.22),
    "bottom_band": (0.0, 0.72, 1.0, 1.0),
    "top_left": (0.0, 0.0, 0.45, 0.25),
    "top_right": (0.55, 0.0, 1.0, 0.25),
    "bottom_left": (0.0, 0.65, 0.55, 1.0),
    "bottom_right": (0.45, 0.65, 1.0, 1.0),
}


def parse_overlay_text(text: str) -> dict:
    normalized = _normalize_ocr_text(text)
    datetime_match = re.search(
        r"(?P<year>\d{4})[-./년\s]+(?P<month>\d{1,2})[-./월\s]+(?P<day>\d{1,2})"
        r"(?:일)?\s+(?P<hour>\d{1,2}):(?P<minute>\d{2}):(?P<second>\d{2})",
        normalized,
    )
    speed_match = re.search(
        r"(?:speed|spd|속도)?\s*[:=]?\s*(?P<speed>\d{1,3}(?:\.\d+)?)\s*(?:km/h|kph|kmh|㎞/h)",
        normalized,
        re.IGNORECASE,
    )
    gps_match = re.search(
        r"(?:gps|위치)?\s*[:=]?\s*(?:lat)?\s*(?P<lat_dir>[NS])?\s*"
        r"(?P<lat>-?\d{1,3}\.\d+)[,\s]+(?:lon|lng)?\s*(?P<lon_dir>[EW])?\s*"
        r"(?P<lon>-?\d{1,3}\.\d+)",
        normalized,
        re.IGNORECASE,
    )
    parsed: dict = {}
    if datetime_match:
        parsed["datetime"] = (
            f"{int(datetime_match.group('year')):04d}-"
            f"{int(datetime_match.group('month')):02d}-"
            f"{int(datetime_match.group('day')):02d} "
            f"{int(datetime_match.group('hour')):02d}:"
            f"{int(datetime_match.group('minute')):02d}:"
            f"{int(datetime_match.group('second')):02d}"
        )
    if speed_match:
        parsed["speed_kmh"] = float(speed_match.group("speed"))
    if gps_match:
        lat = float(gps_match.group("lat"))
        lon = float(gps_match.group("lon"))
        if (gps_match.group("lat_dir") or "").upper() == "S":
            lat = -abs(lat)
        if (gps_match.group("lon_dir") or "").upper() == "W":
            lon = -abs(lon)
        if abs(lat) <= 90 and abs(lon) <= 180:
            parsed["gps"] = {"lat": lat, "lon": lon}
    return parsed


def _normalize_ocr_text(text: str) -> str:
    return (
        text.replace("㎞", "km")
        .replace("Ｋ", "K")
        .replace("ｍ", "m")
        .replace("／", "/")
        .replace("|", " ")
        .strip()
    )


def build_overlay_rois(image: np.ndarray) -> list[dict]:
    height, width = image.shape[:2]
    rois: list[dict] = []
    for name, (x1_ratio, y1_ratio, x2_ratio, y2_ratio) in OVERLAY_ROIS.items():
        x1 = int(round(width * x1_ratio))
        y1 = int(round(height * y1_ratio))
        x2 = int(round(width * x2_ratio))
        y2 = int(round(height * y2_ratio))
        crop = image[y1:y2, x1:x2]
        if crop.size:
            rois.append({"name": name, "bbox": [x1, y1, x2, y2], "image": crop})
    return rois


def _offset_bbox(bbox: list[int] | None, origin_x: int, origin_y: int) -> list[int] | None:
    if bbox is None or len(bbox) != 4:
        return bbox
    return [bbox[0] + origin_x, bbox[1] + origin_y, bbox[2] + origin_x, bbox[3] + origin_y]


def extract_ocr_observations(
    selected_frames: list[SelectedFrame],
    backend: OcrBackend,
    roi_output_dir: Path | None = None,
) -> list[dict]:
    observations: list[dict] = []
    if roi_output_dir:
        roi_output_dir.mkdir(parents=True, exist_ok=True)
    for frame in selected_frames:
        if not frame.path:
            continue
        image = cv2.imread(frame.path)
        if image is None:
            continue
        for roi in build_overlay_rois(image):
            roi_path = Path(frame.path)
            if roi_output_dir:
                roi_path = roi_output_dir / f"{frame.id}_{roi['name']}.jpg"
                cv2.imwrite(str(roi_path), roi["image"])
            origin_x, origin_y = roi["bbox"][0], roi["bbox"][1]
            for item in backend.read_text(roi_path):
                observation = {
                    "time": frame.time,
                    "frame_id": frame.id,
                    "roi_name": roi["name"],
                    "roi_bbox": roi["bbox"],
                    **item,
                    "bbox": _offset_bbox(item.get("bbox"), origin_x, origin_y),
                }
                observation["parsed"] = parse_overlay_text(observation.get("text", ""))
                observations.append(observation)
    return observations


def summarize_ocr_observations(observations: list[dict]) -> dict:
    datetime_values: list[tuple[str, str, float]] = []
    speed_values: list[tuple[float, str, float]] = []
    gps_values: list[tuple[dict, str, float]] = []

    for observation in observations:
        parsed = observation.get("parsed", {})
        confidence = float(observation.get("confidence") or 0.0)
        frame_id = observation.get("frame_id", "unknown")
        if "datetime" in parsed:
            datetime_values.append((parsed["datetime"], frame_id, confidence))
        if "speed_kmh" in parsed:
            speed_values.append((float(parsed["speed_kmh"]), frame_id, confidence))
        if "gps" in parsed:
            gps_values.append((parsed["gps"], frame_id, confidence))

    return {
        "datetime": _vote_text_value(datetime_values, "ocr_overlay"),
        "speed": _summarize_speed_values(speed_values),
        "gps": _summarize_gps_values(gps_values),
        "observation_count": len(observations),
    }


def _vote_text_value(values: list[tuple[str, str, float]], source: str) -> dict:
    if not values:
        return {
            "value": None,
            "status": "unknown",
            "confidence": "unknown",
            "source": source,
            "evidence": [],
        }
    counter = Counter(value for value, _, _ in values)
    value, count = counter.most_common(1)[0]
    evidence = [frame_id for candidate, frame_id, _ in values if candidate == value]
    confidence_score = count / len(values)
    confidence = "high" if confidence_score >= 0.75 else "medium" if confidence_score >= 0.5 else "low"
    return {
        "value": value,
        "status": "observed",
        "confidence": confidence,
        "source": source,
        "evidence": evidence,
        "vote_count": count,
        "total_candidates": len(values),
    }


def _summarize_speed_values(values: list[tuple[float, str, float]]) -> dict:
    if not values:
        return {
            "value": "모름",
            "numeric_kmh": None,
            "range_kmh": None,
            "status": "unknown",
            "confidence": "unknown",
            "source": "ocr_overlay",
            "evidence": [],
        }
    candidates = [
        {"frame_id": frame_id, "value": value, "confidence": confidence}
        for value, frame_id, confidence in values
    ]
    filtered_values, rejected_outliers = _filter_speed_outliers(values)
    filtered_values, rejected_jumps = _filter_temporal_speed_jumps(filtered_values)
    rejected_outliers = [*rejected_outliers, *rejected_jumps]
    speeds = [value for value, _, _ in filtered_values]
    selected = float(median(speeds))
    spread = max(speeds) - min(speeds)
    confidence = "high" if len(values) >= 3 and spread <= 3 else "medium" if spread <= 8 else "low"
    return {
        "value": f"{selected:g}km/h",
        "numeric_kmh": selected,
        "range_kmh": [min(speeds), max(speeds)],
        "status": "observed",
        "confidence": confidence,
        "source": "ocr_overlay",
        "evidence": [frame_id for _, frame_id, _ in filtered_values],
        "sample_count": len(filtered_values),
        "candidates": candidates,
        "rejected_outliers": rejected_outliers,
    }


def _filter_speed_outliers(
    values: list[tuple[float, str, float]],
) -> tuple[list[tuple[float, str, float]], list[dict]]:
    plausible = [item for item in values if 0 <= item[0] <= 180]
    rejected = [
        {"frame_id": frame_id, "value": value}
        for value, frame_id, _ in values
        if value < 0 or value > 180
    ]
    if len(plausible) < 3:
        return plausible or values, rejected
    speed_median = float(median([value for value, _, _ in plausible]))
    filtered = []
    for item in plausible:
        value, frame_id, _ = item
        if abs(value - speed_median) <= 35:
            filtered.append(item)
        else:
            rejected.append({"frame_id": frame_id, "value": value})
    return filtered or plausible, rejected


def _filter_temporal_speed_jumps(
    values: list[tuple[float, str, float]],
    max_step_kmh: float = 45.0,
) -> tuple[list[tuple[float, str, float]], list[dict]]:
    if len(values) < 3:
        return values, []
    filtered = [values[0]]
    rejected: list[dict] = []
    for value, frame_id, confidence in values[1:]:
        previous_value = filtered[-1][0]
        if abs(value - previous_value) <= max_step_kmh:
            filtered.append((value, frame_id, confidence))
            continue
        rejected.append(
            {
                "frame_id": frame_id,
                "value": value,
                "reason": "temporal_jump",
                "previous_value": previous_value,
            }
        )
    return filtered if len(filtered) >= 2 else values, rejected


def _summarize_gps_values(values: list[tuple[dict, str, float]]) -> dict:
    if not values:
        return {
            "value": None,
            "status": "unknown",
            "confidence": "unknown",
            "source": "ocr_overlay",
            "evidence": [],
        }
    latitudes = [float(value["lat"]) for value, _, _ in values]
    longitudes = [float(value["lon"]) for value, _, _ in values]
    return {
        "value": {"lat": float(median(latitudes)), "lon": float(median(longitudes))},
        "status": "observed",
        "confidence": "medium" if len(values) >= 2 else "low",
        "source": "ocr_overlay",
        "evidence": [frame_id for _, frame_id, _ in values],
        "sample_count": len(values),
    }
