from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from statistics import median
from typing import Protocol

import cv2

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


def parse_overlay_text(text: str) -> dict:
    datetime_match = re.search(r"\d{4}[-./]\d{2}[-./]\d{2}\s+\d{2}:\d{2}:\d{2}", text)
    speed_match = re.search(r"(\d{1,3}(?:\.\d+)?)\s*(?:km/h|kph|KM/H)", text, re.IGNORECASE)
    gps_match = re.search(r"(-?\d{1,3}\.\d+)[,\s]+(-?\d{1,3}\.\d+)", text)
    parsed: dict = {}
    if datetime_match:
        parsed["datetime"] = datetime_match.group(0).replace(".", "-").replace("/", "-")
    if speed_match:
        parsed["speed_kmh"] = float(speed_match.group(1))
    if gps_match:
        parsed["gps"] = {"lat": float(gps_match.group(1)), "lon": float(gps_match.group(2))}
    return parsed


def extract_ocr_observations(
    selected_frames: list[SelectedFrame],
    backend: OcrBackend,
) -> list[dict]:
    observations: list[dict] = []
    for frame in selected_frames:
        if not frame.path:
            continue
        for item in backend.read_text(Path(frame.path)):
            observation = {
                "time": frame.time,
                "frame_id": frame.id,
                **item,
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
    speeds = [value for value, _, _ in values]
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
        "evidence": [frame_id for _, frame_id, _ in values],
        "sample_count": len(values),
    }


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
