from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
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
