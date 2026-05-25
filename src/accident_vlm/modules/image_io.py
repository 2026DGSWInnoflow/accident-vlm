from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np


_IMAGE_CACHE: dict[tuple[str, int, int, int], np.ndarray] = {}


def read_cv_image(path: Any, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    if not path:
        return None
    image_path = Path(str(path))
    try:
        stat = image_path.stat()
    except OSError:
        return None

    max_cached = _image_cache_size()
    if max_cached <= 0:
        return cv2.imread(str(image_path), flags)

    cache_key = (
        str(image_path.resolve()),
        int(stat.st_mtime_ns),
        int(stat.st_size),
        int(flags),
    )
    cached = _IMAGE_CACHE.get(cache_key)
    if cached is not None:
        return cached.copy()

    image = cv2.imread(str(image_path), flags)
    if image is None:
        return None
    _IMAGE_CACHE[cache_key] = image
    while len(_IMAGE_CACHE) > max_cached:
        _IMAGE_CACHE.pop(next(iter(_IMAGE_CACHE)))
    return image.copy()


def clear_cv_image_cache() -> None:
    _IMAGE_CACHE.clear()


def _image_cache_size() -> int:
    try:
        return max(0, int(os.getenv("ACCIDENT_VLM_CV_IMAGE_CACHE_SIZE", "64")))
    except ValueError:
        return 64
