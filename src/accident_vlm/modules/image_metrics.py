from __future__ import annotations

import numpy as np


def gray_percentile_range(gray: np.ndarray, low_percentile: float = 5.0, high_percentile: float = 95.0) -> float:
    if gray.size == 0:
        return 0.0
    if gray.dtype != np.uint8:
        return float(np.percentile(gray, high_percentile) - np.percentile(gray, low_percentile))

    flat = gray.reshape(-1)
    histogram = np.bincount(flat, minlength=256)
    cumulative = np.cumsum(histogram)
    total = int(cumulative[-1])
    if total <= 0:
        return 0.0
    low_rank = max(0, int(np.ceil(total * low_percentile / 100.0)) - 1)
    high_rank = max(0, int(np.ceil(total * high_percentile / 100.0)) - 1)
    low_value = int(np.searchsorted(cumulative, low_rank + 1))
    high_value = int(np.searchsorted(cumulative, high_rank + 1))
    return float(high_value - low_value)
