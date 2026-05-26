import numpy as np

from accident_vlm.modules.image_metrics import gray_percentile_range


def test_gray_percentile_range_uses_uint8_histogram() -> None:
    gray = np.array([[0, 0, 10, 10], [200, 200, 255, 255]], dtype=np.uint8)

    assert gray_percentile_range(gray) == 255.0


def test_gray_percentile_range_handles_empty_image() -> None:
    assert gray_percentile_range(np.array([], dtype=np.uint8)) == 0.0
