import cv2
import numpy as np

from accident_vlm.modules import image_io


def test_read_cv_image_caches_by_path_mtime_and_size(monkeypatch, tmp_path) -> None:
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"fake")
    calls = []

    def fake_imread(path, flags):
        calls.append((path, flags))
        return np.full((4, 4, 3), len(calls), dtype=np.uint8)

    image_io.clear_cv_image_cache()
    monkeypatch.setattr(cv2, "imread", fake_imread)

    first = image_io.read_cv_image(image_path)
    second = image_io.read_cv_image(image_path)

    assert len(calls) == 1
    assert first is not second
    assert int(first[0, 0, 0]) == 1
    assert int(second[0, 0, 0]) == 1

    image_path.write_bytes(b"changed")
    third = image_io.read_cv_image(image_path)

    assert len(calls) == 2
    assert int(third[0, 0, 0]) == 2


def test_read_cv_image_cache_returns_mutation_safe_copy(monkeypatch, tmp_path) -> None:
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"fake")

    image_io.clear_cv_image_cache()
    monkeypatch.setattr(
        cv2,
        "imread",
        lambda path, flags: np.full((4, 4, 3), 7, dtype=np.uint8),
    )

    first = image_io.read_cv_image(image_path)
    assert first is not None
    first[:, :] = 99

    second = image_io.read_cv_image(image_path)

    assert second is not None
    assert int(second[0, 0, 0]) == 7


def test_cache_cv_image_primes_read_cache(monkeypatch, tmp_path) -> None:
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"fake")
    image_io.clear_cv_image_cache()

    def fail_imread(path, flags):
        raise AssertionError("primed cache should avoid cv2.imread")

    monkeypatch.setattr(cv2, "imread", fail_imread)

    image_io.cache_cv_image(image_path, np.full((4, 4, 3), 11, dtype=np.uint8))
    image = image_io.read_cv_image(image_path)

    assert image is not None
    assert int(image[0, 0, 0]) == 11
