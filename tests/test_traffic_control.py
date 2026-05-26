import cv2
import numpy as np

from accident_vlm.modules.traffic_control import _detect_off_signal_candidates, analyze_traffic_control
from accident_vlm.schemas.preprocessing import SelectedFrame


def test_analyze_traffic_control_creates_signal_crop_and_speed_limit_sign(tmp_path) -> None:
    frame_path = tmp_path / "traffic.jpg"
    image = np.zeros((120, 200, 3), dtype=np.uint8)
    cv2.circle(image, (100, 25), 8, (0, 0, 255), -1)
    cv2.imwrite(str(frame_path), image)

    result = analyze_traffic_control(
        [
            SelectedFrame(
                id="frame_000001",
                time="00:00.033",
                frame_index=1,
                purpose="regular_context",
                path=str(frame_path),
            )
        ],
        [
            {
                "frame_id": "frame_000001",
                "text": "제한속도 30",
                "confidence": 0.85,
                "bbox": [10, 20, 50, 60],
                "source": "ocr",
            }
        ],
        output_dir=tmp_path / "traffic",
    )

    assert result["signal"]["value"] == "적색"
    assert result["signal"]["method"] == "traffic_light_hsv_crop_temporal_vote"
    assert result["signal"]["crops"]
    assert result["signal"]["vote_diagnostics"]["vote_count_by_color"]["적색"] >= 1
    assert cv2.imread(result["signal"]["crops"][0]["path"]) is not None
    assert result["signs"][0]["value"] == "제한속도 30"
    assert result["signs"][0]["evidence"] == ["frame_000001"]


def test_analyze_traffic_control_votes_signs_and_saves_failure_cases(tmp_path) -> None:
    frame_path = tmp_path / "no_signal.jpg"
    image = np.zeros((120, 200, 3), dtype=np.uint8)
    cv2.imwrite(str(frame_path), image)

    result = analyze_traffic_control(
        [
            SelectedFrame(
                id="frame_000010",
                time="00:00.333",
                frame_index=10,
                purpose="regular_context",
                path=str(frame_path),
            )
        ],
        [
            {"frame_id": "frame_000010", "text": "제한속도 50", "confidence": 0.70, "source": "ocr"},
            {"frame_id": "frame_000011", "text": "속도제한 50", "confidence": 0.80, "source": "ocr"},
            {"frame_id": "frame_000012", "text": "제한속도 30", "confidence": 0.55, "source": "ocr"},
        ],
        output_dir=tmp_path / "traffic",
    )

    assert result["sign_votes"][0]["value"] == "제한속도 50"
    assert result["sign_votes"][0]["vote_count"] == 2
    assert result["failure_cases"][0]["reason"] == "traffic_light_not_detected"
    assert cv2.imread(result["failure_cases"][0]["path"]) is not None


def test_analyze_traffic_control_classifies_left_turn_signal_and_sign_crops(tmp_path) -> None:
    frame_path = tmp_path / "left_turn.jpg"
    image = np.zeros((160, 240, 3), dtype=np.uint8)
    cv2.circle(image, (80, 30), 7, (0, 0, 255), -1)
    cv2.arrowedLine(image, (155, 30), (130, 30), (0, 255, 0), 5, tipLength=0.45)
    cv2.circle(image, (180, 105), 18, (0, 0, 255), 3)
    cv2.putText(image, "50", (168, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.imwrite(str(frame_path), image)

    result = analyze_traffic_control(
        [
            SelectedFrame(
                id="frame_000020",
                time="00:00.667",
                frame_index=20,
                purpose="insurance_context",
                path=str(frame_path),
            )
        ],
        [{"frame_id": "frame_000020", "text": "제한속도 50", "confidence": 0.88, "source": "ocr"}],
        output_dir=tmp_path / "traffic",
    )

    assert result["signal"]["value"] == "적색+좌회전"
    assert result["signal"]["shape"] == "left_arrow"
    assert result["signal"]["crops"][0]["purpose"] == "traffic_light_crop"
    assert result["sign_crops"]
    assert result["sign_votes"][0]["confidence_margin"] >= 1


def test_analyze_traffic_control_detects_flashing_signal_sequence(tmp_path) -> None:
    frames = []
    for index, lit in enumerate([True, False, True], start=1):
        frame_path = tmp_path / f"flash_{index}.jpg"
        image = np.zeros((120, 200, 3), dtype=np.uint8)
        if lit:
            cv2.circle(image, (100, 25), 8, (0, 255, 0), -1)
        cv2.imwrite(str(frame_path), image)
        frames.append(
            SelectedFrame(
                id=f"frame_00000{index}",
                time=f"00:00.0{index}0",
                frame_index=index,
                purpose="regular_context",
                path=str(frame_path),
            )
        )

    result = analyze_traffic_control(frames, [], output_dir=tmp_path / "traffic")

    assert result["signal"]["value"] == "점멸"
    assert result["signal"]["base_color"] == "녹색"
    assert result["signal"]["frame_sequence"]["missing_frame_count"] == 1


def test_analyze_traffic_control_detects_off_signal_head(tmp_path) -> None:
    frame_path = tmp_path / "off.jpg"
    image = np.zeros((120, 200, 3), dtype=np.uint8)
    cv2.circle(image, (100, 25), 11, (45, 45, 45), -1)
    cv2.circle(image, (100, 25), 13, (90, 90, 90), 2)
    cv2.imwrite(str(frame_path), image)

    result = analyze_traffic_control(
        [
            SelectedFrame(
                id="frame_000100",
                time="00:03.333",
                frame_index=100,
                purpose="regular_context",
                path=str(frame_path),
            )
        ],
        [],
        output_dir=tmp_path / "traffic",
    )

    assert result["signal"]["value"] == "꺼짐"
    assert result["signal"]["visible"] is True
    assert result["signal"]["crops"]


def test_detect_off_signal_candidates_uses_downscaled_hough_input(monkeypatch) -> None:
    image = np.full((720, 1280, 3), 210, dtype=np.uint8)
    cv2.circle(image, (640, 150), 28, (45, 45, 45), -1)
    shapes = []
    original_hough = cv2.HoughCircles

    def record_hough(gray, *args, **kwargs):
        shapes.append(gray.shape)
        return original_hough(gray, *args, **kwargs)

    monkeypatch.setattr(cv2, "HoughCircles", record_hough)

    candidates = _detect_off_signal_candidates(image)

    assert candidates
    assert shapes
    assert all(width <= 320 for _height, width in shapes)
